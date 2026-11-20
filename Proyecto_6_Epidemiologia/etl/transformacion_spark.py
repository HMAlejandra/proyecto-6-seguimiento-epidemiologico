import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    LongType,
    DoubleType,
    DateType,
    StringType,
)
from pyspark.sql.window import Window

from etl.spark_config import get_spark_session
from etl.extraccion import RAW_GOOGLE_DIR

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = PROJECT_ROOT / "data" / "staging"


# ---------------------------------------------------------------------------
# Columnas conocidas de los archivos de Google Cloud Platform v3
# ---------------------------------------------------------------------------

# Columnas epidemiológicas (conteos — NO son tasas)
EPIDEMIOLOGY_COLS = {
    "new_confirmed": "long",
    "new_deceased": "long",
    "new_recovered": "long",
    "new_tested": "long",
    "cumulative_confirmed": "long",
    "cumulative_deceased": "long",
    "cumulative_recovered": "long",
    "cumulative_tested": "long",
}

# Columnas por edad
BY_AGE_COLS = {
    "age_bucket": "string",
    "cumulative_confirmed": "long",
    "cumulative_deceased": "long",
    "cumulative_tested": "long",
    "new_confirmed": "long",
    "new_deceased": "long",
}

# Columnas demográficas (nombres reales del archivo)
DEMOGRAPHICS_COLS = {
    "population": "long",
    "population_male": "long",
    "population_female": "long",
    "population_rural": "long",
    "population_urban": "long",
    "population_density": "double",
    "human_development_index": "double",
    "population_age_00_09": "double",
    "population_age_10_19": "double",
    "population_age_20_29": "double",
    "population_age_30_39": "double",
    "population_age_40_49": "double",
    "population_age_50_59": "double",
    "population_age_60_69": "double",
    "population_age_70_79": "double",
    "population_age_80_and_older": "double",
}

# Columnas climáticas (nombres reales del archivo)
WEATHER_COLS = {
    "average_temperature_celsius": "double",
    "minimum_temperature_celsius": "double",
    "maximum_temperature_celsius": "double",
    "rainfall_mm": "double",
    "snowfall_mm": "double",
    "dew_point": "double",
    "relative_humidity": "double",
}

# Columnas de hospitalizaciones (nombres reales del archivo)
HOSPITALIZATIONS_COLS = {
    "new_hospitalized_patients": "long",
    "cumulative_hospitalized_patients": "long",
    "current_hospitalized_patients": "long",
    "new_intensive_care_patients": "long",
    "cumulative_intensive_care_patients": "long",
    "current_intensive_care_patients": "long",
    "new_ventilator_patients": "long",
    "cumulative_ventilator_patients": "long",
    "current_ventilator_patients": "long",
}


# ---------------------------------------------------------------------------
# Funciones auxiliares de transformación
# ---------------------------------------------------------------------------

def leer_csv_crudo(spark, nombre_archivo: str) -> Optional[DataFrame]:
  
    ruta = RAW_GOOGLE_DIR / nombre_archivo
    if not ruta.exists():
        logger.error("Archivo no encontrado: %s", ruta)
        return None

    df = (
        spark.read
        .option("header", True)
        # inferSchema=False para controlar los tipos manualmente
        .option("inferSchema", False)
        .csv(str(ruta))
    )
    logger.info("Archivo %s cargado: %d columnas", nombre_archivo, len(df.columns))
    return df


def convertir_tipos(df: DataFrame, columnas_tipo: dict) -> DataFrame:
    
    tipo_map = {
        "long": LongType(),
        "int": IntegerType(),
        "double": DoubleType(),
        "string": StringType(),
    }

    for col_name, tipo_str in columnas_tipo.items():
        if col_name not in df.columns:
            continue

        if tipo_str == "date":
            # Convertir string a fecha
            df = df.withColumn(col_name, F.to_date(F.col(col_name), "yyyy-MM-dd"))
        elif tipo_str in tipo_map:
            # 1. Guardar la columna original (string) ANTES del cast.
            orig_col = f"_orig_{col_name}"
            df = df.withColumn(orig_col, F.col(col_name))

            # 2. Castear al tipo destino.
            df = df.withColumn(col_name, F.col(col_name).cast(tipo_map[tipo_str]))

            flag_col = f"_cast_failed_{col_name}"
            df = df.withColumn(
                flag_col,
                F.col(orig_col).isNotNull() & F.col(col_name).isNull()
            )

            logger.info(
                "Cast de '%s' a %s programado. Flag '%s' añadida para detectar "
                "filas no convertibles (se evalúa lazily, sin count() extra).",
                col_name, tipo_str, flag_col,
            )

            # 4. Eliminar la columna temporal original.
            df = df.drop(orig_col)

    return df


def eliminar_duplicados(df: DataFrame, subset: list = None,
                       orden_por: str = None,
                       contar_eliminados: bool = False) -> DataFrame:
    
    if contar_eliminados:
        total_antes = df.count()

    if subset is None:
        df_limpio = df.dropDuplicates()
    elif orden_por is not None and orden_por in df.columns:
        # Usar ventana para conservar la fila con el valor más alto en orden_por
        w = Window.partitionBy(*subset).orderBy(F.desc_nulls_last(F.col(orden_por)))
        df_limpio = (
            df.withColumn("_row_num", F.row_number().over(w))
            .filter(F.col("_row_num") == 1)
            .drop("_row_num")
        )
    else:
        df_limpio = df.dropDuplicates(subset)

    if contar_eliminados:
        total_despues = df_limpio.count()
        eliminados = total_antes - total_despues
        if eliminados > 0:
            logger.info("Duplicados eliminados: %d (de %d a %d)", eliminados, total_antes, total_despues)
        else:
            logger.info("No se encontraron duplicados.")
    else:
        logger.info("Deduplicación completada (contar_eliminados=False — sin count() adicionales).")

    return df_limpio


def validar_no_negativos(df: DataFrame, columnas_numericas: list) -> DataFrame:
  
    for col_name in columnas_numericas:
        if col_name not in df.columns:
            continue
        negativos = df.filter(F.col(col_name) < 0).count()
        if negativos > 0:
            logger.warning(
                "Columna '%s': %d registros con valores negativos. "
                "Se conservan para análisis (pueden ser correcciones).",
                col_name, negativos,
            )
    return df


def filtrar_colombia(df: DataFrame, col_location: str = "location_key") -> DataFrame:
    
    if col_location not in df.columns:
        logger.warning("Columna '%s' no encontrada. No se puede filtrar por Colombia.", col_location)
        return df

    df_col = df.filter(F.col(col_location).startswith("CO"))
    total = df_col.count()
    logger.info("Registros de Colombia: %d", total)
    return df_col


def filtrar_nivel_geografico(df: DataFrame, nivel: str = "pais",
                             col_location: str = "location_key") -> DataFrame:
    
    if col_location not in df.columns:
        logger.warning("Columna '%s' no encontrada. No se puede filtrar por nivel.", col_location)
        return df

    num_guiones = F.size(F.split(F.col(col_location), "_")) - 1

    if nivel == "pais":
        df_filtrado = df.filter(num_guiones == 0)
    elif nivel == "subregion":
        df_filtrado = df.filter(num_guiones == 1)
    elif nivel == "municipio":
        df_filtrado = df.filter(num_guiones == 2)
    else:
        logger.warning("Nivel '%s' no reconocido. Opciones: pais, subregion, municipio.", nivel)
        return df

    total = df_filtrado.count()
    logger.info("Filtrado a nivel '%s': %d registros.", nivel, total)
    return df_filtrado

_AGE_BUCKET_NORMALIZACION = {
    "80-":    "80+",
    "80-89":  "80+",
    "80-99":  "80+",
    "90+":    "80+",
    "90-":    "80+",
    "90-99":  "80+",
    "100+":   "80+",
}

def _expr_normalizar_age_bucket(col_name: str = "age_bucket"):

    expr = None
    for variante, canonica in _AGE_BUCKET_NORMALIZACION.items():
        condicion = F.col(col_name) == F.lit(variante)
        if expr is None:
            expr = F.when(condicion, F.lit(canonica))
        else:
            expr = expr.when(condicion, F.lit(canonica))
    # Para cualquier etiqueta no listada en el mapa, conservar el valor original
    return expr.otherwise(F.col(col_name)) if expr is not None else F.col(col_name)



def transformar_epidemiologia(spark) -> Optional[DataFrame]:
   
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: epidemiology.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "epidemiology.csv")
    if df is None:
        return None

    # Convertir tipos
    tipos = {"date": "date"}
    tipos.update(EPIDEMIOLOGY_COLS)
    df = convertir_tipos(df, tipos)

    if "date" in df.columns:
        df = df.filter(
            F.col("date").isNull() | (F.col("date") >= F.lit("2019-12-01").cast("date"))
        )

    # Eliminar duplicados (misma fecha + misma ubicación)
    if "location_key" in df.columns and "date" in df.columns:
        df = eliminar_duplicados(df, subset=["date", "location_key"])

    if "location_key" in df.columns:
        antes = df
        df = df.filter(
            F.col("location_key").isNotNull() &
            (F.length(F.trim(F.col("location_key"))) > 0)
        )
        logger.info("Registros sin location_key eliminados (nulos o vacíos).")

    # Validar valores negativos en columnas de conteo
    validar_no_negativos(df, list(EPIDEMIOLOGY_COLS.keys()))

    logger.info("Transformación de epidemiología completada.")
    return df


def transformar_by_age(spark) -> Optional[DataFrame]:
 
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: by-age.csv (WIDE -> LONG con stack)")
    logger.info("=" * 60)

    # Leer solo las columnas necesarias (no las 152 columnas completas)
    ruta = RAW_GOOGLE_DIR / "by-age.csv"
    if not ruta.exists():
        logger.error("Archivo no encontrado: %s", ruta)
        return None

    # Seleccionar solo las columnas que necesitamos para el unpivot
    cols_needed = ["date", "location_key"]
    for i in range(10):
        cols_needed.append(f"age_bin_{i}")
        cols_needed.append(f"new_confirmed_age_{i}")
        cols_needed.append(f"new_deceased_age_{i}")
        cols_needed.append(f"cumulative_confirmed_age_{i}")
        cols_needed.append(f"cumulative_deceased_age_{i}")

    # Leer el CSV y seleccionar solo las columnas que existen
    df_all = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(str(ruta))
    )
    cols_present = [c for c in cols_needed if c in df_all.columns]
    df = df_all.select(*cols_present)

    logger.info("Archivo by-age.csv cargado: %d columnas seleccionadas (de 152 totales)", len(df.columns))

    # Convertir date a tipo fecha
    df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

    
    NULL_STR = F.lit(None).cast("string")
    for i in range(10):
        for col_opcional in [f"new_deceased_age_{i}",
                             f"cumulative_confirmed_age_{i}",
                             f"cumulative_deceased_age_{i}"]:
            if col_opcional not in df.columns:
                df = df.withColumn(col_opcional, NULL_STR)
                logger.debug("Columna '%s' ausente en by-age.csv — añadida como NULL.", col_opcional)

    stack_args = []
    for i in range(10):
        age_bin = f"age_bin_{i}"
        new_conf = f"new_confirmed_age_{i}"
        new_dec = f"new_deceased_age_{i}"
        cum_conf = f"cumulative_confirmed_age_{i}"
        cum_dec = f"cumulative_deceased_age_{i}"
        if age_bin in df.columns and new_conf in df.columns:
            stack_args.extend([age_bin, new_conf, new_dec, cum_conf, cum_dec])

    num_cols_per_group = 5  # age_bucket, new_confirmed, new_deceased, cumulative_confirmed, cumulative_deceased
    num_groups = len(stack_args) // num_cols_per_group

    # Crear vista temporal y usar SQL stack
    df.createOrReplaceTempView("by_age_raw")

    stack_expr = ", ".join(stack_args)
    sql_query = f"""
        SELECT 
            date, 
            location_key, 
            stack({num_groups}, {stack_expr}) AS (age_bucket, new_confirmed, new_deceased, cumulative_confirmed, cumulative_deceased)
        FROM by_age_raw
    """

    df_long = spark.sql(sql_query)

    # Filtrar filas donde age_bucket es nulo
    df_long = df_long.filter(F.col("age_bucket").isNotNull())

    df_long = df_long.withColumn("age_bucket", _expr_normalizar_age_bucket("age_bucket"))
    logger.info("Etiquetas de age_bucket normalizadas (variantes '80-x' → '80+').")

    # Convertir columnas numéricas de string a long
    # (el CSV se leyó con inferSchema=False, así que todas son strings)
    df_long = df_long.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))
    df_long = df_long.withColumn("new_deceased", F.col("new_deceased").cast("long"))
    df_long = df_long.withColumn("cumulative_confirmed", F.col("cumulative_confirmed").cast("long"))
    df_long = df_long.withColumn("cumulative_deceased", F.col("cumulative_deceased").cast("long"))

    df_long = df_long.filter(
        F.col("date").isNull() | (F.col("date") >= F.lit("2019-12-01").cast("date"))
    )

    # Eliminar duplicados (fecha + ubicación + grupo etario)
    df_long = eliminar_duplicados(df_long, subset=["date", "location_key", "age_bucket"])

    # Validar negativos
    validar_no_negativos(df_long, ["new_confirmed", "new_deceased",
                                   "cumulative_confirmed", "cumulative_deceased"])

    logger.info("Transformacion de by-age completada (WIDE -> LONG con stack): %d filas.", df_long.count())
    return df_long


def transformar_demographics(spark) -> Optional[DataFrame]:
   
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: demographics.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "demographics.csv")
    if df is None:
        return None

    tipos = {"location_key": "string"}
    tipos.update(DEMOGRAPHICS_COLS)
    df = convertir_tipos(df, tipos)

    # Eliminar duplicados por location_key
    if "location_key" in df.columns:
        df = eliminar_duplicados(df, subset=["location_key"])

    logger.info("Transformación de demographics completada.")
    return df


def transformar_index(spark) -> Optional[DataFrame]:
   
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: index.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "index.csv")
    if df is None:
        return None

    # Eliminar duplicados por location_key
    if "location_key" in df.columns:
        df = eliminar_duplicados(df, subset=["location_key"])

    logger.info("Transformación de index completada.")
    return df


def transformar_weather(spark) -> Optional[DataFrame]:
   
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: weather.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "weather.csv")
    if df is None:
        return None

    tipos = {"date": "date", "location_key": "string"}
    tipos.update(WEATHER_COLS)
    df = convertir_tipos(df, tipos)

    # Eliminar fechas anteriores al inicio de la pandemia.
    if "date" in df.columns:
        df = df.filter(
            F.col("date").isNull() | (F.col("date") >= F.lit("2019-12-01").cast("date"))
        )

    # Eliminar duplicados
    subset = [c for c in ["date", "location_key"] if c in df.columns]
    if subset:
        df = eliminar_duplicados(df, subset=subset)

    logger.info("Transformación de weather completada.")
    return df


def transformar_hospitalizations(spark) -> Optional[DataFrame]:
    
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: hospitalizations.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "hospitalizations.csv")
    if df is None:
        return None

    tipos = {"date": "date", "location_key": "string"}
    tipos.update(HOSPITALIZATIONS_COLS)
    df = convertir_tipos(df, tipos)

    # Filtrar fechas absurdas (año < 2019) producidas por parsing incorrecto en la fuente.
    # También se eliminan fechas anteriores a 2019-12-01 (inicio de la pandemia).
    if "date" in df.columns:
        df = df.filter(
            F.col("date").isNull() | (F.col("date") >= F.lit("2019-12-01").cast("date"))
        )

    # Eliminar duplicados
    subset = [c for c in ["date", "location_key"] if c in df.columns]
    if subset:
        df = eliminar_duplicados(df, subset=subset)

    # Validar negativos
    validar_no_negativos(df, list(HOSPITALIZATIONS_COLS.keys()))

    logger.info("Transformación de hospitalizations completada.")
    return df


def integrar_con_index(df: DataFrame, df_index: DataFrame) -> DataFrame:
   
    if "location_key" not in df.columns or "location_key" not in df_index.columns:
        logger.warning("No se puede integrar: falta la columna location_key.")
        return df

    # Seleccionar solo las columnas útiles del index
    cols_index = ["location_key"]
    for c in ["country_name", "country_code", "subregion1_name", "subregion1_code"]:
        if c in df_index.columns:
            cols_index.append(c)

    df_index_sel = df_index.select(*cols_index)

    df_integrado = df.join(df_index_sel, on="location_key", how="left")

    logger.info("Integración con index completada. Columnas: %s", df_integrado.columns)
    return df_integrado


def integrar_con_demographics(df: DataFrame, df_demographics: DataFrame) -> DataFrame:
    
    if "location_key" not in df.columns or "location_key" not in df_demographics.columns:
        logger.warning("No se puede integrar con demographics: falta location_key.")
        return df

    # Tomar solo population y location_key
    cols_demo = ["location_key"]
    if "population" in df_demographics.columns:
        cols_demo.append("population")
    if "population_density" in df_demographics.columns:
        cols_demo.append("population_density")

    df_demo_sel = df_demographics.select(*cols_demo)

    df_integrado = df.join(df_demo_sel, on="location_key", how="left")

    logger.info("Integración con demographics completada.")
    return df_integrado


def calcular_promedio_casos_por_pais(df_epidemiologia: DataFrame) -> DataFrame:
   
    logger.info("Calculando promedio de casos nuevos por país...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    
    agg = (
        df_pais
        .filter(
            F.col("new_confirmed").isNotNull() &
            (F.col("new_confirmed") > 0) &
            F.col("country_name").isNotNull()
        )
        .groupBy("country_name")
        .agg(
            F.avg("new_confirmed").alias("avg_new_confirmed"),
            F.sum("new_confirmed").alias("total_new_confirmed"),
            F.count("new_confirmed").alias("dias_con_contagio"),
            F.max("new_confirmed").alias("max_new_confirmed"),
            F.min("new_confirmed").alias("min_new_confirmed"),
        )
        .orderBy(F.desc("total_new_confirmed"))
    )

    return agg


def calcular_distribucion_por_edad(df_by_age: DataFrame) -> DataFrame:
   
    logger.info("Calculando distribución de contagios por grupo etario...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_by_age, nivel="pais")

    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(F.sum("new_confirmed").alias("total_new_confirmed"))
    if "new_deceased" in df_pais.columns:
        agg_cols.append(F.sum("new_deceased").alias("total_new_deceased"))

    if "age_bucket" not in df_pais.columns:
        logger.warning("Columna 'age_bucket' no encontrada en by-age.")
        return df_pais

    filtros = F.col("age_bucket").isNotNull()
    if "new_confirmed" in df_pais.columns:
        filtros = filtros & (F.col("new_confirmed").isNull() | (F.col("new_confirmed") >= 0))
    if "new_deceased" in df_pais.columns:
        filtros = filtros & (F.col("new_deceased").isNull() | (F.col("new_deceased") >= 0))

    agg = (
        df_pais
        .filter(filtros)
        .groupBy("age_bucket")
        .agg(*agg_cols)
        .orderBy("age_bucket")
    )

    return agg


def calcular_evolucion_temporal(df_epidemiologia: DataFrame) -> DataFrame:
   
    logger.info("Calculando evolución temporal de contagios...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    # Excluir negativos antes de agregar.
    filtros = F.col("date").isNotNull()
    if "new_confirmed" in df_pais.columns:
        filtros = filtros & (F.col("new_confirmed").isNull() | (F.col("new_confirmed") >= 0))
    if "new_deceased" in df_pais.columns:
        filtros = filtros & (F.col("new_deceased").isNull() | (F.col("new_deceased") >= 0))

    df_filtrado = df_pais.filter(filtros)

    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(F.sum("new_confirmed").alias("total_new_confirmed"))
    if "new_deceased" in df_pais.columns:
        agg_cols.append(F.sum("new_deceased").alias("total_new_deceased"))

    if "cumulative_confirmed" in df_pais.columns and "location_key" in df_pais.columns:
        
        agg_cols.append(
            F.sum(
                F.when(
                    F.col("cumulative_confirmed").isNotNull() &
                    (F.col("cumulative_confirmed") >= 0),
                    F.col("cumulative_confirmed")
                )
            ).alias("total_cumulative_confirmed")
        )
        logger.info(
            "total_cumulative_confirmed = suma del acumulado de cada país en cada fecha "
            "(correcto porque hay 1 fila por país gracias a filtrar_nivel_geografico)."
        )

    agg = (
        df_filtrado
        .groupBy("date")
        .agg(*agg_cols)
        .orderBy("date")
    )

    return agg


def calcular_tasa_contagio_por_pais(df_epidemiologia: DataFrame,
                                    df_demographics: DataFrame) -> DataFrame:
    
    logger.info("Calculando tasa de contagio por país...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

   
    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(
            F.sum(
                F.when(F.col("new_confirmed") >= 0, F.col("new_confirmed"))
            ).alias("total_confirmed")
        )
    if "cumulative_confirmed" in df_pais.columns:
        # Se conserva el acumulado máximo como referencia informativa,
        # pero NO se usa para calcular la tasa.
        agg_cols.append(F.max("cumulative_confirmed").alias("max_cumulative_confirmed"))

    if "country_name" not in df_pais.columns:
        logger.warning("Falta country_name en epidemiologia. No se puede calcular tasa por país.")
        return df_pais

    group_cols = ["location_key", "country_name"]
    if "country_code" in df_pais.columns:
        group_cols.append("country_code")

    df_agg = (
        df_pais
        .filter(
            F.col("country_name").isNotNull() &
            F.col("location_key").isNotNull()
        )
        .groupBy(*group_cols)
        .agg(*agg_cols)
    )

    if "population" in df_demographics.columns and "location_key" in df_demographics.columns:
        df_pop = (
            filtrar_nivel_geografico(df_demographics, nivel="pais")
            .select("location_key", "population")
            .filter(F.col("population").isNotNull() & (F.col("population") > 0))
            .distinct()
        )

        if "location_key" in df_agg.columns:
            # Join directo por location_key — el más exacto
            df_agg = df_agg.join(df_pop, on="location_key", how="left")
            logger.info("Join con demographics realizado por location_key.")
        elif "country_code" in df_agg.columns:
            # Fallback: usar country_code como proxy de location_key de nivel país
            nulos_cc = df_agg.filter(F.col("country_code").isNull()).count()
            if nulos_cc > 0:
                logger.warning(
                    "country_code tiene %d nulos en df_agg. "
                    "Los países sin country_code quedarán sin población y sin tasa.",
                    nulos_cc,
                )
            df_agg = df_agg.join(
                df_pop,
                df_agg["country_code"] == df_pop["location_key"],
                how="left"
            ).drop(df_pop["location_key"])
            logger.info("Join con demographics realizado por country_code (fallback).")
        else:
            logger.warning(
                "Ni location_key ni country_code disponibles en df_agg. "
                "No se puede unir con demographics; tasa no calculable."
            )

    # Calcular tasa por 100 mil habitantes
    if "total_confirmed" in df_agg.columns and "population" in df_agg.columns:
        df_agg = df_agg.withColumn(
            "tasa_por_100k_habitantes",
            F.when(F.col("population") > 0,
                   (F.col("total_confirmed") / F.col("population")) * 100000)
            .otherwise(None)
        )
        df_agg = df_agg.orderBy(F.desc("tasa_por_100k_habitantes"))

    return df_agg


def calcular_top_10_paises_tasa(df_tasa: DataFrame) -> DataFrame:
    
    logger.info("Calculando top 10 países por tasa de contagio...")

    if "tasa_por_100k_habitantes" not in df_tasa.columns:
        logger.warning("Columna 'tasa_por_100k_habitantes' no encontrada.")
        return df_tasa

    top10 = (
        df_tasa
        .filter(F.col("tasa_por_100k_habitantes").isNotNull())
        .orderBy(F.desc("tasa_por_100k_habitantes"))
        .limit(10)
    )

    return top10


def calcular_relacion_casos_fallecimientos(df_epidemiologia: DataFrame) -> DataFrame:
    
    logger.info("Calculando relación entre casos nuevos y fallecimientos...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(
            F.sum(F.when(F.col("new_confirmed") > 0, F.col("new_confirmed")))
            .alias("total_new_confirmed")
        )
    if "new_deceased" in df_pais.columns:
      
        agg_cols.append(
            F.sum(F.when(F.col("new_deceased") > 0, F.col("new_deceased")))
            .alias("total_new_deceased")
        )

    if "country_name" not in df_pais.columns:
        logger.warning("Falta country_name. No se puede calcular relación por país.")
        return df_pais

    # Excluir negativos en ambas columnas para que la tasa de letalidad
    # no se distorsione por correcciones contables.
    filtros_letal = F.col("country_name").isNotNull()
    if "new_confirmed" in df_pais.columns:
        filtros_letal = filtros_letal & (
            F.col("new_confirmed").isNull() | (F.col("new_confirmed") >= 0)
        )
    if "new_deceased" in df_pais.columns:
        filtros_letal = filtros_letal & (
            F.col("new_deceased").isNull() | (F.col("new_deceased") >= 0)
        )

    df_agg = (
        df_pais
        .filter(filtros_letal)
        .groupBy("country_name")
        .agg(*agg_cols)
    )

    # Calcular tasa de letalidad solo para países con datos de fallecimientos.
    # Si total_new_deceased es nulo significa que el país nunca reportó fallecimientos;
    # en ese caso la tasa queda nula (sin dato), no 0%.
    if "total_new_confirmed" in df_agg.columns and "total_new_deceased" in df_agg.columns:
        df_agg = df_agg.withColumn(
            "tasa_letalidad_pct",
            F.when(
                F.col("total_new_confirmed") > 0,
                (F.col("total_new_deceased") / F.col("total_new_confirmed")) * 100
            ).otherwise(None)
        )
        # Ordenar: países con tasa calculada primero (nulos al final)
        df_agg = df_agg.orderBy(F.desc_nulls_last("tasa_letalidad_pct"))

    return df_agg


def calcular_relacion_hospitalizaciones(df_epidemiologia: DataFrame,
                                        df_hospitalizaciones: DataFrame) -> DataFrame:
    logger.info("Calculando relación entre hospitalizaciones y contagios...")

    # Filtrar a nivel país para evitar doble conteo en ambos datasets
    df_epi_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")
    df_hosp_pais = filtrar_nivel_geografico(df_hospitalizaciones, nivel="pais")

    if "new_confirmed" not in df_epi_pais.columns:
        return df_epi_pais

    hosp_col_principal = None
    for c in ["new_hospitalized_patients", "current_hospitalized_patients"]:
        if c in df_hosp_pais.columns:
            hosp_col_principal = c
            break

    if hosp_col_principal is None:
        logger.warning("No se encontraron columnas de hospitalizaciones.")
        return df_epi_pais

    # Identificar los location_keys que tienen al menos un dato de hospitalización.
    # El ratio solo tiene sentido sobre este subconjunto de países.
    location_keys_con_hosp = (
        df_hosp_pais
        .filter(F.col(hosp_col_principal).isNotNull() & (F.col(hosp_col_principal) > 0))
        .select("location_key")
        .distinct()
    )
    n_paises = location_keys_con_hosp.count()
    logger.info(
        "Países con datos de hospitalización: %d. "
        "El ratio se calculará solo sobre estos países.",
        n_paises,
    )

    # Restringir epidemiología a los mismos países con datos de hospitalización.
    # Sin este filtro, el denominador del ratio incluye casos de ~200 países
    # mientras el numerador solo cubre ~20, produciendo un ratio subestimado.
    df_epi_restringida = df_epi_pais.join(
        location_keys_con_hosp, on="location_key", how="inner"
    )

    # Agregar epidemiología restringida por fecha
    epi_agg = (
        df_epi_restringida
        .filter(
            F.col("date").isNotNull() &
            (F.col("new_confirmed").isNull() | (F.col("new_confirmed") >= 0))
        )
        .groupBy("date")
        .agg(
            F.sum("new_confirmed").alias("total_new_confirmed"),
            F.countDistinct("location_key").alias("n_paises_con_hospitalizaciones"),
        )
    )

    # Agregar hospitalizaciones por fecha (excluyendo negativos)
    hosp_agg_cols = []
    if "new_hospitalized_patients" in df_hosp_pais.columns:
        hosp_agg_cols.append(
            F.sum(F.when(F.col("new_hospitalized_patients") >= 0,
                         F.col("new_hospitalized_patients")))
            .alias("total_new_hospitalized")
        )
    if "new_intensive_care_patients" in df_hosp_pais.columns:
        hosp_agg_cols.append(
            F.sum(F.when(F.col("new_intensive_care_patients") >= 0,
                         F.col("new_intensive_care_patients")))
            .alias("total_new_intensive_care")
        )
    if "current_hospitalized_patients" in df_hosp_pais.columns:
        hosp_agg_cols.append(
            F.sum(F.when(F.col("current_hospitalized_patients") >= 0,
                         F.col("current_hospitalized_patients")))
            .alias("total_current_hospitalized")
        )

    hosp_agg = df_hosp_pais.groupBy("date").agg(*hosp_agg_cols)

    # Unir por fecha
    df_join = epi_agg.join(hosp_agg, on="date", how="inner")

    # Calcular ratio sobre el conjunto comparable de países
    if "total_new_confirmed" in df_join.columns and "total_new_hospitalized" in df_join.columns:
        df_join = df_join.withColumn(
            "ratio_hospitalizados",
            F.when(F.col("total_new_confirmed") > 0,
                   F.col("total_new_hospitalized") / F.col("total_new_confirmed"))
            .otherwise(None)
        )

    df_join = df_join.orderBy("date")
    return df_join


def calcular_distribucion_edad_sexo(df_by_age: DataFrame) -> DataFrame:
    logger.info("Calculando distribución por edad (sexo no disponible en la fuente)...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_by_age, nivel="pais")

    if "age_bucket" not in df_pais.columns:
        logger.warning("Columna 'age_bucket' no encontrada.")
        return df_pais

    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(F.sum("new_confirmed").alias("total_new_confirmed"))
    if "new_deceased" in df_pais.columns:
        agg_cols.append(F.sum("new_deceased").alias("total_new_deceased"))

  
    filtros_sexo = F.col("age_bucket").isNotNull()
    if "new_confirmed" in df_pais.columns:
        filtros_sexo = filtros_sexo & (F.col("new_confirmed").isNull() | (F.col("new_confirmed") >= 0))
    if "new_deceased" in df_pais.columns:
        filtros_sexo = filtros_sexo & (F.col("new_deceased").isNull() | (F.col("new_deceased") >= 0))

    agg = (
        df_pais
        .filter(filtros_sexo)
        .groupBy("age_bucket")
        .agg(*agg_cols)
        .orderBy("age_bucket")
    )

    return agg


def calcular_relacion_clima_contagios(df_epidemiologia: DataFrame,
                                       df_weather: DataFrame) -> DataFrame:
   
    logger.info("Preparando datos de relación clima-contagios...")

    # Filtrar a nivel país para evitar mezclar granularidades
    df_epi_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")
    df_weather_pais = filtrar_nivel_geografico(df_weather, nivel="pais")

    # Filtrar columnas relevantes
    epi_cols = ["date", "location_key"]
    if "new_confirmed" in df_epi_pais.columns:
        epi_cols.append("new_confirmed")

    weather_cols = ["date", "location_key"]
    for c in ["average_temperature_celsius", "relative_humidity", "rainfall_mm"]:
        if c in df_weather_pais.columns:
            weather_cols.append(c)

    df_epi_sel = df_epi_pais.select(*[c for c in epi_cols if c in df_epi_pais.columns])
    df_weather_sel = df_weather_pais.select(*[c for c in weather_cols if c in df_weather_pais.columns])

    df_join = df_epi_sel.join(df_weather_sel, on=["date", "location_key"], how="left")

    logger.info("Datos clima-contagios preparados: %d filas.", df_join.count())
    return df_join


# ---------------------------------------------------------------------------
# Orquestación de todas las transformaciones
# ---------------------------------------------------------------------------

def ejecutar_transformaciones(spark) -> dict:
    
    logger.info("=" * 60)
    logger.info("INICIANDO TRANSFORMACIONES CON SPARK")
    logger.info("=" * 60)

    resultados = {}

    # 1. Transformar archivos individuales
    df_index = transformar_index(spark)
    df_demo = transformar_demographics(spark)
    df_epi = transformar_epidemiologia(spark)
    df_age = transformar_by_age(spark)
    df_weather = transformar_weather(spark)
    df_hosp = transformar_hospitalizations(spark)

    # 2. Integrar epidemiología con index para obtener nombres de país
    if df_epi is not None and df_index is not None:
        df_epi = integrar_con_index(df_epi, df_index)
        logger.info("Epidemiología integrada con index.")

    # 3. Guardar DataFrames base para uso posterior
    resultados["df_index"] = df_index
    resultados["df_demographics"] = df_demo
    resultados["df_epidemiologia"] = df_epi
    resultados["df_by_age"] = df_age
    resultados["df_weather"] = df_weather
    resultados["df_hospitalizations"] = df_hosp

    # 4. Generar agregaciones para responder las preguntas del proyecto
    if df_epi is not None:
        resultados["promedio_casos_por_pais"] = calcular_promedio_casos_por_pais(df_epi)
        resultados["evolucion_temporal"] = calcular_evolucion_temporal(df_epi)
        resultados["relacion_casos_fallecimientos"] = calcular_relacion_casos_fallecimientos(df_epi)

        if df_demo is not None:
            resultados["tasa_contagio_por_pais"] = calcular_tasa_contagio_por_pais(df_epi, df_demo)
            if "tasa_contagio_por_pais" in resultados:
                resultados["top_10_paises_tasa"] = calcular_top_10_paises_tasa(
                    resultados["tasa_contagio_por_pais"]
                )

    if df_age is not None:
        resultados["distribucion_por_edad"] = calcular_distribucion_por_edad(df_age)
        resultados["distribucion_edad_sexo"] = calcular_distribucion_edad_sexo(df_age)

    if df_epi is not None and df_weather is not None:
        resultados["relacion_clima_contagios"] = calcular_relacion_clima_contagios(df_epi, df_weather)

    if df_epi is not None and df_hosp is not None:
        resultados["relacion_hospitalizaciones"] = calcular_relacion_hospitalizaciones(df_epi, df_hosp)

    logger.info("=" * 60)
    logger.info("TRANSFORMACIONES COMPLETADAS")
    logger.info("=" * 60)

    return resultados
