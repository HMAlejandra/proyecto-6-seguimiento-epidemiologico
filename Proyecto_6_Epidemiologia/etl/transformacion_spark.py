"""
transformacion_spark.py
-----------------------
Transformación de datos con PySpark.

Esta etapa se desarrolla principalmente con PySpark y realiza:
1. Lectura de archivos originales.
2. Estandarización de nombres de columnas.
3. Conversión de tipos de datos.
4. Normalización de fechas.
5. Tratamiento de valores nulos (sin reemplazar por cero automáticamente).
6. Identificación de registros duplicados.
7. Validación de valores negativos o inconsistentes.
8. Estandarización de países y regiones.
9. Limpieza de variables epidemiológicas.
10. Preparación de conjuntos de datos por temática.
11. Integración de fuentes cuando sea técnicamente válida.
12. Agregaciones necesarias para responder las preguntas del proyecto.

Decisiones de limpieza documentadas:
- NO se reemplazan valores nulos por cero. Un nulo no significa ausencia de contagios.
- NO se eliminan registros indiscriminadamente; se evalúa cada caso.
- NO se convierten conteos acumulados en diarios sin verificar el significado.
- NO se confunden conteos, tasas y porcentajes.
"""

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
    """
    Lee un archivo CSV crudo desde data/raw/google/.

    Parámetros
    ----------
    spark : SparkSession
    nombre_archivo : str
        Nombre del archivo (ej: "epidemiology.csv").

    Retorna
    -------
    DataFrame o None
    """
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
    """
    Convierte las columnas especificadas a los tipos indicados.

    Parámetros
    ----------
    df : DataFrame
    columnas_tipo : dict
        {nombre_columna: "long" | "double" | "int" | "string" | "date"}

    Retorna
    -------
    DataFrame con las columnas convertidas y columnas flag
    ``_cast_failed_<col>`` (booleanas) para cada columna numérica donde al
    menos una fila no pudo convertirse.

    Optimización de rendimiento
    ---------------------------
    La versión anterior ejecutaba dos ``df.filter(...).count()`` por cada
    columna numérica para medir nulos antes/después del cast.  Sobre un
    DataFrame con 8 columnas eso generaba 16 jobs Spark adicionales, cada uno
    releyendo el CSV completo desde disco (el DF no estaba cacheado).

    La versión actual reemplaza esos ``count()`` por columnas derivadas
    calculadas dentro del plan lógico de Spark:

    1. Se guarda ``_orig_<col>`` (valor original como string).
    2. Se castea la columna al tipo destino.
    3. Se añade ``_cast_failed_<col>`` = ``_orig.isNotNull() & col.isNull()``.
       Esta expresión es True exactamente en las filas que tenían un valor no
       nulo antes del cast y quedaron nulas después, es decir las que fallaron.
    4. Se elimina ``_orig_<col>``.

    La detección de cast failures queda embebida en el plan lógico y se evalúa
    en un único job junto al resto de transformaciones, sin ``count()`` extra.
    El número de filas fallidas se reporta sólo si ``generar_reporte_calidad``
    detecta la columna flag; ese conteo se hace una única vez sobre el DF ya
    transformado y cacheado.
    """
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

            # 3. Añadir columna flag para identificar filas donde el cast falló.
            #    La expresión es True sólo cuando:
            #      - el valor original era no-nulo (había algo que convertir), Y
            #      - el valor casteado es nulo (la conversión falló).
            #    Esto distingue "nulo de origen" (orig nulo → ambos nulos, flag=False)
            #    de "nulo por cast fallido" (orig no-nulo → casteado nulo, flag=True).
            #    No se usa count() aquí; el flag se evalúa lazily dentro del plan.
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
    """
    Elimina filas duplicadas, conservando la primera ocurrencia.

    Usa una ventana con row_number() para garantizar que la fila conservada
    sea determinista. Si se especifica orden_por, se conserva la fila con el
    valor más alto en esa columna (útil para quedarse con el registro más
    completo). Si no se especifica, se conserva la primera fila encontrada
    por el orden natural de las columnas del subset.

    Parámetros
    ----------
    df : DataFrame
    subset : list, opcional
        Columnas a considerar para identificar duplicados.
        Si None, usa todas las columnas.
    orden_por : str, opcional
        Nombre de columna para ordenar descendente al desempatar.
        Si None, usa el orden natural de Spark.
    contar_eliminados : bool, opcional
        Si True, ejecuta df.count() antes y después de deduplicar para
        loggear cuántos duplicados se eliminaron. Por defecto False.
        ADVERTENCIA: sobre DataFrames grandes (>10 M filas) cada count()
        desencadena un job Spark completo; actívalo solo cuando el diagnóstico
        lo justifique.

    Retorna
    -------
    DataFrame sin duplicados.
    """
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
    """
    Marca registros con valores negativos en columnas donde no deberían existir.

    En lugar de eliminar los registros, se conservan y se registran en logs
    para análisis posterior. Los valores negativos en conteos acumulados pueden
    deberse a correcciones o reinicios.

    Retorna
    -------
    DataFrame con los mismos registros (no se eliminan).
    """
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
    """
    Filtra el DataFrame para conservar únicamente registros de Colombia.

    En Google Cloud Platform, Colombia se identifica con location_key
    que empieza con "CO" (ISO 3166-1 alpha-2).

    Parámetros
    ----------
    df : DataFrame
    col_location : str
        Nombre de la columna de ubicación.

    Retorna
    -------
    DataFrame con solo registros de Colombia.
    """
    if col_location not in df.columns:
        logger.warning("Columna '%s' no encontrada. No se puede filtrar por Colombia.", col_location)
        return df

    df_col = df.filter(F.col(col_location).startswith("CO"))
    total = df_col.count()
    logger.info("Registros de Colombia: %d", total)
    return df_col


def filtrar_nivel_geografico(df: DataFrame, nivel: str = "pais",
                             col_location: str = "location_key") -> DataFrame:
    """
    Filtra el DataFrame para conservar un único nivel de granularidad geográfica.

    Los datos de Google Cloud Platform COVID-19 Open Data v3 contienen filas a
    múltiples niveles jerárquicos para el mismo país:
        - Nivel país:     location_key = "CO"           (sin guion bajo)
        - Nivel subregión: location_key = "CO_25"        (un guion bajo)
        - Nivel municipio: location_key = "CO_25_001"    (dos guiones bajos)

    Si no se filtra, las agregaciones que agrupan por país o fecha producen
    doble y triple conteo, porque suman las filas de país + subregiones + municipios.

    Parámetros
    ----------
    df : DataFrame
    nivel : str
        Nivel de granularidad a conservar: "pais", "subregion" o "municipio".
    col_location : str
        Nombre de la columna de ubicación.

    Retorna
    -------
    DataFrame filtrado a un solo nivel de granularidad.
    """
    if col_location not in df.columns:
        logger.warning("Columna '%s' no encontrada. No se puede filtrar por nivel.", col_location)
        return df

    # Contar guiones bajos en el location_key para determinar el nivel
    # location_key sin guion bajo = país (ej: "CO")
    # location_key con un guion bajo = subregión (ej: "CO_25")
    # location_key con dos guiones bajos = municipio (ej: "CO_25_001")
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


# ---------------------------------------------------------------------------
# Normalización de etiquetas de grupos etarios
# ---------------------------------------------------------------------------

# Mapa de variantes conocidas del grupo "80+" que GCP ha usado en distintos dumps
# hacia una etiqueta canónica única. Todas las demás etiquetas se conservan tal cual.
_AGE_BUCKET_NORMALIZACION = {
    "80-":    "80+",
    "80-89":  "80+",
    "80-99":  "80+",
    "90+":    "80+",
    "90-":    "80+",
    "90-99":  "80+",
    "100+":   "80+",
}

# Expresión Spark que aplica el mapa de normalización en una sola pasada.
# Se construye como CASE WHEN ... ELSE age_bucket END para evitar N withColumn()
# encadenados (uno por variante).
def _expr_normalizar_age_bucket(col_name: str = "age_bucket"):
    """
    Construye una expresión Spark que reemplaza variantes del grupo '80+'
    por la etiqueta canónica '80+'. El resto de etiquetas no se modifican.

    Se usa F.when().when()...otherwise() en lugar de una UDF para mantenerse
    dentro de la API Spark pura (más eficiente, compatible con optimizer).
    """
    expr = None
    for variante, canonica in _AGE_BUCKET_NORMALIZACION.items():
        condicion = F.col(col_name) == F.lit(variante)
        if expr is None:
            expr = F.when(condicion, F.lit(canonica))
        else:
            expr = expr.when(condicion, F.lit(canonica))
    # Para cualquier etiqueta no listada en el mapa, conservar el valor original
    return expr.otherwise(F.col(col_name)) if expr is not None else F.col(col_name)


# ---------------------------------------------------------------------------
# Transformaciones por archivo
# ---------------------------------------------------------------------------

def transformar_epidemiologia(spark) -> Optional[DataFrame]:
    """
    Transforma el archivo epidemiology.csv.

    Pasos:
    1. Leer CSV crudo.
    2. Convertir tipos (date, columnas numéricas).
    3. Eliminar duplicados.
    4. Validar valores negativos.
    5. Integrar con index.csv para obtener nombres de país/región.

    Retorna
    -------
    DataFrame transformado o None si el archivo no existe.
    """
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

    # Eliminar duplicados (misma fecha + misma ubicación)
    if "location_key" in df.columns and "date" in df.columns:
        df = eliminar_duplicados(df, subset=["date", "location_key"])

    # Validar valores negativos en columnas de conteo
    validar_no_negativos(df, list(EPIDEMIOLOGY_COLS.keys()))

    logger.info("Transformación de epidemiología completada.")
    return df


def transformar_by_age(spark) -> Optional[DataFrame]:
    """
    Transforma el archivo by-age.csv.

    El archivo original está en formato WIDE: cada grupo etario es una columna
    (new_confirmed_age_0, new_confirmed_age_1, ..., age_bin_0, age_bin_1, ...).
    Hay 10 grupos etarios (age_0 a age_9).

    Esta función convierte el formato WIDE a LONG usando la función `stack`
    de Spark SQL, que hace el unpivot en una sola pasada (mucho más eficiente
    que múltiples uniones).

    Produce un DataFrame con columnas:
    date, location_key, age_bucket, new_confirmed, new_deceased,
    cumulative_confirmed, cumulative_deceased.

    Los age_bin_X contienen las etiquetas reales: "0-9", "10-19", ..., "90-".

    Retorna
    -------
    DataFrame transformado en formato LONG o None.
    """
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

    # Construir la expresión stack para unpivot en una sola pasada.
    # stack(n, col1_1, col1_2, ...) produce n filas con 5 columnas cada una:
    # age_bucket, new_confirmed, new_deceased, cumulative_confirmed, cumulative_deceased.
    #
    # CORRECCIÓN: las columnas opcionales que no existen en el CSV se añaden al
    # DataFrame como columnas nulas explícitas (F.lit(None).cast("string")) antes
    # de construir la vista temporal. De esta forma, la expresión stack siempre
    # referencia nombres de columnas reales y nunca inyecta SQL literal como
    # "CAST(NULL AS STRING)", que se interpretaría como nombre de columna y
    # produciría un error de análisis o un resultado incorrecto.
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

    # Normalizar etiquetas del grupo etario superior.
    # GCP ha usado distintas variantes a lo largo del tiempo para el último
    # grupo: "80-", "80-89", "80-99", "90+", etc. Sin normalización, el ANOVA
    # trata cada variante como un grupo distinto, lo que rompe la comparación
    # entre países y hace que preparar_datos_anova() no encuentre denominador
    # poblacional para los buckets no listados en BUCKET_A_DEMOGRAPHICS.
    # La función _expr_normalizar_age_bucket() convierte todas las variantes
    # a la etiqueta canónica "80+" en una sola pasada, sin UDFs.
    df_long = df_long.withColumn("age_bucket", _expr_normalizar_age_bucket("age_bucket"))
    logger.info("Etiquetas de age_bucket normalizadas (variantes '80-x' → '80+').")

    # Convertir columnas numéricas de string a long
    # (el CSV se leyó con inferSchema=False, así que todas son strings)
    df_long = df_long.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))
    df_long = df_long.withColumn("new_deceased", F.col("new_deceased").cast("long"))
    df_long = df_long.withColumn("cumulative_confirmed", F.col("cumulative_confirmed").cast("long"))
    df_long = df_long.withColumn("cumulative_deceased", F.col("cumulative_deceased").cast("long"))

    # Eliminar duplicados (fecha + ubicación + grupo etario)
    df_long = eliminar_duplicados(df_long, subset=["date", "location_key", "age_bucket"])

    # Validar negativos
    validar_no_negativos(df_long, ["new_confirmed", "new_deceased",
                                   "cumulative_confirmed", "cumulative_deceased"])

    logger.info("Transformacion de by-age completada (WIDE -> LONG con stack): %d filas.", df_long.count())
    return df_long


def transformar_demographics(spark) -> Optional[DataFrame]:
    """
    Transforma el archivo demographics.csv.

    Contiene datos demográficos por ubicación (no tiene columna date).
    Proporciona denominadores poblacionales para calcular tasas.

    Retorna
    -------
    DataFrame transformado o None.
    """
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
    """
    Transforma el archivo index.csv (índice geográfico).

    Proporciona los nombres de países y regiones asociados a cada location_key.

    Retorna
    -------
    DataFrame transformado o None.
    """
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
    """
    Transforma el archivo weather.csv (variables climáticas).

    Retorna
    -------
    DataFrame transformado o None.
    """
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: weather.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "weather.csv")
    if df is None:
        return None

    tipos = {"date": "date", "location_key": "string"}
    tipos.update(WEATHER_COLS)
    df = convertir_tipos(df, tipos)

    # Eliminar duplicados
    subset = [c for c in ["date", "location_key"] if c in df.columns]
    if subset:
        df = eliminar_duplicados(df, subset=subset)

    logger.info("Transformación de weather completada.")
    return df


def transformar_hospitalizations(spark) -> Optional[DataFrame]:
    """
    Transforma el archivo hospitalizations.csv.

    Retorna
    -------
    DataFrame transformado o None.
    """
    logger.info("=" * 60)
    logger.info("TRANSFORMANDO: hospitalizations.csv")
    logger.info("=" * 60)

    df = leer_csv_crudo(spark, "hospitalizations.csv")
    if df is None:
        return None

    tipos = {"date": "date", "location_key": "string"}
    tipos.update(HOSPITALIZATIONS_COLS)
    df = convertir_tipos(df, tipos)

    # Eliminar duplicados
    subset = [c for c in ["date", "location_key"] if c in df.columns]
    if subset:
        df = eliminar_duplicados(df, subset=subset)

    # Validar negativos
    validar_no_negativos(df, list(HOSPITALIZATIONS_COLS.keys()))

    logger.info("Transformación de hospitalizations completada.")
    return df


# ---------------------------------------------------------------------------
# Integración de fuentes
# ---------------------------------------------------------------------------

def integrar_con_index(df: DataFrame, df_index: DataFrame) -> DataFrame:
    """
    Integra un DataFrame con el índice geográfico para añadir nombres de país/región.

    Parámetros
    ----------
    df : DataFrame
        DataFrame con columna location_key.
    df_index : DataFrame
        DataFrame del index.csv con nombres geográficos.

    Retorna
    -------
    DataFrame con columnas adicionales: country_name, subregion1_name.
    """
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
    """
    Integra con demographics para añadir población (denominador para tasas).

    Retorna
    -------
    DataFrame con columna population añadida.
    """
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


# ---------------------------------------------------------------------------
# Agregaciones para responder las preguntas del proyecto
# ---------------------------------------------------------------------------

def calcular_promedio_casos_por_pais(df_epidemiologia: DataFrame) -> DataFrame:
    """
    Pregunta 1: ¿Cuál es el promedio de casos nuevos de COVID-19 por país o región?

    Calcula el promedio de new_confirmed agrupado por country_name.

    Retorna
    -------
    DataFrame con columnas: country_name, avg_new_confirmed, total_new_confirmed, dias_con_datos.
    """
    logger.info("Calculando promedio de casos nuevos por país...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    # Excluir nulos, negativos y filas sin país identificado.
    # Los negativos son correcciones contables, no contagios reales; incluirlos
    # distorsionaría promedios y sumas de forma inconsistente con el ANOVA,
    # que ya los excluye en preparar_datos_anova().
    # Los nulos en country_name provienen de location_keys sin coincidencia en
    # index.csv (left join); se excluyen para no crear un grupo espurio "null".
    agg = (
        df_pais
        .filter(
            F.col("new_confirmed").isNotNull() &
            (F.col("new_confirmed") >= 0) &
            F.col("country_name").isNotNull()
        )
        .groupBy("country_name")
        .agg(
            F.avg("new_confirmed").alias("avg_new_confirmed"),
            F.sum("new_confirmed").alias("total_new_confirmed"),
            F.count("new_confirmed").alias("dias_con_datos"),
            F.max("new_confirmed").alias("max_new_confirmed"),
            F.min("new_confirmed").alias("min_new_confirmed"),
        )
        .orderBy(F.desc("total_new_confirmed"))
    )

    return agg


def calcular_distribucion_por_edad(df_by_age: DataFrame) -> DataFrame:
    """
    Pregunta 2: ¿Cómo se distribuyen los contagios según los diferentes grupos etarios?

    Agrega casos por grupo etario.

    Retorna
    -------
    DataFrame con columnas: age_bucket, total_new_confirmed, total_cumulative_confirmed,
                            total_new_deceased, total_cumulative_deceased.
    """
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

    # Excluir negativos antes de agregar.
    # Los negativos en datos de GCP son correcciones contables, no contagios reales.
    # Sin este filtro, las sumas por grupo etario quedan subcontadas respecto a
    # calcular_promedio_casos_por_pais() y calcular_evolucion_temporal(), que sí
    # los excluyen, generando inconsistencia interna entre las métricas del proyecto.
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
    """
    Pregunta 4: ¿Cómo ha evolucionado la cantidad de contagios a lo largo del tiempo?

    Agrega casos nuevos por fecha a nivel global.

    Retorna
    -------
    DataFrame con columnas: date, total_new_confirmed, total_new_deceased,
                            total_cumulative_confirmed.
    """
    logger.info("Calculando evolución temporal de contagios...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(F.sum("new_confirmed").alias("total_new_confirmed"))
    if "new_deceased" in df_pais.columns:
        agg_cols.append(F.sum("new_deceased").alias("total_new_deceased"))
    if "cumulative_confirmed" in df_pais.columns:
        agg_cols.append(F.max("cumulative_confirmed").alias("max_cumulative_confirmed"))

    # Excluir negativos para que la evolución temporal sea coherente con el
    # filtrado aplicado en el ANOVA (preparar_datos_anova excluye new_confirmed < 0).
    # Los valores negativos corresponden a correcciones contables; incluirlos
    # reduciría artificialmente los totales diarios globales.
    filtros = F.col("date").isNotNull()
    if "new_confirmed" in df_pais.columns:
        filtros = filtros & ((F.col("new_confirmed").isNull()) | (F.col("new_confirmed") >= 0))
    if "new_deceased" in df_pais.columns:
        filtros = filtros & ((F.col("new_deceased").isNull()) | (F.col("new_deceased") >= 0))

    agg = (
        df_pais
        .filter(filtros)
        .groupBy("date")
        .agg(*agg_cols)
        .orderBy("date")
    )

    return agg


def calcular_tasa_contagio_por_pais(df_epidemiologia: DataFrame,
                                    df_demographics: DataFrame) -> DataFrame:
    """
    Pregunta 5: ¿Cuál es la tasa de contagio por país o región?

    La tasa de contagio se calcula como:
        (total de casos confirmados / población) * 100000 (tasa por 100 mil habitantes)

    Requiere la integración con demographics para obtener la población.

    Retorna
    -------
    DataFrame con columnas: country_name, total_confirmed, population,
                            tasa_por_100k_habitantes.
    """
    logger.info("Calculando tasa de contagio por país...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    # Agregar casos totales por país.
    # Se usa sum(new_confirmed) excluyendo negativos en lugar de
    # max(cumulative_confirmed), porque los acumulados pueden tener reinicios
    # o correcciones que hacen que el máximo no represente el total real.
    # Esto es consistente con el resto de agregaciones del pipeline.
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

    # Agrupar por country_name y country_code (necesario para el join con demographics)
    group_cols = ["country_name"]
    if "country_code" in df_pais.columns:
        group_cols.append("country_code")

    df_agg = (
        df_pais
        .filter(F.col("country_name").isNotNull())
        .groupBy(*group_cols)
        .agg(*agg_cols)
    )

    # Integrar con demographics para obtener población por país.
    # En demographics, la fila a nivel país tiene location_key = country_code (ej: "CO").
    # Se filtra demographics a nivel país y se une por country_code == location_key.
    if "population" in df_demographics.columns and "location_key" in df_demographics.columns:
        df_pop = (
            filtrar_nivel_geografico(df_demographics, nivel="pais")
            .select("location_key", "population")
            .distinct()
        )
        if "country_code" in df_agg.columns:
            df_agg = df_agg.join(
                df_pop,
                df_agg["country_code"] == df_pop["location_key"],
                how="left"
            ).drop(df_pop["location_key"])
        else:
            logger.warning("Falta country_code para unir con demographics. La tasa no se puede calcular.")

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
    """
    Pregunta 7: ¿Cuáles son los diez países o regiones con las tasas de contagio más altas?

    Retorna
    -------
    DataFrame con los 10 países con mayor tasa de contagio.
    """
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
    """
    Pregunta 9: ¿Qué relación existe entre los casos nuevos y los fallecimientos?

    Calcula la tasa de letalidad: (fallecimientos / casos confirmados) * 100

    Retorna
    -------
    DataFrame con columnas: country_name, total_new_confirmed, total_new_deceased,
                            tasa_letalidad_pct.
    """
    logger.info("Calculando relación entre casos nuevos y fallecimientos...")

    # Filtrar a nivel país para evitar doble conteo
    df_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")

    agg_cols = []
    if "new_confirmed" in df_pais.columns:
        agg_cols.append(F.sum("new_confirmed").alias("total_new_confirmed"))
    if "new_deceased" in df_pais.columns:
        agg_cols.append(F.sum("new_deceased").alias("total_new_deceased"))

    if "country_name" not in df_pais.columns:
        logger.warning("Falta country_name. No se puede calcular relación por país.")
        return df_pais

    # Excluir negativos en ambas columnas para que la tasa de letalidad
    # no se distorsione por correcciones contables.
    # Se permite que new_deceased sea nulo (no todos los países reportan
    # fallecimientos diarios), pero no negativo.
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

    # Calcular tasa de letalidad
    if "total_new_confirmed" in df_agg.columns and "total_new_deceased" in df_agg.columns:
        df_agg = df_agg.withColumn(
            "tasa_letalidad_pct",
            F.when(F.col("total_new_confirmed") > 0,
                   (F.col("total_new_deceased") / F.col("total_new_confirmed")) * 100)
            .otherwise(None)
        )
        df_agg = df_agg.orderBy(F.desc("tasa_letalidad_pct"))

    return df_agg


def calcular_relacion_hospitalizaciones(df_epidemiologia: DataFrame,
                                        df_hospitalizaciones: DataFrame) -> DataFrame:
    """
    Pregunta 8: ¿Cómo se comportan las hospitalizaciones y los casos graves
    en relación con los contagios?

    Integra epidemiología con hospitalizaciones por fecha y ubicación.

    Retorna
    -------
    DataFrame con columnas: date, total_new_confirmed, total_new_hospitalized,
                            total_new_intensive_care, ratio_hospitalizados.
    """
    logger.info("Calculando relación entre hospitalizaciones y contagios...")

    # Filtrar a nivel país para evitar doble conteo en ambos datasets
    df_epi_pais = filtrar_nivel_geografico(df_epidemiologia, nivel="pais")
    df_hosp_pais = filtrar_nivel_geografico(df_hospitalizaciones, nivel="pais")

    # Agregar epidemiología por fecha
    epi_agg = df_epi_pais
    if "new_confirmed" in df_epi_pais.columns:
        epi_agg = epi_agg.groupBy("date").agg(
            F.sum("new_confirmed").alias("total_new_confirmed")
        )
    else:
        return df_epi_pais

    # Agregar hospitalizaciones por fecha
    hosp_agg_cols = []
    if "new_hospitalized_patients" in df_hosp_pais.columns:
        hosp_agg_cols.append(F.sum("new_hospitalized_patients").alias("total_new_hospitalized"))
    if "new_intensive_care_patients" in df_hosp_pais.columns:
        hosp_agg_cols.append(F.sum("new_intensive_care_patients").alias("total_new_intensive_care"))
    if "current_hospitalized_patients" in df_hosp_pais.columns:
        hosp_agg_cols.append(F.sum("current_hospitalized_patients").alias("total_current_hospitalized"))

    if hosp_agg_cols:
        hosp_agg = df_hosp_pais.groupBy("date").agg(*hosp_agg_cols)
    else:
        logger.warning("No se encontraron columnas de hospitalizaciones.")
        return epi_agg

    # Unir por fecha
    df_join = epi_agg.join(hosp_agg, on="date", how="left")

    # Calcular ratio
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
    """
    Pregunta 10: ¿Cómo se distribuyen los casos según edad y sexo?

    NOTA: El archivo by-age.csv de Google Cloud Platform no contiene
    desglose por sexo directamente. Solo tiene age_bucket.
    La variable sexo no está disponible en esta fuente.

    Retorna
    -------
    DataFrame con la distribución por edad (sin sexo por limitación de la fuente).
    """
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

    agg = (
        df_pais
        .filter(F.col("age_bucket").isNotNull())
        .groupBy("age_bucket")
        .agg(*agg_cols)
        .orderBy("age_bucket")
    )

    return agg


def calcular_relacion_clima_contagios(df_epidemiologia: DataFrame,
                                       df_weather: DataFrame) -> DataFrame:
    """
    Pregunta 6: ¿Existe una relación estadística entre variables ambientales
    y los contagios?

    Integra epidemiología con clima por fecha y ubicación.
    Calcula correlación entre temperatura/humedad y casos nuevos.

    NOTA: Esta función prepara los datos. La correlación estadística
    se evalúa por separado (no se afirma causalidad).

    Retorna
    -------
    DataFrame con columnas: date, location_key, new_confirmed,
                            average_temperature, relative_humidity, rainfall.
    """
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

    df_join = df_epi_sel.join(df_weather_sel, on=["date", "location_key"], how="inner")

    logger.info("Datos clima-contagios preparados: %d filas.", df_join.count())
    return df_join


# ---------------------------------------------------------------------------
# Orquestación de todas las transformaciones
# ---------------------------------------------------------------------------

def ejecutar_transformaciones(spark) -> dict:
    """
    Ejecuta todas las transformaciones y devuelve los DataFrames resultantes.

    Retorna
    -------
    dict
        Diccionario con todos los DataFrames transformados y las agregaciones.
    """
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
