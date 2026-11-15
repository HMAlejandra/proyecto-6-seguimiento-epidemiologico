"""
preparacion_anova.py
--------------------
Preparación de datos para el análisis ANOVA.

Pregunta de investigación:
    ¿Existen diferencias estadísticamente significativas entre las tasas
    de contagio de los diferentes grupos etarios?

Hipótesis nula (H0):
    Las medias de las tasas de contagio de los diferentes grupos etarios
    son iguales.

Hipótesis alternativa (H1):
    Al menos una de las medias de las tasas de contagio de los grupos
    etarios es diferente.

Nivel de significancia: α = 0.05

Este módulo prepara los datos desde PySpark, extrayendo las observaciones
por grupo etario. Los datos se exportan como un diccionario de arrays numpy
para que scipy/statsmodels pueda procesarlos.

NOTA IMPORTANTE:
    - Se intenta calcular tasas por 100 000 habitantes usando los denominadores
      poblacionales de demographics.csv (population_age_00_09 … population_age_80_and_older).
      Estas fracciones son proporciones de la población total, NO conteos por grupo etario,
      por lo que las tasas resultantes son aproximaciones.  Se documenta esta limitación
      y se expone la variable analizada ("tasa_aprox" o "conteo") en los metadatos para
      que el ANOVA y los informes lo reflejen correctamente.
    - Si el cruce con demographics no es posible (archivo ausente, columnas faltantes o
      población nula), se recae automáticamente en conteos de new_confirmed y se registra
      el motivo en los metadatos.
    - Las observaciones diarias consecutivas pueden estar correlacionadas;
      no se asume independencia automáticamente.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"

# Mapeo entre los nombres de age_bucket que produce by-age.csv tras el unpivot
# y las columnas de proporción poblacional de demographics.csv.
# Los age_bin_X de Google Cloud Platform v3 corresponden a grupos de 10 años.
# Formato: age_bucket_label -> columna_demographics
BUCKET_A_DEMOGRAPHICS = {
    "0-9":   "population_age_00_09",
    "10-19": "population_age_10_19",
    "20-29": "population_age_20_29",
    "30-39": "population_age_30_39",
    "40-49": "population_age_40_49",
    "50-59": "population_age_50_59",
    "60-69": "population_age_60_69",
    "70-79": "population_age_70_79",
    "80+":   "population_age_80_and_older",
    # Alias alternativos que algunos dumps de GCP usan
    "80-":   "population_age_80_and_older",
    "80-89": "population_age_80_and_older",
    "90+":   "population_age_80_and_older",
}


def _calcular_poblacion_por_bucket(df_demographics: DataFrame,
                                   location_keys: list) -> dict:
    """
    Calcula la población aproximada de cada grupo etario para las ubicaciones dadas.

    demographics.csv almacena las fracciones poblacionales como proporciones
    de la población total (p.ej. population_age_00_09 = 0.17 significa que el
    17 % de la población tiene 0-9 años).  Multiplicando por population se obtiene
    el número absoluto de personas en ese grupo etario.

    Como by-age.csv puede cubrir múltiples países, se suma la población de todos
    los location_keys recibidos para obtener el denominador total por bucket.

    Parámetros
    ----------
    df_demographics : DataFrame
        DataFrame transformado de demographics.csv.
    location_keys : list
        Lista de location_key del nivel país que aparecen en by-age.

    Retorna
    -------
    dict
        {age_bucket_label: poblacion_absoluta_aproximada}
        Vacío si el cruce no es posible.
    """
    if df_demographics is None:
        logger.warning("demographics no disponible: no se pueden calcular denominadores.")
        return {}

    cols_necesarias = ["location_key", "population"] + list(BUCKET_A_DEMOGRAPHICS.values())
    cols_presentes = [c for c in cols_necesarias if c in df_demographics.columns]

    if "location_key" not in cols_presentes or "population" not in cols_presentes:
        logger.warning("demographics no tiene 'location_key' o 'population'. "
                       "Se omite el cruce.")
        return {}

    cols_prop_presentes = [c for c in list(BUCKET_A_DEMOGRAPHICS.values())
                           if c in df_demographics.columns]
    if not cols_prop_presentes:
        logger.warning("No se encontraron columnas de proporción etaria en demographics. "
                       "Se omite el cruce.")
        return {}

    # Filtrar a los location_keys de nivel país que están en by-age
    df_filtrado = (
        df_demographics
        .filter(F.col("location_key").isin(location_keys))
        .filter(F.col("population").isNotNull() & (F.col("population") > 0))
        .select(*cols_presentes)
    )

    if df_filtrado.count() == 0:
        logger.warning("demographics no tiene filas para los location_keys de by-age. "
                       "Se omite el cruce.")
        return {}

    # ---------------------------------------------------------------------------
    # Validar que las proporciones de cada fila sumen ~1.0 antes de usarlas
    # como denominador.
    #
    # demographics.csv almacena las fracciones etarias como proporciones de la
    # población total (p.ej. population_age_00_09 = 0.17 → 17 % del país).
    # Si están bien completadas, la suma de todos los grupos debe ser ≈ 1.0.
    # Cuando suman muy por debajo de 0.85 significa que faltan grupos en el
    # archivo (denominador subestimado → tasa sobreestimada) o que las
    # proporciones están en otra escala (p.ej. 0–100 en lugar de 0–1).
    # Cuando suman por encima de 1.15 indica doble conteo o escala incorrecta.
    # En ambos casos la multiplicación population × proporción produce conteos
    # de grupo etario erróneos, por lo que descartamos esas filas.
    #
    # Se usa el conjunto canónico de columnas (sin alias duplicados del grupo 80+)
    # para evitar sumar la misma proporción más de una vez.
    # ---------------------------------------------------------------------------
    COLS_PROP_CANONICAS = [
        "population_age_00_09", "population_age_10_19", "population_age_20_29",
        "population_age_30_39", "population_age_40_49", "population_age_50_59",
        "population_age_60_69", "population_age_70_79", "population_age_80_and_older",
    ]
    cols_prop_en_df = [c for c in COLS_PROP_CANONICAS if c in df_filtrado.columns]

    TOLERANCIA_SUMA_MIN = 0.85
    TOLERANCIA_SUMA_MAX = 1.15

    if cols_prop_en_df:
        # Calcular la suma de proporciones por fila para identificar filas inválidas
        suma_props_expr = sum(
            F.coalesce(F.col(c), F.lit(0.0)) for c in cols_prop_en_df
        )
        df_con_suma = df_filtrado.withColumn("_suma_props", suma_props_expr)

        total_filas = df_con_suma.count()
        filas_invalidas = df_con_suma.filter(
            (F.col("_suma_props") < TOLERANCIA_SUMA_MIN) |
            (F.col("_suma_props") > TOLERANCIA_SUMA_MAX)
        ).count()

        if filas_invalidas > 0:
            pct_invalidas = filas_invalidas / total_filas * 100
            logger.warning(
                "DENOMINADORES POCO FIABLES: %d de %d filas de demographics (%.1f%%) "
                "tienen proporciones etarias que suman fuera del rango [%.2f, %.2f]. "
                "Esto indica proporciones incompletas o en escala incorrecta. "
                "Se excluyen esas filas del cálculo del denominador para evitar "
                "tasas de contagio sobreestimadas o subestimadas en el ANOVA.",
                filas_invalidas, total_filas, pct_invalidas,
                TOLERANCIA_SUMA_MIN, TOLERANCIA_SUMA_MAX,
            )
            df_filtrado = df_con_suma.filter(
                (F.col("_suma_props") >= TOLERANCIA_SUMA_MIN) &
                (F.col("_suma_props") <= TOLERANCIA_SUMA_MAX)
            ).drop("_suma_props")

            if df_filtrado.count() == 0:
                logger.warning(
                    "Ninguna fila de demographics superó la validación de proporciones. "
                    "No se pueden calcular denominadores fiables. Se omite el cruce."
                )
                return {}
        else:
            logger.info(
                "Validación de proporciones: las %d filas de demographics tienen "
                "proporciones etarias dentro del rango esperado [%.2f, %.2f]. ✓",
                total_filas, TOLERANCIA_SUMA_MIN, TOLERANCIA_SUMA_MAX,
            )
            df_filtrado = df_con_suma.drop("_suma_props")
    else:
        logger.warning(
            "No se encontraron columnas canónicas de proporción etaria en demographics "
            "para validar la suma. Se continúa sin esta validación."
        )

    # Calcular población absoluta por grupo etario en cada fila y luego sumar
    # sobre todos los países para obtener el denominador global.
    agg_exprs = []
    for bucket_label, col_prop in BUCKET_A_DEMOGRAPHICS.items():
        if col_prop in df_filtrado.columns:
            agg_exprs.append(
                F.sum(
                    F.when(
                        F.col(col_prop).isNotNull() & (F.col(col_prop) > 0),
                        F.col("population") * F.col(col_prop)
                    ).otherwise(F.lit(0.0))
                ).alias(f"pop_{bucket_label.replace('-', '_').replace('+', 'plus')}")
            )

    if not agg_exprs:
        return {}

    fila = df_filtrado.agg(*agg_exprs).collect()
    if not fila:
        return {}

    row = fila[0].asDict()
    poblacion_por_bucket = {}
    for bucket_label, col_prop in BUCKET_A_DEMOGRAPHICS.items():
        alias = f"pop_{bucket_label.replace('-', '_').replace('+', 'plus')}"
        valor = row.get(alias)
        if valor and valor > 0:
            # Si el mismo bucket tiene varios alias (ej. "80+" y "80-"), usa el mayor
            if bucket_label not in poblacion_por_bucket or valor > poblacion_por_bucket[bucket_label]:
                poblacion_por_bucket[bucket_label] = float(valor)

    logger.info("Poblaciones aproximadas calculadas para %d grupos etarios.",
                len(poblacion_por_bucket))
    for b, p in sorted(poblacion_por_bucket.items()):
        logger.info("  %s: %.0f personas aprox.", b, p)

    return poblacion_por_bucket


def preparar_datos_anova(df_by_age: DataFrame,
                         df_demographics: DataFrame = None) -> Optional[dict]:
    """
    Prepara los datos de by-age.csv para el análisis ANOVA.

    Intenta calcular tasas de contagio por 100 000 habitantes por grupo etario
    cruzando con demographics.csv.  Si el cruce no es posible (demographics
    ausente, columnas faltantes, población nula), recae en conteos absolutos
    de new_confirmed y lo documenta en los metadatos.

    Extrae las observaciones de new_confirmed (o tasa) por grupo etario (age_bucket).
    Cada observación es el valor para una combinación (fecha, ubicación, age_bucket).

    Para que el ANOVA sea válido, necesitamos:
    1. Datos por grupo etario (age_bucket).
    2. Múltiples observaciones por grupo (fechas diferentes).
    3. Valores no nulos y no negativos.

    Decisiones metodológicas:
    - Se filtra a nivel país para no mezclar granularidades geográficas.
    - Se filtran valores nulos y negativos en new_confirmed.
    - Se agrega por (age_bucket, date, location_key) antes de collect() para
      evitar OOM en el driver con el dataset global.
    - Si se calculan tasas: tasa = (new_confirmed / poblacion_bucket) * 100_000.
      Los denominadores poblacionales son proporciones de la población total
      del país, por lo que las tasas son aproximaciones documentadas.
    - Cada grupo etario debe tener al menos 30 observaciones.

    Parámetros
    ----------
    df_by_age : DataFrame
        DataFrame transformado de by-age.csv.
    df_demographics : DataFrame, opcional
        DataFrame transformado de demographics.csv.  Si se suministra,
        se intenta el cruce para calcular tasas aproximadas.

    Retorna
    -------
    dict o None
        {
            "grupos": {age_bucket: numpy.array de valores},
            "metadatos": {
                "variable": "tasa_por_100k_aprox" | "new_confirmed",
                "tipo": "tasa aproximada por 100k hab." | "conteo (no tasa)",
                "n_grupos": int,
                "n_observaciones_total": int,
                "limitaciones": [lista de strings],
            }
        }
        None si no hay datos suficientes.
    """
    logger.info("=" * 60)
    logger.info("PREPARACIÓN DE DATOS PARA ANOVA")
    logger.info("=" * 60)

    if df_by_age is None:
        logger.error("DataFrame de by-age es None.")
        return None

    # Verificar que existan las columnas necesarias
    columnas_necesarias = ["age_bucket", "new_confirmed"]
    for col_name in columnas_necesarias:
        if col_name not in df_by_age.columns:
            logger.error("Columna '%s' no encontrada en by-age. Columnas: %s",
                         col_name, df_by_age.columns)
            return None

    # Filtrar datos:
    # 1. Solo nivel país (location_key sin guiones bajos) para evitar mezclar granularidades.
    # 2. Eliminar nulos y negativos en new_confirmed.
    num_guiones = F.size(F.split(F.col("location_key"), "_")) - 1
    df_filtrado = (
        df_by_age
        .filter(
            (num_guiones == 0) &
            F.col("age_bucket").isNotNull() &
            F.col("new_confirmed").isNotNull() &
            (F.col("new_confirmed") >= 0)
        )
        .select("age_bucket", "new_confirmed", "date", "location_key")
    )

    total_observaciones = df_filtrado.count()
    logger.info("Total de observaciones válidas: %d", total_observaciones)

    if total_observaciones == 0:
        logger.error("No hay observaciones válidas para ANOVA.")
        return None

    # Identificar grupos etarios y location_keys disponibles
    grupos_rows = df_filtrado.select("age_bucket").distinct().orderBy("age_bucket").collect()
    lista_grupos = [row["age_bucket"] for row in grupos_rows]
    logger.info("Grupos etarios encontrados: %s", lista_grupos)

    if len(lista_grupos) < 2:
        logger.error("Se necesitan al menos 2 grupos etarios para ANOVA. Encontrados: %d",
                     len(lista_grupos))
        return None

    location_keys_pais = [
        row["location_key"]
        for row in df_filtrado.select("location_key").distinct().collect()
    ]

    # -----------------------------------------------------------------------
    # Intentar cruce con demographics para calcular tasas aproximadas
    # -----------------------------------------------------------------------
    usar_tasas = False
    poblacion_por_bucket = {}
    razon_fallback = None

    if df_demographics is not None:
        logger.info("-" * 40)
        logger.info("Intentando cruce con demographics para denominadores poblacionales...")
        poblacion_por_bucket = _calcular_poblacion_por_bucket(
            df_demographics, location_keys_pais
        )

        # Verificar que todos los grupos etarios del dataset tengan denominador
        grupos_sin_denominador = [
            g for g in lista_grupos if g not in poblacion_por_bucket
        ]

        if grupos_sin_denominador:
            razon_fallback = (
                f"Los siguientes grupos etarios no tienen denominador poblacional "
                f"en demographics: {grupos_sin_denominador}. "
                f"No se puede calcular tasa sin denominador para todos los grupos."
            )
            logger.warning(razon_fallback)
            usar_tasas = False
        elif not poblacion_por_bucket:
            razon_fallback = (
                "El cruce con demographics no produjo denominadores válidos "
                "(población nula o columnas ausentes)."
            )
            logger.warning(razon_fallback)
            usar_tasas = False
        else:
            usar_tasas = True
            logger.info(
                "Cruce con demographics exitoso. Se calcularán tasas por 100 000 hab. "
                "(denominadores aproximados basados en proporciones poblacionales)."
            )
    else:
        razon_fallback = (
            "df_demographics no fue suministrado a preparar_datos_anova(). "
            "Se trabaja con conteos de new_confirmed."
        )
        logger.warning(razon_fallback)

    # -----------------------------------------------------------------------
    # Agregar en Spark antes de collect() para reducir volumen en el driver
    # -----------------------------------------------------------------------
    df_agregado = (
        df_filtrado
        .groupBy("age_bucket", "date", "location_key")
        .agg(F.sum("new_confirmed").alias("new_confirmed"))
        .select("age_bucket", "new_confirmed")
    )

    grupos_datos = {}
    metadatos_grupos = {}
    min_observaciones = 30

    for grupo in lista_grupos:
        valores_rows = (
            df_agregado
            .filter(F.col("age_bucket") == grupo)
            .select("new_confirmed")
            .collect()
        )
        array_conteos = np.array([row["new_confirmed"] for row in valores_rows], dtype=float)

        if usar_tasas:
            denominador = poblacion_por_bucket[grupo]
            # Tasa por 100 000 habitantes; denominador es aproximado
            array_valores = (array_conteos / denominador) * 100_000.0
            logger.info(
                "  Grupo '%s': denominador=%.0f, tasa media=%.4f por 100k",
                grupo, denominador, float(np.mean(array_valores)) if len(array_valores) > 0 else 0,
            )
        else:
            array_valores = array_conteos

        grupos_datos[grupo] = array_valores
        metadatos_grupos[grupo] = {
            "n_observaciones": len(array_valores),
            "media": float(np.mean(array_valores)) if len(array_valores) > 0 else None,
            "desviacion_estandar": float(np.std(array_valores, ddof=1)) if len(array_valores) > 1 else None,
            "minimo": float(np.min(array_valores)) if len(array_valores) > 0 else None,
            "maximo": float(np.max(array_valores)) if len(array_valores) > 0 else None,
        }
        if usar_tasas:
            metadatos_grupos[grupo]["denominador_poblacional"] = poblacion_por_bucket[grupo]

        logger.info("  Grupo '%s': n=%d, media=%.4f, std=%.4f",
                    grupo, len(array_valores),
                    metadatos_grupos[grupo]["media"] or 0,
                    metadatos_grupos[grupo]["desviacion_estandar"] or 0)

    # Verificar grupos con pocas observaciones
    grupos_insuficientes = [
        g for g, datos in metadatos_grupos.items()
        if datos["n_observaciones"] < min_observaciones
    ]
    if grupos_insuficientes:
        logger.warning(
            "Grupos con menos de %d observaciones: %s. "
            "Los resultados del ANOVA pueden no ser confiables para estos grupos.",
            min_observaciones, grupos_insuficientes,
        )

    # Documentar limitaciones
    if usar_tasas:
        variable = "tasa_por_100k_aprox"
        tipo = "tasa aproximada por 100 000 hab. (denominador basado en proporciones de demographics)"
        limitaciones = [
            "Las tasas se calculan como (new_confirmed / población_grupo_etario) × 100 000. "
            "Los denominadores son aproximaciones: demographics.csv almacena proporciones "
            "de la población total del país, NO conteos absolutos por grupo etario. "
            "Se validó que las proporciones de cada fila sumen entre 0.85 y 1.15; "
            "las filas fuera de ese rango fueron excluidas del cálculo del denominador.",
            "Se filtra a nivel país para evitar mezclar granularidades geográficas.",
            "Las observaciones diarias consecutivas pueden estar correlacionadas "
            "(no hay independencia garantizada entre observaciones temporales).",
            "Los valores extremos pueden corresponder a correcciones acumuladas "
            "y distorsionar el análisis.",
        ]
    else:
        variable = "new_confirmed"
        tipo = "conteo (no tasa)"
        limitaciones = [
            razon_fallback or "Se utilizan conteos de new_confirmed, NO tasas.",
            "ADVERTENCIA: Al trabajar con conteos absolutos (no tasas), los grupos etarios "
            "más numerosos demográficamente producirán sistemáticamente más casos aunque su "
            "riesgo relativo sea idéntico al de otros grupos. Un resultado estadísticamente "
            "significativo puede reflejar diferencias de tamaño poblacional, no de riesgo "
            "epidemiológico real.",
            "Se filtra a nivel país para evitar mezclar granularidades geográficas.",
            "Las observaciones diarias consecutivas pueden estar correlacionadas "
            "(no hay independencia garantizada entre observaciones temporales).",
            "Los valores extremos pueden corresponder a correcciones acumuladas "
            "y distorsionar el análisis.",
        ]

    if grupos_insuficientes:
        limitaciones.append(
            f"Grupos con menos de {min_observaciones} observaciones: {grupos_insuficientes}."
        )

    resultado = {
        "grupos": grupos_datos,
        "metadatos": {
            "variable": variable,
            "tipo": tipo,
            "usar_tasas": usar_tasas,
            "n_grupos": len(lista_grupos),
            "n_observaciones_total": total_observaciones,
            "grupos_info": metadatos_grupos,
            "limitaciones": limitaciones,
            "nivel_significancia": 0.05,
        },
    }

    logger.info(
        "Preparación completada. Variable: '%s'. %d grupos, %d observaciones totales.",
        variable, len(lista_grupos), total_observaciones,
    )

    return resultado
