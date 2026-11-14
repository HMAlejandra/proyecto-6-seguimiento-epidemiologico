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
    - No se calculan tasas con solo conteos.
    - No se inventan denominadores poblacionales.
    - Si los datos no permiten calcular tasas válidas, se trabaja con
      conteos por grupo etario y se documenta la limitación.
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


def preparar_datos_anova(df_by_age: DataFrame) -> Optional[dict]:
    """
    Prepara los datos de by-age.csv para el análisis ANOVA.

    Extrae las observaciones de new_confirmed por grupo etario (age_bucket).
    Cada observación es el valor de new_confirmed para una combinación
    (fecha, ubicación, age_bucket).

    Para que el ANOVA sea válido, necesitamos:
    1. Datos por grupo etario (age_bucket).
    2. Múltiples observaciones por grupo (fechas diferentes).
    3. Valores no nulos.

    Decisiones metodológicas:
    - Se usan los conteos de new_confirmed directamente (NO tasas).
      Razón: El archivo by-age.csv no incluye denominadores poblacionales
      por grupo etario. La columna 'population' está en demographics.csv
      a nivel de ubicación, no por edad. Calcular tasas con la población
      total dividida entre grupos etarios sería una aproximación que
      introduciría error. Se documenta esta limitación.
    - Se filtran valores nulos en new_confirmed.
    - Se filtran valores negativos (pueden ser correcciones).
    - Cada grupo etario debe tener al menos 30 observaciones para que
      las pruebas estadísticas tengan potencia razonable.

    Parámetros
    ----------
    df_by_age : DataFrame
        DataFrame transformado de by-age.csv.

    Retorna
    -------
    dict o None
        {
            "grupos": {age_bucket: numpy.array de valores},
            "metadatos": {
                "variable": "new_confirmed",
                "tipo": "conteo (no tasa)",
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
    # 1. Filtrar a nivel país para evitar mezclar granularidades geográficas.
    #    Los datos de by-age.csv contienen filas a múltiples niveles (país, subregión,
    #    municipio). Mezclarlas inflaría la varianza dentro de cada grupo etario
    #    y sesgaría el ANOVA.
    # 2. Eliminar nulos y negativos en new_confirmed.
    num_guiones = F.size(F.split(F.col("location_key"), "_")) - 1
    df_filtrado = (
        df_by_age
        .filter(
            (num_guiones == 0) &  # Solo nivel país (location_key sin guion bajo)
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

    # Identificar grupos etarios disponibles
    grupos = df_filtrado.select("age_bucket").distinct().orderBy("age_bucket").collect()
    lista_grupos = [row["age_bucket"] for row in grupos]
    logger.info("Grupos etarios encontrados: %s", lista_grupos)

    if len(lista_grupos) < 2:
        logger.error("Se necesitan al menos 2 grupos etarios para ANOVA. Encontrados: %d",
                     len(lista_grupos))
        return None

    # Extraer observaciones por grupo como arrays numpy
    grupos_datos = {}
    metadatos_grupos = {}

    min_observaciones = 30  # Mínimo para potencia estadística razonable

    for grupo in lista_grupos:
        valores = (
            df_filtrado
            .filter(F.col("age_bucket") == grupo)
            .select("new_confirmed")
            .collect()
        )
        array_valores = np.array([row["new_confirmed"] for row in valores], dtype=float)

        # Filtrar valores extremos atípicos (outliers) que puedan distorsionar el ANOVA.
        # No eliminamos registros, pero sí limitamos a valores razonables.
        # Un valor de new_confirmed negativo ya fue filtrado. Valores muy altos
        # (como correcciones acumuladas) se conservan pero se documentan.

        grupos_datos[grupo] = array_valores
        metadatos_grupos[grupo] = {
            "n_observaciones": len(array_valores),
            "media": float(np.mean(array_valores)) if len(array_valores) > 0 else None,
            "desviacion_estandar": float(np.std(array_valores, ddof=1)) if len(array_valores) > 1 else None,
            "minimo": float(np.min(array_valores)) if len(array_valores) > 0 else None,
            "maximo": float(np.max(array_valores)) if len(array_valores) > 0 else None,
        }

        logger.info("  Grupo '%s': n=%d, media=%.2f, std=%.2f",
                     grupo, len(array_valores),
                     metadatos_grupos[grupo]["media"] or 0,
                     metadatos_grupos[grupo]["desviacion_estandar"] or 0)

    # Verificar que todos los grupos tengan suficientes observaciones
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
    limitaciones = [
        "Se utilizan conteos de new_confirmed, NO tasas. "
        "El archivo by-age.csv no incluye denominadores poblacionales por grupo etario.",
        "Se filtran los datos a nivel país (location_key sin guion bajo) para "
        "evitar mezclar granularidades geográficas que sesgarían el ANOVA.",
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
            "variable": "new_confirmed",
            "tipo": "conteo (no tasa)",
            "n_grupos": len(lista_grupos),
            "n_observaciones_total": total_observaciones,
            "grupos_info": metadatos_grupos,
            "limitaciones": limitaciones,
            "nivel_significancia": 0.05,
        },
    }

    logger.info("Preparación completada. %d grupos, %d observaciones totales.",
                len(lista_grupos), total_observaciones)

    return resultado
