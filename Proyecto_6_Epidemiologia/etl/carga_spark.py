"""
carga_spark.py
--------------
Carga de resultados: escritura de archivos en staging, CSV y Parquet.

Responsabilidades:
- Guardar resultados intermedios en data/staging/.
- Guardar conjuntos finales en data/processed/csv/ y data/processed/parquet/.
- Utilizar PySpark para escribir los resultados.
- Verificar que los archivos se hayan generado correctamente.
- Controlar la escritura para evitar duplicados al ejecutar el ETL varias veces.
- Organizar las salidas para que sean fáciles de localizar e importar.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from etl.spark_config import get_spark_session

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
PROCESSED_CSV_DIR = PROJECT_ROOT / "data" / "processed" / "csv"
PROCESSED_PARQUET_DIR = PROJECT_ROOT / "data" / "processed" / "parquet"


def _ensure_dirs():
    """Crea los directorios de salida si no existen."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def escribir_csv(df: DataFrame, nombre: str, directorio: Path,
                 particiones: int = 1) -> bool:
    """
    Escribe un DataFrame como CSV.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre base del archivo (sin extensión).
    directorio : Path
        Directorio donde escribir.
    particiones : int
        Número de archivos part- a generar (1 para un solo archivo).

    Retorna
    -------
    bool
        True si la escritura fue exitosa.
    """
    _ensure_dirs()
    ruta_salida = directorio / nombre

    try:
        (
            df.coalesce(particiones)
            .write
            .mode("overwrite")
            .option("header", True)
            .option("encoding", "UTF-8")
            .csv(str(ruta_salida))
        )
        logger.info("CSV escrito: %s", ruta_salida)
        return True
    except Exception as e:
        logger.error("Error al escribir CSV %s: %s", nombre, e)
        return False


def escribir_parquet(df: DataFrame, nombre: str, directorio: Path,
                     particiones: int = 1) -> bool:
    """
    Escribe un DataFrame como Parquet.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre base del archivo (sin extensión).
    directorio : Path
        Directorio donde escribir.
    particiones : int
        Número de archivos part- a generar.

    Retorna
    -------
    bool
        True si la escritura fue exitosa.
    """
    _ensure_dirs()
    ruta_salida = directorio / nombre

    try:
        (
            df.coalesce(particiones)
            .write
            .mode("overwrite")
            .parquet(str(ruta_salida))
        )
        logger.info("Parquet escrito: %s", ruta_salida)
        return True
    except Exception as e:
        logger.error("Error al escribir Parquet %s: %s", nombre, e)
        return False


def verificar_csv(ruta: Path) -> bool:
    """
    Verifica que un archivo CSV se haya generado correctamente.

    Retorna
    -------
    bool
        True si el directorio existe y contiene al menos un archivo part-.
    """
    if not ruta.exists():
        logger.error("No se encontró el directorio: %s", ruta)
        return False

    archivos = list(ruta.glob("part-*.csv"))
    if not archivos:
        logger.error("No se encontraron archivos part- en %s", ruta)
        return False

    # Verificar que el primer archivo no esté vacío
    primer_archivo = archivos[0]
    if primer_archivo.stat().st_size == 0:
        logger.error("Archivo vacío: %s", primer_archivo)
        return False

    logger.info("Verificación OK: %s (%d archivos part-)", ruta, len(archivos))
    return True


def verificar_parquet(ruta: Path) -> bool:
    """
    Verifica que un archivo Parquet se haya generado correctamente.

    Retorna
    -------
    bool
        True si el directorio existe y contiene al menos un archivo .parquet.
    """
    if not ruta.exists():
        logger.error("No se encontró el directorio: %s", ruta)
        return False

    archivos = list(ruta.glob("*.parquet"))
    if not archivos:
        logger.error("No se encontraron archivos .parquet en %s", ruta)
        return False

    logger.info("Verificación OK: %s (%d archivos parquet)", ruta, len(archivos))
    return True


def guardar_en_staging(df: DataFrame, nombre: str) -> bool:
    """
    Guarda un DataFrame en data/staging/ en formato Parquet.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre del conjunto de datos.

    Retorna
    -------
    bool
        True si la escritura fue exitosa.
    """
    return escribir_parquet(df, nombre, STAGING_DIR)


def guardar_en_processed(df: DataFrame, nombre: str) -> dict:
    """
    Guarda un DataFrame en data/processed/ en CSV y Parquet.

    Ambos formatos representan los mismos datos.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre del conjunto de datos.

    Retorna
    -------
    dict
        {csv: bool, parquet: bool, csv_verificado: bool, parquet_verificado: bool}
    """
    resultado = {}

    # Escribir CSV
    resultado["csv"] = escribir_csv(df, nombre, PROCESSED_CSV_DIR)
    if resultado["csv"]:
        resultado["csv_verificado"] = verificar_csv(PROCESSED_CSV_DIR / nombre)

    # Escribir Parquet
    resultado["parquet"] = escribir_parquet(df, nombre, PROCESSED_PARQUET_DIR)
    if resultado["parquet"]:
        resultado["parquet_verificado"] = verificar_parquet(PROCESSED_PARQUET_DIR / nombre)

    return resultado


# ---------------------------------------------------------------------------
# Tablas finales para Power BI
# ---------------------------------------------------------------------------

# Definición de las tablas finales que se generarán.
# Cada entrada indica qué DataFrame del ETL se guarda y con qué nombre.
TABLAS_FINALES = [
    # Datos base transformados
    {"df_key": "df_epidemiologia", "nombre": "epidemiologia", "descripcion": "Datos epidemiológicos principales"},
    {"df_key": "df_by_age", "nombre": "datos_por_edad", "descripcion": "Casos y fallecimientos por grupo etario"},
    {"df_key": "df_demographics", "nombre": "demografia", "descripcion": "Datos demográficos por ubicación"},
    {"df_key": "df_index", "nombre": "indice_geografico", "descripcion": "Índice geográfico con nombres de países/regiones"},
    {"df_key": "df_weather", "nombre": "clima", "descripcion": "Variables climáticas por fecha y ubicación"},
    {"df_key": "df_hospitalizations", "nombre": "hospitalizaciones", "descripcion": "Hospitalizaciones, UCI y casos graves"},
    # Agregaciones para responder las preguntas del proyecto
    {"df_key": "promedio_casos_por_pais", "nombre": "promedio_casos_por_pais", "descripcion": "Pregunta 1: Promedio de casos nuevos por país"},
    {"df_key": "distribucion_por_edad", "nombre": "distribucion_por_edad", "descripcion": "Pregunta 2: Distribución de contagios por grupo etario"},
    {"df_key": "evolucion_temporal", "nombre": "evolucion_temporal", "descripcion": "Pregunta 4: Evolución de contagios en el tiempo"},
    {"df_key": "tasa_contagio_por_pais", "nombre": "tasa_contagio_por_pais", "descripcion": "Pregunta 5: Tasa de contagio por país"},
    {"df_key": "top_10_paises_tasa", "nombre": "top_10_paises_tasa", "descripcion": "Pregunta 7: Top 10 países con mayor tasa de contagio"},
    {"df_key": "relacion_clima_contagios", "nombre": "relacion_clima_contagios", "descripcion": "Pregunta 6: Relación entre clima y contagios"},
    {"df_key": "relacion_hospitalizaciones", "nombre": "relacion_hospitalizaciones", "descripcion": "Pregunta 8: Hospitalizaciones vs contagios"},
    {"df_key": "relacion_casos_fallecimientos", "nombre": "relacion_casos_fallecimientos", "descripcion": "Pregunta 9: Relación entre casos nuevos y fallecimientos"},
    {"df_key": "distribucion_edad_sexo", "nombre": "distribucion_edad_sexo", "descripcion": "Pregunta 10: Distribución por edad (sin sexo por limitación)"},
]


def guardar_todas_las_tablas(resultados_etl: dict) -> dict:
    """
    Guarda todas las tablas finales en CSV y Parquet.

    Parámetros
    ----------
    resultados_etl : dict
        Diccionario con los DataFrames transformados y las agregaciones.

    Retorna
    -------
    dict
        Reporte de guardado por cada tabla.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO CARGA DE RESULTADOS")
    logger.info("=" * 60)

    _ensure_dirs()

    reporte_carga = {}

    for tabla_def in TABLAS_FINALES:
        df_key = tabla_def["df_key"]
        nombre = tabla_def["nombre"]
        descripcion = tabla_def["descripcion"]

        df = resultados_etl.get(df_key)

        if df is None:
            logger.warning("Tabla '%s' (df_key='%s') no disponible. Se omite.", nombre, df_key)
            reporte_carga[nombre] = {"estado": "omitida", "descripcion": descripcion}
            continue

        # Guardar en staging (intermedio)
        guardar_en_staging(df, nombre)

        # Guardar en processed (finales: CSV + Parquet)
        resultado = guardar_en_processed(df, nombre)
        resultado["descripcion"] = descripcion
        resultado["estado"] = "guardada"

        reporte_carga[nombre] = resultado

        if resultado.get("csv_verificado") and resultado.get("parquet_verificado"):
            logger.info("Tabla '%s' guardada correctamente (CSV + Parquet).", nombre)
        else:
            logger.warning("Tabla '%s': verificación incompleta.", nombre)

    # Resumen
    guardadas = sum(1 for r in reporte_carga.values() if r.get("estado") == "guardada")
    omitidas = sum(1 for r in reporte_carga.values() if r.get("estado") == "omitida")

    logger.info("-" * 60)
    logger.info("RESUMEN DE CARGA:")
    logger.info("  Tablas guardadas: %d", guardadas)
    logger.info("  Tablas omitidas: %d", omitidas)
    logger.info("  CSV en: %s", PROCESSED_CSV_DIR)
    logger.info("  Parquet en: %s", PROCESSED_PARQUET_DIR)
    logger.info("=" * 60)

    return reporte_carga


def listar_archivos_procesados() -> dict:
    """
    Lista todos los archivos generados en data/processed/.

    Retorna
    -------
    dict
        {csv: [lista de rutas], parquet: [lista de rutas]}
    """
    csvs = []
    parquets = []

    if PROCESSED_CSV_DIR.exists():
        for d in sorted(PROCESSED_CSV_DIR.iterdir()):
            if d.is_dir():
                archivos = list(d.glob("part-*.csv"))
                if archivos:
                    csvs.append(str(d))

    if PROCESSED_PARQUET_DIR.exists():
        for d in sorted(PROCESSED_PARQUET_DIR.iterdir()):
            if d.is_dir():
                archivos = list(d.glob("*.parquet"))
                if archivos:
                    parquets.append(str(d))

    return {"csv": csvs, "parquet": parquets}
