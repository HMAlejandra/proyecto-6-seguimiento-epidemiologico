import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from etl.spark_config import get_spark_session, stop_spark_session

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_GOOGLE_DIR = PROJECT_ROOT / "data" / "raw" / "google"


def cargar_csv(spark, nombre_archivo: str) -> Optional[DataFrame]:
  
    ruta = RAW_GOOGLE_DIR / nombre_archivo
    if not ruta.exists():
        logger.warning("Archivo no encontrado: %s", ruta)
        return None

    logger.info("Cargando %s ...", ruta)
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)  # Inferir tipos para la exploración inicial
        .csv(str(ruta))
    )
    logger.info("Archivo cargado: %d filas, %d columnas.", df.count(), len(df.columns))
    return df


def explorar_esquema(df: DataFrame, nombre: str) -> dict:
   
    logger.info("=" * 60)
    logger.info("ESQUEMA DE: %s", nombre)
    logger.info("=" * 60)

    schema_info = {}
    for field in df.schema.fields:
        schema_info[field.name] = str(field.dataType)
        logger.info("  %s: %s (nullable=%s)", field.name, field.dataType, field.nullable)

    return schema_info


def explorar_nulos(df: DataFrame, nombre: str) -> dict:
   
    logger.info("-" * 40)
    logger.info("VALORES NULOS EN: %s", nombre)

    nulos_info = {}
    for col_name in df.columns:
        nulos = df.filter(F.col(col_name).isNull()).count()
        nulos_info[col_name] = nulos
        total = df.count()
        pct = (nulos / total * 100) if total > 0 else 0
        if nulos > 0:
            logger.info("  %s: %d nulos (%.1f%%)", col_name, nulos, pct)

    return nulos_info


def explorar_duplicados(df: DataFrame, nombre: str) -> int:
    
    total = df.count()
    distintos = df.distinct().count()
    duplicados = total - distintos
    logger.info("-" * 40)
    logger.info("DUPLICADOS EN: %s — Total: %d, Distintos: %d, Duplicados: %d",
                nombre, total, distintos, duplicados)
    return duplicados


def explorar_fechas(df: DataFrame, nombre: str, col_fecha: str = "date") -> dict:
    
    if col_fecha not in df.columns:
        logger.info("La columna '%s' no existe en %s. Columnas: %s",
                     col_fecha, nombre, df.columns)
        return {}

    logger.info("-" * 40)
    logger.info("FECHAS EN: %s (columna: %s)", nombre, col_fecha)

    try:
        fecha_min = df.agg(F.min(col_fecha)).collect()[0][0]
        fecha_max = df.agg(F.max(col_fecha)).collect()[0][0]
        logger.info("  Fecha mínima: %s", fecha_min)
        logger.info("  Fecha máxima: %s", fecha_max)
        return {"fecha_min": str(fecha_min), "fecha_max": str(fecha_max)}
    except Exception as e:
        logger.warning("No se pudieron calcular fechas para %s: %s", nombre, e)
        return {}


def explorar_ubicaciones(df: DataFrame, nombre: str,
                         col_location: str = "location_key") -> dict:
    
    if col_location not in df.columns:
        logger.info("La columna '%s' no existe en %s.", col_location, nombre)
        return {}

    logger.info("-" * 40)
    logger.info("UBICACIONES EN: %s (columna: %s)", nombre, col_location)

    total_ubic = df.select(col_location).distinct().count()
    logger.info("  Total de ubicaciones distintas: %d", total_ubic)

    # Mostrar ejemplos
    ejemplos = df.select(col_location).distinct().limit(10).collect()
    ejemplos_lista = [row[col_location] for row in ejemplos]
    logger.info("  Ejemplos: %s", ejemplos_lista)

    return {"total_ubicaciones": total_ubic, "ejemplos": ejemplos_lista}


def explorar_colombia(df: DataFrame, nombre: str,
                      col_location: str = "location_key") -> dict:
   
    if col_location not in df.columns:
        return {"disponible": False, "motivo": f"Columna '{col_location}' no existe"}

    logger.info("-" * 40)
    logger.info("COBERTURA DE COLOMBIA EN: %s", nombre)

    # Filtrar registros de Colombia (location_key empieza con "CO_")
    df_colombia = df.filter(F.col(col_location).startswith("CO"))

    total_col = df_colombia.count()
    ubicaciones_col = df_colombia.select(col_location).distinct().count()

    logger.info("  Registros de Colombia: %d", total_col)
    logger.info("  Ubicaciones (departamentos/municipios): %d", ubicaciones_col)

    if ubicaciones_col > 0:
        ejemplos = df_colombia.select(col_location).distinct().limit(20).collect()
        ejemplos_lista = [row[col_location] for row in ejemplos]
        logger.info("  Ejemplos de ubicaciones: %s", ejemplos_lista)

    return {
        "disponible": total_col > 0,
        "total_registros": total_col,
        "total_ubicaciones": ubicaciones_col,
    }


def explorar_archivo(spark, nombre_archivo: str) -> dict:
   
    nombre = nombre_archivo.replace(".csv", "")
    df = cargar_csv(spark, nombre_archivo)

    if df is None:
        return {"archivo": nombre, "error": "Archivo no encontrado"}

    resultado = {"archivo": nombre}
    resultado["total_filas"] = df.count()
    resultado["total_columnas"] = len(df.columns)
    resultado["columnas"] = df.columns
    resultado["esquema"] = explorar_esquema(df, nombre)
    resultado["nulos"] = explorar_nulos(df, nombre)
    resultado["duplicados"] = explorar_duplicados(df, nombre)
    resultado["fechas"] = explorar_fechas(df, nombre)

    # Explorar ubicaciones
    if "location_key" in df.columns:
        resultado["ubicaciones"] = explorar_ubicaciones(df, nombre)
        resultado["colombia"] = explorar_colombia(df, nombre)
    else:
        resultado["ubicaciones"] = {}
        resultado["colombia"] = {"disponible": False, "motivo": "Sin location_key"}

    return resultado


def explorar_todas_las_fuentes() -> dict:
    
    spark = get_spark_session("Exploracion_Epidemiologia")

    logger.info("=" * 60)
    logger.info("INICIANDO EXPLORACIÓN DE FUENTES DE DATOS")
    logger.info("=" * 60)

    archivos = [
        "index.csv",
        "epidemiology.csv",
        "by-age.csv",
        "demographics.csv",
        "weather.csv",
        "hospitalizations.csv",
    ]

    resultados = {}
    for archivo in archivos:
        try:
            logger.info("\n" + "#" * 60)
            logger.info("# EXPLORANDO: %s", archivo)
            logger.info("#" * 60)
            resultados[archivo] = explorar_archivo(spark, archivo)
        except Exception as e:
            logger.error("Error al explorar %s: %s", archivo, e)
            resultados[archivo] = {"archivo": archivo, "error": str(e)}

    logger.info("=" * 60)
    logger.info("EXPLORACIÓN COMPLETADA")
    logger.info("=" * 60)

    stop_spark_session()
    return resultados
