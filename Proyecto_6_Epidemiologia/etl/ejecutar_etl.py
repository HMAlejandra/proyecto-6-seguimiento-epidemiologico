"""
ejecutar_etl.py
---------------
Orquestación del proceso ETL completo.

Coordina las etapas del proceso:
1. Verificar dependencias.
2. Descargar los datos.
3. Explorar las fuentes.
4. Ejecutar transformaciones.
5. Validar calidad.
6. Generar archivos finales.
7. Ejecutar análisis estadístico cuando existan datos suficientes.
8. Registrar resultados y errores.

Implementa manejo de excepciones y asegura el cierre de la sesión Spark.
Evita ejecutar etapas innecesarias si una etapa crítica ha fallado.
"""

import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession

from etl.spark_config import get_spark_session, stop_spark_session
from etl import extraccion, exploracion, transformacion_spark, calidad_datos, carga_spark

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def configurar_logging():
    """Configura el logging para escribir en consola y archivo."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )
    logger.info("Log file: %s", log_file)
    return log_file


def verificar_dependencias() -> bool:
    """
    Verifica que las dependencias principales estén disponibles.

    Retorna
    -------
    bool
        True si todas las dependencias están disponibles.
    """
    logger.info("=" * 60)
    logger.info("VERIFICANDO DEPENDENCIAS")
    logger.info("=" * 60)

    dependencias_ok = True

    # PySpark
    try:
        import pyspark
        logger.info("PySpark: %s", pyspark.__version__)
    except ImportError:
        logger.error("PySpark no está instalado.")
        dependencias_ok = False

    # Requests
    try:
        import requests
        logger.info("Requests: %s", requests.__version__)
    except ImportError:
        logger.error("Requests no está instalado.")
        dependencias_ok = False

    # SciPy
    try:
        import scipy
        logger.info("SciPy: %s", scipy.__version__)
    except ImportError:
        logger.error("SciPy no está instalado.")
        dependencias_ok = False

    # Statsmodels
    try:
        import statsmodels
        logger.info("Statsmodels: %s", statsmodels.__version__)
    except ImportError:
        logger.error("Statsmodels no está instalado.")
        dependencias_ok = False

    # pytest
    try:
        import pytest
        logger.info("Pytest: %s", pytest.__version__)
    except ImportError:
        logger.error("Pytest no está instalado.")
        dependencias_ok = False

    if dependencias_ok:
        logger.info("Todas las dependencias están disponibles.")
    else:
        logger.error("Faltan dependencias. Instala con: pip install -r requirements.txt")

    return dependencias_ok


def etapa_extraccion(forzar: bool = False) -> bool:
    """
    Ejecuta la etapa de extracción (descarga de datos).

    Retorna
    -------
    bool
        True si al menos los archivos principales se descargaron.
    """
    logger.info("\n" + "#" * 60)
    logger.info("# ETAPA 1: EXTRACCIÓN")
    logger.info("#" * 60)

    try:
        resultados = extraccion.descargar_todas_las_fuentes(forzar=forzar)

        # Verificar que al menos epidemiology e index se descargaron
        google = resultados.get("google", {})
        criticos = ["epidemiology", "index"]
        criticos_ok = all(
            google.get(c, {}).get("exitoso", False) for c in criticos
        )

        if not criticos_ok:
            logger.error("Archivos críticos no descargados. No se puede continuar.")
            return False

        return True

    except Exception as e:
        logger.error("Error en extracción: %s", e)
        logger.error(traceback.format_exc())
        return False


def etapa_exploracion() -> bool:
    """
    Ejecuta la etapa de exploración.

    Retorna
    -------
    bool
        True si la exploración se completó sin errores críticos.
    """
    logger.info("\n" + "#" * 60)
    logger.info("# ETAPA 2: EXPLORACIÓN")
    logger.info("#" * 60)

    try:
        resultados = exploracion.explorar_todas_las_fuentes()
        logger.info("Exploración completada.")
        return True

    except Exception as e:
        logger.error("Error en exploración: %s", e)
        logger.error(traceback.format_exc())
        return False


def etapa_transformacion(spark) -> dict:
    """
    Ejecuta la etapa de transformación con PySpark.

    Retorna
    -------
    dict
        Diccionario con los DataFrames transformados y las agregaciones.
        Vacío si falla.
    """
    logger.info("\n" + "#" * 60)
    logger.info("# ETAPA 3: TRANSFORMACIÓN")
    logger.info("#" * 60)

    try:
        resultados = transformacion_spark.ejecutar_transformaciones(spark)
        return resultados

    except Exception as e:
        logger.error("Error en transformación: %s", e)
        logger.error(traceback.format_exc())
        return {}


def etapa_calidad(resultados_etl: dict) -> dict:
    """
    Ejecuta la etapa de validación de calidad.

    Retorna
    -------
    dict
        Reportes de calidad.
    """
    logger.info("\n" + "#" * 60)
    logger.info("# ETAPA 4: CALIDAD DE DATOS")
    logger.info("#" * 60)

    try:
        reportes = calidad_datos.validar_todos_los_datasets(resultados_etl)
        return reportes

    except Exception as e:
        logger.error("Error en validación de calidad: %s", e)
        logger.error(traceback.format_exc())
        return {}


def etapa_carga(resultados_etl: dict) -> dict:
    """
    Ejecuta la etapa de carga (escritura de archivos finales).

    Retorna
    -------
    dict
        Reporte de carga.
    """
    logger.info("\n" + "#" * 60)
    logger.info("# ETAPA 5: CARGA DE RESULTADOS")
    logger.info("#" * 60)

    try:
        reporte = carga_spark.guardar_todas_las_tablas(resultados_etl)
        return reporte

    except Exception as e:
        logger.error("Error en carga: %s", e)
        logger.error(traceback.format_exc())
        return {}


def etapa_analisis_estadistico(resultados_etl: dict) -> bool:
    """
    Ejecuta el análisis estadístico (ANOVA) si hay datos suficientes.

    Retorna
    -------
    bool
        True si el análisis se ejecutó correctamente.
    """
    logger.info("\n" + "#" * 60)
    logger.info("# ETAPA 6: ANÁLISIS ESTADÍSTICO (ANOVA)")
    logger.info("#" * 60)

    try:
        # Importar aquí para evitar dependencias circulares
        from analisis_estadistico.preparacion_anova import preparar_datos_anova
        from analisis_estadistico.anova import ejecutar_anova

        df_by_age = resultados_etl.get("df_by_age")

        if df_by_age is None:
            logger.warning("No hay datos por edad disponibles. No se puede ejecutar ANOVA.")
            return False

        # Pasar también demographics para intentar calcular tasas aproximadas
        df_demographics = resultados_etl.get("df_demographics")

        # Preparar datos
        datos_preparados = preparar_datos_anova(df_by_age, df_demographics)

        if datos_preparados is None:
            logger.warning("No se pudieron preparar datos para ANOVA.")
            return False

        # Ejecutar ANOVA
        ejecutar_anova(datos_preparados)

        return True

    except Exception as e:
        logger.error("Error en análisis estadístico: %s", e)
        logger.error(traceback.format_exc())
        return False


def ejecutar_etl_completo(forzar_descarga: bool = False,
                          saltar_exploracion: bool = False) -> dict:
    """
    Ejecuta el proceso ETL completo de principio a fin.

    Parámetros
    ----------
    forzar_descarga : bool
        Si True, descarga nuevamente todos los archivos.
    saltar_exploracion : bool
        Si True, omite la etapa de exploración (para ejecuciones repetidas).

    Retorna
    -------
    dict
        Resumen consolidado de todas las etapas.
    """
    configurar_logging()

    logger.info("=" * 60)
    logger.info("INICIANDO PROCESO ETL COMPLETO")
    logger.info("Fecha: %s", datetime.now().isoformat())
    logger.info("=" * 60)

    resumen = {
        "inicio": datetime.now().isoformat(),
        "etapas": {},
    }

    # 1. Verificar dependencias
    if not verificar_dependencias():
        logger.error("Dependencias insuficientes. Abortando.")
        resumen["etapas"]["dependencias"] = "FALLO"
        resumen["fin"] = datetime.now().isoformat()
        return resumen
    resumen["etapas"]["dependencias"] = "OK"

    # 2. Extracción
    if not etapa_extraccion(forzar=forzar_descarga):
        logger.error("Extracción falló. Abortando.")
        resumen["etapas"]["extraccion"] = "FALLO"
        resumen["fin"] = datetime.now().isoformat()
        return resumen
    resumen["etapas"]["extraccion"] = "OK"

    # 3. Exploración (opcional)
    if not saltar_exploracion:
        etapa_exploracion()  # No aborta si falla
    resumen["etapas"]["exploracion"] = "OK" if not saltar_exploracion else "OMITIDA"

    # 4–7. Etapas que requieren Spark.
    # El bloque try/finally garantiza que stop_spark_session() se ejecuta
    # siempre, incluso si una etapa lanza una excepción no capturada internamente.
    # Sin este bloque, un fallo en etapa_carga() o etapa_analisis_estadistico()
    # dejaba la sesión JVM activa hasta el fin del proceso.
    spark = get_spark_session("ETL_Epidemiologia")
    try:
        # 4. Transformación
        resultados_etl = etapa_transformacion(spark)
        if not resultados_etl:
            logger.error("Transformación falló. Abortando.")
            resumen["etapas"]["transformacion"] = "FALLO"
            resumen["fin"] = datetime.now().isoformat()
            return resumen
        resumen["etapas"]["transformacion"] = "OK"

        # 5. Calidad
        reportes_calidad = etapa_calidad(resultados_etl)

        # Evaluar estado de calidad: si algún dataset es CRÍTICO, el pipeline se detiene.
        # Un estado CRÍTICO indica nulos en columnas clave o duplicados no eliminados,
        # lo que produciría resultados incorrectos en el análisis estadístico.
        datasets_criticos = [
            nombre
            for nombre, reporte in reportes_calidad.items()
            if isinstance(reporte, dict) and reporte.get("estado") == "CRÍTICO"
        ]
        if datasets_criticos:
            logger.error(
                "Calidad de datos CRÍTICA en: %s. "
                "El pipeline se detiene para evitar análisis con datos corruptos. "
                "Revisa los logs de calidad y corrige los problemas antes de continuar.",
                datasets_criticos,
            )
            resumen["etapas"]["calidad"] = f"CRÍTICO — datasets afectados: {datasets_criticos}"
            resumen["fin"] = datetime.now().isoformat()
            return resumen

        resumen["etapas"]["calidad"] = "OK"

        # 6. Carga
        etapa_carga(resultados_etl)
        resumen["etapas"]["carga"] = "OK"

        # 7. Análisis estadístico
        etapa_analisis_estadistico(resultados_etl)
        resumen["etapas"]["analisis_estadistico"] = "OK"

    finally:
        # Cerrar Spark siempre, independientemente de éxito o excepción
        stop_spark_session()

    resumen["fin"] = datetime.now().isoformat()

    logger.info("=" * 60)
    logger.info("PROCESO ETL COMPLETADO")
    logger.info("Resumen de etapas:")
    for etapa, estado in resumen["etapas"].items():
        logger.info("  %s: %s", etapa, estado)
    logger.info("=" * 60)

    return resumen


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ejecutar_etl_completo()
