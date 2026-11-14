"""
spark_config.py
---------------
Configuración centralizada de la sesión Apache Spark para el Proyecto 6.

Configura automáticamente las variables de entorno necesarias para ejecutar
PySpark en Windows:
- PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON: apuntan a Python 3.12 (compatible con PySpark).
- HADOOP_HOME: ruta donde se encuentra winutils.exe (necesario para escritura en Windows).

Reglas de uso:
- Llama a get_spark_session() para obtener (o reutilizar) la sesión activa.
- Llama a stop_spark_session() al finalizar el proceso ETL para liberar recursos.
- No crees múltiples sesiones simultáneas; Spark mantiene un singleton por proceso.
"""

import logging
import os
import sys
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración de entorno para PySpark en Windows
# ---------------------------------------------------------------------------

# Ruta al ejecutable de Python 3.12 (compatible con PySpark 4.0.0).
# Python 3.14 (versión por defecto del sistema) tiene incompatibilidades con el
# protocolo de workers de Spark, por lo que usamos Python 3.12 explícitamente.
#
# Se lee de la variable de entorno PYSPARK_PYTHON_312 si está definida.
# Si no, se intenta la ruta por defecto. Configura la variable antes de
# lanzar el pipeline para hacer el proyecto portable entre máquinas:
#   $env:PYSPARK_PYTHON_312 = "C:\Users\TU_USUARIO\...\Python312\python.exe"
_DEFAULT_PYTHON_312 = r"C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe"
PYTHON_312_PATH = os.environ.get("PYSPARK_PYTHON_312", _DEFAULT_PYTHON_312)

# Ruta a la instalación de Hadoop (winutils.exe) necesaria para operaciones
# de escritura de Spark en Windows.
#
# Se lee de la variable de entorno HADOOP_HOME si está definida.
# Si no, se usa C:\hadoop como valor por defecto.
#   $env:HADOOP_HOME = "C:\hadoop"
HADOOP_HOME = os.environ.get("HADOOP_HOME", r"C:\hadoop")


def _configure_spark_env():
    """
    Configura las variables de entorno que PySpark necesita en Windows.

    Debe llamarse ANTES de crear la SparkSession para que la JVM herede
    estas variables correctamente.
    """
    # Usar Python 3.12 para los workers de Spark
    if os.path.exists(PYTHON_312_PATH):
        os.environ['PYSPARK_PYTHON'] = PYTHON_312_PATH
        os.environ['PYSPARK_DRIVER_PYTHON'] = PYTHON_312_PATH
        logger.debug("PYSPARK_PYTHON configurado a: %s", PYTHON_312_PATH)
    else:
        # Fallback: usar el intérprete actual
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        logger.warning(
            "Python 3.12 no encontrado en %s. Usando intérprete actual: %s. "
            "Si experimentas errores de serialización, instala Python 3.12.",
            PYTHON_312_PATH, sys.executable,
        )

    # Configurar HADOOP_HOME para winutils.exe (necesario para escritura en Windows)
    if os.path.exists(HADOOP_HOME):
        os.environ['HADOOP_HOME'] = HADOOP_HOME
        # Añadir hadoop\bin al PATH para que la JVM encuentre las DLLs nativas
        hadoop_bin = os.path.join(HADOOP_HOME, 'bin')
        if hadoop_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ.get('PATH', '')
        logger.debug("HADOOP_HOME configurado a: %s", HADOOP_HOME)
    else:
        logger.warning(
            "HADOOP_HOME no encontrado en %s. La escritura de archivos puede fallar. "
            "Descarga winutils.exe y colócalo en C:\\hadoop\\bin\\",
            HADOOP_HOME,
        )


def get_spark_session(app_name: str = "Proyecto6_Epidemiologia") -> SparkSession:
    """
    Crea o recupera la sesión Spark configurada para ejecución local.

    Configuración aplicada:
    - master: local[*]  → usa todos los núcleos disponibles en la máquina local.
    - driver memory: 4g → memoria asignada al driver (ajustable según equipo).
    - Serialización Kryo para mejor rendimiento.
    - Log level WARNING para reducir ruido en consola.
    - Variables de entorno PYSPARK_PYTHON y HADOOP_HOME configuradas automáticamente.

    Parámetros
    ----------
    app_name : str
        Nombre de la aplicación Spark visible en la Spark UI.

    Retorna
    -------
    SparkSession
        Sesión activa de Spark.
    """
    # Configurar variables de entorno ANTES de crear la sesión
    _configure_spark_env()

    # Intentar recuperar sesión existente antes de crear una nueva
    existing = SparkSession.getActiveSession()
    if existing is not None:
        logger.info("Reutilizando sesión Spark existente: %s", existing.sparkContext.appName)
        return existing

    logger.info("Iniciando nueva sesión Spark con app_name='%s'", app_name)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # Memoria del driver — ajusta si tu equipo tiene menos RAM disponible
        .config("spark.driver.memory", "4g")
        # Serialización más eficiente
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Reduce el número de particiones por defecto para datos locales pequeños-medianos
        .config("spark.sql.shuffle.partitions", "8")
        # Permite sobre-escritura de archivos Parquet/CSV ya existentes
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        # Parser de fechas estricto: falla ruidosamente ante fechas malformadas
        # en lugar de enmascarar errores con el modo LEGACY.
        # Si algún CSV de la fuente tiene fechas en formato no estándar y el
        # pipeline falla aquí, revisar extraccion.py para limpiar ese campo antes.
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        # Ruta del warehouse local de Spark (evita crear spark-warehouse en raíz)
        .config(
            "spark.sql.warehouse.dir",
            os.path.join(os.path.dirname(__file__), "..", "spark-warehouse")
        )
        .getOrCreate()
    )

    # Reducir verbosidad de los logs internos de Spark
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        "Sesión Spark iniciada. Versión: %s — Master: %s",
        spark.version,
        spark.sparkContext.master,
    )

    return spark


def stop_spark_session() -> None:
    """
    Detiene la sesión Spark activa de forma segura.

    Debe llamarse al finalizar el proceso ETL para liberar todos los recursos
    de la JVM subyacente.
    """
    session = SparkSession.getActiveSession()
    if session is not None:
        logger.info("Deteniendo sesión Spark: %s", session.sparkContext.appName)
        session.stop()
        logger.info("Sesión Spark detenida correctamente.")
    else:
        logger.warning("No hay sesión Spark activa para detener.")
