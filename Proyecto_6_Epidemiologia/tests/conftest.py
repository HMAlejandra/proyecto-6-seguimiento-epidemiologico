"""
conftest.py
-----------
Configuración común para las pruebas con pytest.

- Añade la raíz del proyecto al sys.path para importar los módulos.
- Configura las variables de entorno necesarias para PySpark en Windows.
"""

import os
import sys
from pathlib import Path

# Añadir la raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configurar variables de entorno para PySpark en Windows.
# Se delega a spark_config._configure_spark_env() para centralizar la lógica:
# si Python 3.12 existe en la ruta indicada por PYSPARK_PYTHON_312 lo usa;
# si no, cae al intérprete activo. Nunca se hardcodea una ruta de usuario
# específica aquí para que las pruebas sean portables entre máquinas.
os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
# PYSPARK_PYTHON y PYSPARK_DRIVER_PYTHON se configuran automáticamente en
# spark_config._configure_spark_env() al crear la sesión Spark.
# Si necesitas apuntar a Python 3.12 en tu máquina, define antes de ejecutar:
#   $env:PYSPARK_PYTHON_312 = "C:\Users\TU_USUARIO\...\Python312\python.exe"
