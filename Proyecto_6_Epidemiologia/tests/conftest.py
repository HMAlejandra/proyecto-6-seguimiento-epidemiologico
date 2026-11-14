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

# Configurar variables de entorno para PySpark en Windows
os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
os.environ.setdefault('PYSPARK_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')
os.environ.setdefault('PYSPARK_DRIVER_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')
