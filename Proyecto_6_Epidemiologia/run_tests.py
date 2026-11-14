"""Script para ejecutar las pruebas con la configuración correcta."""
import os
import sys
import subprocess

# Configurar entorno antes de ejecutar pytest
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PYSPARK_PYTHON'] = r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe'

# Añadir hadoop\bin al PATH
current_path = os.environ.get('PATH', '')
hadoop_bin = r'C:\hadoop\bin'
if hadoop_bin not in current_path:
    os.environ['PATH'] = hadoop_bin + os.pathsep + current_path

project_root = os.path.dirname(os.path.abspath(__file__))

result = subprocess.run(
    [sys.executable, "-m", "pytest", os.path.join(project_root, "tests"), "-v", "--tb=short"],
    cwd=project_root,
    capture_output=False,
)

sys.exit(result.returncode)
