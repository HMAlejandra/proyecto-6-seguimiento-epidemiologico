"""Script para ejecutar el ETL con la configuración correcta."""
import os
import sys

# ---------------------------------------------------------------------------
# Configuración de entorno para PySpark en Windows
# ---------------------------------------------------------------------------
# Las rutas se leen de variables de entorno para que el proyecto sea portable.
# Establécelas antes de ejecutar este script si tus rutas son distintas:
#
#   PowerShell:
#     $env:HADOOP_HOME       = "C:\hadoop"
#     $env:PYSPARK_PYTHON_312 = "C:\Users\TU_USUARIO\...\Python312\python.exe"
#
# Si las variables no están definidas se usan los valores por defecto de
# spark_config.py (C:\hadoop y la ruta de Helen), que seguirán funcionando
# en la máquina original sin ningún cambio.
# ---------------------------------------------------------------------------

_hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
_python_312  = os.environ.get(
    "PYSPARK_PYTHON_312",
    r"C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe",
)

os.environ["HADOOP_HOME"]             = _hadoop_home
os.environ["PYSPARK_PYTHON"]          = _python_312
os.environ["PYSPARK_DRIVER_PYTHON"]   = _python_312

# Añadir hadoop\bin al PATH
_hadoop_bin = os.path.join(_hadoop_home, "bin")
_current_path = os.environ.get("PATH", "")
if _hadoop_bin not in _current_path:
    os.environ["PATH"] = _hadoop_bin + os.pathsep + _current_path

# Añadir la raíz del proyecto al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from etl.ejecutar_etl import ejecutar_etl_completo

if __name__ == '__main__':
    # saltar_exploracion=True: ya se exploró en la ejecución anterior
    # forzar_descarga=False: los datos ya están descargados
    resumen = ejecutar_etl_completo(forzar_descarga=False, saltar_exploracion=True)
    print("\n" + "=" * 60)
    print("RESUMEN DEL ETL:")
    for etapa, estado in resumen.get("etapas", {}).items():
        print(f"  {etapa}: {estado}")
    print("=" * 60)
