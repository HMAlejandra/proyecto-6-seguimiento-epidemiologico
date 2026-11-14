"""Script para ejecutar el ETL con la configuración correcta."""
import os
import sys

# Configurar entorno antes de importar cualquier módulo de Spark
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PYSPARK_PYTHON'] = r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe'

# Añadir hadoop\bin al PATH
current_path = os.environ.get('PATH', '')
hadoop_bin = r'C:\hadoop\bin'
if hadoop_bin not in current_path:
    os.environ['PATH'] = hadoop_bin + os.pathsep + current_path

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
