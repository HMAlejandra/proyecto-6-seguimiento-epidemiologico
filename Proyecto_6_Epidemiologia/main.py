
import logging
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def configurar_logging():
    """Configura el logging básico."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ejecutar_etl():
    """Ejecuta el proceso ETL completo."""
    from etl.ejecutar_etl import ejecutar_etl_completo
    ejecutar_etl_completo()


def ejecutar_descarga():
    """Ejecuta solo la descarga de datos."""
    from etl.extraccion import descargar_todas_las_fuentes
    descargar_todas_las_fuentes()


def ejecutar_exploracion():
    """Ejecuta solo la exploración de datos."""
    from etl.exploracion import explorar_todas_las_fuentes
    explorar_todas_las_fuentes()


def ejecutar_anova():
    """Ejecuta solo el análisis estadístico ANOVA."""
    from etl.spark_config import get_spark_session, stop_spark_session
    from etl.transformacion_spark import transformar_by_age
    from analisis_estadistico.preparacion_anova import preparar_datos_anova
    from analisis_estadistico.anova import ejecutar_anova

    spark = get_spark_session("ANOVA_Epidemiologia")
    try:
        df_by_age = transformar_by_age(spark)
        if df_by_age is None:
            print("No hay datos por edad disponibles. Ejecuta el ETL completo primero.")
            return

        datos = preparar_datos_anova(df_by_age)
        if datos is not None:
            ejecutar_anova(datos)
        else:
            print("No se pudieron preparar los datos para ANOVA.")
    finally:
        stop_spark_session()


def ejecutar_pruebas():
    """Ejecuta las pruebas automatizadas con pytest."""
    tests_dir = PROJECT_ROOT / "tests"
    print(f"Ejecutando pruebas en: {tests_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-v"],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    configurar_logging()

    if len(sys.argv) > 1:
        comando = sys.argv[1]
        if comando == "etl":
            ejecutar_etl()
        elif comando == "descarga":
            ejecutar_descarga()
        elif comando == "exploracion":
            ejecutar_exploracion()
        elif comando == "anova":
            ejecutar_anova()
        elif comando == "tests":
            ejecutar_pruebas()
        else:
            print(f"Comando no reconocido: '{comando}'")
            print("Opciones disponibles: etl, descarga, exploracion, anova, tests")
    else:
        # Sin argumento ejecuta el ETL completo
        ejecutar_etl()
