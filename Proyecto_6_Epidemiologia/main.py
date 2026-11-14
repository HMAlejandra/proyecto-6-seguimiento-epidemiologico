"""
main.py
-------
Punto de entrada general del Proyecto 6 — Seguimiento Epidemiológico.

Permite ejecutar:
- El proceso ETL completo (descarga → transformación → calidad → carga → ANOVA).
- Solo la descarga de datos.
- Solo la exploración.
- Solo las pruebas.
- El análisis estadístico por separado.

Uso:
    python main.py etl              # Ejecuta el ETL completo
    python main.py descargar       # Solo descarga los datos
    python main.py explorar        # Solo explora los datos
    python main.py anova           # Solo ejecuta el análisis ANOVA
    python main.py pruebas         # Ejecuta las pruebas con pytest
"""

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


def mostrar_ayuda():
    """Muestra la ayuda de uso."""
    print("""
Proyecto 6 — Seguimiento Epidemiológico de Enfermedades Respiratorias
=====================================================================

Uso:
    python main.py <comando>

Comandos disponibles:
    etl         Ejecuta el proceso ETL completo
                    (descarga → exploración → transformación → calidad → carga → ANOVA)
    descargar    Solo descarga los datos de Google Cloud Platform
    explorar     Solo explora la estructura de los datos descargados
    anova        Solo ejecuta el análisis estadístico ANOVA
    pruebas      Ejecuta las pruebas automatizadas con pytest
    ayuda        Muestra esta ayuda

Ejemplos:
    python main.py etl
    python main.py descargar
    python main.py pruebas
    """)


COMANDOS = {
    "etl": ejecutar_etl,
    "descargar": ejecutar_descarga,
    "explorar": ejecutar_exploracion,
    "anova": ejecutar_anova,
    "pruebas": ejecutar_pruebas,
    "ayuda": mostrar_ayuda,
}


if __name__ == "__main__":
    configurar_logging()

    if len(sys.argv) < 2:
        mostrar_ayuda()
        sys.exit(0)

    comando = sys.argv[1].lower()

    if comando not in COMANDOS:
        print(f"Comando desconocido: '{comando}'")
        mostrar_ayuda()
        sys.exit(1)

    COMANDOS[comando]()
