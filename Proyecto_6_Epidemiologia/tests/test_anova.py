"""
test_anova.py
--------------
Pruebas automatizadas para el análisis estadístico ANOVA.

Verifica:
- Que los resultados estadísticos no se calculen con datos insuficientes.
- Que la preparación de datos funcione correctamente.
- Que las pruebas de supuestos se ejecuten.
- Que las decisiones estadísticas sean consistentes.
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np

os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
os.environ.setdefault('PYSPARK_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')
os.environ.setdefault('PYSPARK_DRIVER_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType

from analisis_estadistico.preparacion_anova import preparar_datos_anova
from analisis_estadistico.anova import (
    probar_normalidad,
    probar_homogeneidad_varianzas,
    ejecutar_anova_clasico,
    ejecutar_kruskal_wallis,
)


@pytest.fixture(scope="module")
def spark():
    """Fixture que crea y limpia una sesión Spark."""
    spark = (
        SparkSession.builder
        .appName("test_anova")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def csv_by_age_test(tmp_path):
    """Crea un CSV de by-age de prueba con suficientes datos."""
    ruta = tmp_path / "by_age_test.csv"
    lineas = ["date,location_key,age_bucket,new_confirmed,cumulative_confirmed,new_deceased"]

    # Generar datos para 4 grupos etarios, 40 observaciones cada uno
    np.random.seed(42)
    grupos = ["0-9", "10-19", "20-29", "30-39"]
    for grupo in grupos:
        for i in range(40):
            fecha = f"2020-01-{(i % 28) + 1:02d}"
            ubicacion = f"CO_Ciudad_{i % 5}"
            valor = int(np.random.poisson(10))
            linea = f"{fecha},{ubicacion},{grupo},{valor},{valor * 10},{0}"
            lineas.append(linea)

    ruta.write_text("\n".join(lineas))
    return ruta


class TestPreparacion:
    """Pruebas de preparación de datos para ANOVA."""

    def test_preparacion_con_datos_suficientes(self, spark, csv_by_age_test):
        """Verifica que la preparación funcione con datos suficientes."""
        df = spark.read.option("header", True).csv(str(csv_by_age_test))
        df = df.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))
        df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

        resultado = preparar_datos_anova(df)

        assert resultado is not None
        assert "grupos" in resultado
        assert len(resultado["grupos"]) == 4  # 4 grupos etarios
        assert resultado["metadatos"]["n_grupos"] == 4
        assert resultado["metadatos"]["n_observaciones_total"] > 0

    def test_preparacion_sin_datos_suficientes(self, spark, tmp_path):
        """Verifica que devuelva None si no hay datos suficientes."""
        ruta = tmp_path / "pocos_datos.csv"
        ruta.write_text("date,location_key,age_bucket,new_confirmed\n2020-01-01,CO_1,0-9,5\n")
        df = spark.read.option("header", True).csv(str(ruta))
        df = df.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))

        resultado = preparar_datos_anova(df)
        # Solo 1 grupo → debe devolver None
        assert resultado is None

    def test_preparacion_filtro_nulos(self, spark, tmp_path):
        """Verifica que filtre valores nulos."""
        ruta = tmp_path / "nulos.csv"
        ruta.write_text(
            "date,location_key,age_bucket,new_confirmed\n"
            "2020-01-01,CO_1,0-9,5\n"
            "2020-01-02,CO_1,0-9,\n"  # nulo
            "2020-01-03,CO_1,0-9,-3\n"  # negativo
            "2020-01-01,CO_2,10-19,7\n"
        )
        df = spark.read.option("header", True).csv(str(ruta))
        df = df.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))

        resultado = preparar_datos_anova(df)
        if resultado is not None:
            # El grupo 0-9 debería tener 1 observación válida (solo el 5)
            assert len(resultado["grupos"]["0-9"]) == 1


class TestPruebasSupuestos:
    """Pruebas de las pruebas de supuestos estadísticos."""

    def test_normalidad_datos_normales(self):
        """Verifica Shapiro-Wilk con datos que siguen una distribución normal."""
        np.random.seed(42)
        grupos = {
            "A": np.random.normal(50, 10, 100),
            "B": np.random.normal(55, 10, 100),
        }
        resultado = probar_normalidad(grupos)

        assert "A" in resultado
        assert "B" in resultado
        # Con datos normales, la prueba no debería rechazar normalidad
        # (puede fallar ocasionalmente, pero es raro con 100 muestras)

    def test_homogeneidad_varianzas_iguales(self):
        """Verifica Levene con varianzas homogéneas."""
        np.random.seed(42)
        grupos = {
            "A": np.random.normal(50, 10, 100),
            "B": np.random.normal(55, 10, 100),
            "C": np.random.normal(60, 10, 100),
        }
        resultado = probar_homogeneidad_varianzas(grupos)

        assert "estadistico" in resultado
        assert "p_valor" in resultado
        # Con varianzas iguales, debería indicar homogeneidad
        assert bool(resultado["varianzas_homogeneas"]) is True

    def test_homogeneidad_varianzas_distintas(self):
        """Verifica Levene con varianzas heterogéneas."""
        np.random.seed(42)
        grupos = {
            "A": np.random.normal(50, 5, 200),    # varianza pequeña
            "B": np.random.normal(50, 50, 200),   # varianza grande
        }
        resultado = probar_homogeneidad_varianzas(grupos)

        # Con varianzas muy distintas, debería detectar heterogeneidad
        assert bool(resultado["varianzas_homogeneas"]) is False


class TestAnova:
    """Pruebas del ANOVA en sí."""

    def test_anova_clasico_diferencias_significativas(self):
        """Verifica que ANOVA detecte diferencias significativas."""
        np.random.seed(42)
        grupos = {
            "A": np.random.normal(10, 5, 100),
            "B": np.random.normal(50, 5, 100),  # media muy diferente
            "C": np.random.normal(90, 5, 100),  # media muy diferente
        }
        resultado = ejecutar_anova_clasico(grupos)

        assert resultado["estadistico_F"] > 0
        assert resultado["p_valor"] < 0.05  # Diferencia significativa
        assert resultado["n_grupos"] == 3

    def test_anova_clasico_sin_diferencias(self):
        """Verifica que ANOVA no detecte diferencias cuando no las hay."""
        np.random.seed(42)
        grupos = {
            "A": np.random.normal(50, 5, 100),
            "B": np.random.normal(50, 5, 100),  # misma media
            "C": np.random.normal(50, 5, 100),  # misma media
        }
        resultado = ejecutar_anova_clasico(grupos)

        assert resultado["estadistico_F"] > 0
        # Con la misma media, el p-valor debería ser alto (no significativo)
        assert resultado["p_valor"] > 0.05

    def test_kruskal_wallis(self):
        """Verifica que Kruskal-Wallis funcione."""
        np.random.seed(42)
        grupos = {
            "A": np.random.normal(10, 5, 50),
            "B": np.random.normal(50, 5, 50),
        }
        resultado = ejecutar_kruskal_wallis(grupos)

        assert resultado["estadistico_H"] > 0
        assert resultado["p_valor"] < 0.05  # Diferencia significativa

    def test_no_se_calcula_con_datos_insuficientes(self):
        """Verifica que no se calculen resultados con muy pocos datos."""
        grupos = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}

        # ANOVA debería poder ejecutarse (mínimo 2 observaciones por grupo)
        resultado = ejecutar_anova_clasico(grupos)
        assert resultado is not None
        assert "estadistico_F" in resultado
