"""
test_calidad.py
----------------
Pruebas automatizadas para el módulo de calidad de datos.

Verifica:
- Detección de nulos.
- Detección de duplicados.
- Validación de fechas.
- Validación de negativos.
- Coherencia entre casos y fallecimientos.
- Integridad de claves.
"""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
os.environ.setdefault('PYSPARK_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')
os.environ.setdefault('PYSPARK_DRIVER_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from etl.calidad_datos import (
    validar_nulos,
    validar_duplicados,
    validar_fechas,
    validar_negativos,
    validar_coherencia_casos_fallecimientos,
    validar_integridad_claves,
    generar_reporte_calidad,
)


@pytest.fixture(scope="module")
def spark():
    """Fixture que crea y limpia una sesión Spark."""
    spark = (
        SparkSession.builder
        .appName("test_calidad")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def csv_bueno(tmp_path):
    """CSV con datos correctos."""
    ruta = tmp_path / "bueno.csv"
    ruta.write_text(
        "date,location_key,new_confirmed,new_deceased,cumulative_confirmed,cumulative_deceased\n"
        "2020-01-01,CO_Bogota,5,0,5,0\n"
        "2020-01-02,CO_Bogota,10,1,15,1\n"
        "2020-01-01,PE_Lima,2,0,2,0\n"
    )
    return ruta


@pytest.fixture
def csv_con_problemas(tmp_path):
    """CSV con problemas de calidad."""
    ruta = tmp_path / "problemas.csv"
    ruta.write_text(
        "date,location_key,new_confirmed,new_deceased,cumulative_confirmed,cumulative_deceased\n"
        "2020-01-01,CO_Bogota,5,0,5,0\n"
        "2020-01-01,CO_Bogota,5,0,5,0\n"  # Duplicado
        "2020-01-02,,10,1,15,1\n"          # location_key nulo
        "2020-01-03,PE_Lima,-5,0,2,0\n"     # new_confirmed negativo
        "2020-01-04,PE_Lima,3,10,3,10\n"    # fallecimientos > casos (incoherente)
    )
    return ruta


class TestValidarNulos:
    """Pruebas de detección de nulos."""

    def test_detecta_nulos(self, spark, csv_con_problemas):
        """Verifica que detecte nulos correctamente."""
        df = spark.read.option("header", True).csv(str(csv_con_problemas))
        resultado = validar_nulos(df, "test_nulos")

        assert "location_key" in resultado
        assert resultado["location_key"]["nulos"] == 1

    def test_sin_nulos(self, spark, csv_bueno):
        """Verifica que reporte correctamente cuando no hay nulos."""
        df = spark.read.option("header", True).csv(str(csv_bueno))
        resultado = validar_nulos(df, "test_sin_nulos", columnas_criticas=["date", "location_key"])

        # Las columnas críticas no deben tener nulos
        assert resultado["date"]["nulos"] == 0
        assert resultado["location_key"]["nulos"] == 0


class TestValidarDuplicados:
    """Pruebas de detección de duplicados."""

    def test_detecta_duplicados(self, spark, csv_con_problemas):
        """Verifica que detecte duplicados."""
        df = spark.read.option("header", True).csv(str(csv_con_problemas))
        resultado = validar_duplicados(df, "test_dup", columnas_clave=["date", "location_key"])

        assert resultado["hay_duplicados"] is True
        assert resultado["duplicados"] == 1

    def test_sin_duplicados(self, spark, csv_bueno):
        """Verifica que reporte correctamente sin duplicados."""
        df = spark.read.option("header", True).csv(str(csv_bueno))
        resultado = validar_duplicados(df, "test_sin_dup", columnas_clave=["date", "location_key"])

        assert resultado["hay_duplicados"] is False


class TestValidarFechas:
    """Pruebas de validación de fechas."""

    def test_fechas_validas(self, spark, csv_bueno):
        """Verifica que las fechas válidas se reporten correctamente."""
        df = spark.read.option("header", True).csv(str(csv_bueno))
        df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
        resultado = validar_fechas(df, "test_fechas")

        assert resultado["total_nulos_fecha"] == 0
        assert resultado["fuera_de_rango"] == 0
        assert "2020-01-01" in resultado["fecha_min"]
        assert "2020-01-02" in resultado["fecha_max"]


class TestValidarNegativos:
    """Pruebas de detección de negativos."""

    def test_detecta_negativos(self, spark, csv_con_problemas):
        """Verifica que detecte valores negativos."""
        df = spark.read.option("header", True).csv(str(csv_con_problemas))
        df = df.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))
        resultado = validar_negativos(df, "test_neg", ["new_confirmed"])

        assert resultado["new_confirmed"] == 1  # Un valor negativo

    def test_sin_negativos(self, spark, csv_bueno):
        """Verifica que reporte correctamente sin negativos."""
        df = spark.read.option("header", True).csv(str(csv_bueno))
        df = df.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))
        resultado = validar_negativos(df, "test_sin_neg", ["new_confirmed"])

        assert resultado["new_confirmed"] == 0


class TestCoherencia:
    """Pruebas de coherencia entre casos y fallecimientos."""

    def test_detecta_incoherencia(self, spark, csv_con_problemas):
        """Verifica que detecte fallecimientos > casos."""
        df = spark.read.option("header", True).csv(str(csv_con_problemas))
        df = (
            df.withColumn("new_confirmed", F.col("new_confirmed").cast("long"))
            .withColumn("cumulative_confirmed", F.col("cumulative_confirmed").cast("long"))
            .withColumn("cumulative_deceased", F.col("cumulative_deceased").cast("long"))
        )
        resultado = validar_coherencia_casos_fallecimientos(df, "test_coherencia")

        # Hay un registro donde cumulative_deceased (10) > cumulative_confirmed (3)
        assert resultado.get("fallecimientos_mayor_casos", 0) >= 1


class TestIntegridadClaves:
    """Pruebas de integridad de claves."""

    def test_detecta_clave_nula(self, spark, csv_con_problemas):
        """Verifica que detecte claves nulas."""
        df = spark.read.option("header", True).csv(str(csv_con_problemas))
        resultado = validar_integridad_claves(df, "test_claves")

        assert resultado["nulos_clave"] == 1

    def test_claves_correctas(self, spark, csv_bueno):
        """Verifica que reporte correctamente cuando las claves están bien."""
        df = spark.read.option("header", True).csv(str(csv_bueno))
        resultado = validar_integridad_claves(df, "test_claves_ok")

        assert resultado["nulos_clave"] == 0
        assert resultado["claves_vacias"] == 0
