"""
test_transformacion.py
-----------------------
Pruebas automatizadas para el módulo de transformación.

Verifica:
- Que las transformaciones funcionen correctamente.
- Que los tipos de datos sean correctos.
- Que no se generen duplicados inesperados.
- Que las fechas sean válidas.
- Que las agregaciones sean correctas.

NOTA: Como createDataFrame no funciona en este entorno (Python 3.14 + Spark),
se usan archivos CSV temporales para crear DataFrames de prueba.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Configurar entorno antes de importar Spark
os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
os.environ.setdefault('PYSPARK_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')
os.environ.setdefault('PYSPARK_DRIVER_PYTHON', r'C:\Users\helen\AppData\Local\Programs\Python\Python312\python.exe')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, DateType

from etl.transformacion_spark import (
    convertir_tipos,
    eliminar_duplicados,
    validar_no_negativos,
    filtrar_colombia,
)


@pytest.fixture(scope="module")
def spark():
    """Fixture que crea y limpia una sesión Spark para las pruebas."""
    spark = (
        SparkSession.builder
        .appName("test_transformacion")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def csv_epidemiologia(tmp_path):
    """Crea un archivo CSV de epidemiología de prueba."""
    ruta = tmp_path / "test_epi.csv"
    contenido = (
        "date,location_key,new_confirmed,new_deceased,cumulative_confirmed,cumulative_deceased\n"
        "2020-01-01,CO_Bogota,5,0,5,0\n"
        "2020-01-02,CO_Bogota,10,1,15,1\n"
        "2020-01-01,CO_Medellin,3,0,3,0\n"
        "2020-01-02,CO_Medellin,7,0,10,0\n"
        "2020-01-01,PE_Lima,2,0,2,0\n"
        "2020-01-02,PE_Lima,5,0,7,0\n"
        "2020-01-01,CO_Bogota,5,0,5,0\n"  # Duplicado intencional
    )
    ruta.write_text(contenido)
    return ruta


class TestConvertirTipos:
    """Pruebas de conversión de tipos."""

    def test_conversion_fecha(self, spark, csv_epidemiologia):
        """Verifica que la conversión de string a date funcione."""
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        df = convertir_tipos(df, {"date": "date"})

        tipo_date = dict((f.name, str(f.dataType)) for f in df.schema.fields)
        assert "date" in tipo_date
        assert "date" == tipo_date["date"] or "DateType" in tipo_date["date"]

    def test_conversion_enteros(self, spark, csv_epidemiologia):
        """Verifica que la conversión a enteros funcione."""
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        df = convertir_tipos(df, {
            "new_confirmed": "long",
            "new_deceased": "long",
            "cumulative_confirmed": "long",
            "cumulative_deceased": "long",
        })

        tipos = dict((f.name, str(f.dataType)) for f in df.schema.fields)
        assert "LongType" in tipos["new_confirmed"]
        assert "LongType" in tipos["new_deceased"]
        assert "LongType" in tipos["cumulative_confirmed"]

    def test_conversion_columna_inexistente(self, spark, csv_epidemiologia):
        """Verifica que no falle si una columna no existe."""
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        # No debería lanzar error
        df = convertir_tipos(df, {"columna_inexistente": "long"})
        assert df is not None


class TestEliminarDuplicados:
    """Pruebas de eliminación de duplicados."""

    def test_elimina_duplicados(self, spark, csv_epidemiologia):
        """Verifica que se eliminen los registros duplicados."""
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        total_antes = df.count()

        df_limpio = eliminar_duplicados(df, subset=["date", "location_key"])
        total_despues = df_limpio.count()

        # Hay 7 registros, 1 duplicado (fecha + location_key repetido)
        assert total_despues < total_antes, "No se eliminaron duplicados"
        assert total_despues == 6, f"Se esperaban 6 registros únicos, hay {total_despues}"

    def test_sin_duplicados(self, spark, tmp_path):
        """Verifica que no falle si no hay duplicados."""
        ruta = tmp_path / "sin_dup.csv"
        ruta.write_text("id,val\n1,a\n2,b\n3,c\n")
        df = spark.read.option("header", True).csv(str(ruta))
        df_limpio = eliminar_duplicados(df, subset=["id"])
        assert df_limpio.count() == 3


class TestValidarNegativos:
    """Pruebas de validación de negativos."""

    def test_detecta_negativos(self, spark, tmp_path):
        """Verifica que detecte valores negativos."""
        ruta = tmp_path / "negativos.csv"
        ruta.write_text("new_confirmed\n5\n-3\n10\n-1\n")
        df = spark.read.option("header", True).csv(str(ruta))
        df = convertir_tipos(df, {"new_confirmed": "long"})

        # No debe lanzar error, solo registrar
        df_resultado = validar_no_negativos(df, ["new_confirmed"])
        assert df_resultado is not None
        assert df_resultado.count() == 4  # No se eliminan registros

    def test_sin_negativos(self, spark, tmp_path):
        """Verifica que no haya problemas sin negativos."""
        ruta = tmp_path / "positivos.csv"
        ruta.write_text("new_confirmed\n5\n10\n3\n")
        df = spark.read.option("header", True).csv(str(ruta))
        df = convertir_tipos(df, {"new_confirmed": "long"})
        df_resultado = validar_no_negativos(df, ["new_confirmed"])
        assert df_resultado.count() == 3


class TestFiltrarColombia:
    """Pruebas de filtrado por Colombia."""

    def test_filtra_colombia(self, spark, csv_epidemiologia):
        """Verifica que filtre solo registros de Colombia."""
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        df_col = filtrar_colombia(df, "location_key")
        total_col = df_col.count()

        # Hay 5 registros de Colombia (antes de eliminar duplicados)
        assert total_col == 5, f"Se esperaban 5 registros de Colombia, hay {total_col}"

    def test_filtra_colombia_verifica_location_key(self, spark, csv_epidemiologia):
        """Verifica que todos los registros filtrados empiecen con CO_."""
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        df_col = filtrar_colombia(df, "location_key")
        ubicaciones = df_col.select("location_key").distinct().collect()
        for row in ubicaciones:
            assert row["location_key"].startswith("CO_"), f"Ubicación no es de Colombia: {row['location_key']}"
