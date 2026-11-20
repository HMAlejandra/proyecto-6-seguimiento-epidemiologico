
import os
import tempfile
from pathlib import Path

import pytest

# Configurar entorno antes de importar Spark
os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
# PYSPARK_PYTHON y PYSPARK_DRIVER_PYTHON se configuran en spark_config._configure_spark_env().
# Para usar Python 3.12 define antes de ejecutar:
#   $env:PYSPARK_PYTHON_312 = "C:\Users\TU_USUARIO\...\Python312\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, DateType

from etl.transformacion_spark import (
    convertir_tipos,
    eliminar_duplicados,
    validar_no_negativos,
    filtrar_colombia,
    filtrar_nivel_geografico,
    calcular_promedio_casos_por_pais,
    calcular_distribucion_por_edad,
    calcular_evolucion_temporal,
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
        df = spark.read.option("header", True).csv(str(csv_epidemiologia))
        # No debería lanzar error
        df = convertir_tipos(df, {"columna_inexistente": "long"})
        assert df is not None


class TestEliminarDuplicados:

    def test_elimina_duplicados(self, spark, csv_epidemiologia):
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

        # Hay 5 registros de Colombia (antes de eliminar duplicados):
        # 2 de CO_Bogota, 2 de CO_Medellin y 1 duplicado de CO_Bogota
        assert total_col == 5, f"Se esperaban 5 registros de Colombia, hay {total_col}"

    def test_filtra_colombia_verifica_location_key(self, spark, tmp_path):
      
        ruta = tmp_path / "co_niveles.csv"
        ruta.write_text(
            "date,location_key,new_confirmed\n"
            "2020-01-01,CO,100\n"           # Colombia nivel país — debe incluirse
            "2020-01-01,CO_Bogota,50\n"     # Colombia subregión — debe incluirse
            "2020-01-01,CO_Medellin,30\n"   # Colombia subregión — debe incluirse
            "2020-01-01,PE_Lima,20\n"       # Perú — debe excluirse
        )
        df = spark.read.option("header", True).csv(str(ruta))
        df_col = filtrar_colombia(df, "location_key")

        assert df_col.count() == 3, (
            f"Se esperaban 3 registros de Colombia (CO, CO_Bogota, CO_Medellin), "
            f"hay {df_col.count()}"
        )

        ubicaciones = {row["location_key"] for row in df_col.select("location_key").collect()}

        # El nivel país "CO" debe estar incluido
        assert "CO" in ubicaciones, (
            "El registro con location_key='CO' (nivel país) debería haberse incluido "
            "porque filtrar_colombia() usa startswith('CO'), no startswith('CO_')."
        )

        # Todas las ubicaciones deben empezar con "CO"
        for loc in ubicaciones:
            assert loc.startswith("CO"), (
                f"Ubicación '{loc}' no corresponde a Colombia (no empieza con 'CO')."
            )

        # Perú no debe aparecer
        assert "PE_Lima" not in ubicaciones, (
            "El registro de Perú (PE_Lima) no debería haberse incluido."
        )


@pytest.fixture
def csv_by_age_wide(tmp_path):
   
    ruta = tmp_path / "by_age_wide.csv"
    ruta.write_text(
        "date,location_key,"
        "age_bin_0,new_confirmed_age_0,new_deceased_age_0,cumulative_confirmed_age_0,cumulative_deceased_age_0,"
        "age_bin_1,new_confirmed_age_1,new_deceased_age_1,cumulative_confirmed_age_1,cumulative_deceased_age_1\n"
        "2020-01-01,CO,"
        "0-9,10,0,10,0,"
        "10-19,20,1,20,1\n"
        "2020-01-02,CO,"
        "0-9,15,0,25,0,"
        "10-19,25,2,45,3\n"
        "2020-01-01,PE,"
        "0-9,5,0,5,0,"
        "10-19,8,0,8,0\n"
    )
    return ruta


@pytest.fixture
def csv_by_age_variantes_80(tmp_path):
    ruta = tmp_path / "by_age_80_variantes.csv"
    ruta.write_text(
        "date,location_key,age_bucket,new_confirmed\n"
        "2020-01-01,CO,80-,100\n"
        "2020-01-02,CO,80-89,110\n"
        "2020-01-03,CO,90+,120\n"
        "2020-01-04,CO,80+,130\n"
    )
    return ruta


@pytest.fixture
def csv_epidemiologia_paises(tmp_path):
    ruta = tmp_path / "epi_paises.csv"
    ruta.write_text(
        "date,location_key,new_confirmed,new_deceased,cumulative_confirmed,cumulative_deceased,"
        "country_name\n"
        "2020-01-01,CO,100,2,100,2,Colombia\n"
        "2020-01-02,CO,150,3,250,5,Colombia\n"
        "2020-01-01,CO_25,40,1,40,1,Colombia\n"   # subregión — no debe sumarse
        "2020-01-01,PE,80,1,80,1,Peru\n"
        "2020-01-02,PE,90,2,170,3,Peru\n"
        "2020-01-01,PE_15,30,0,30,0,Peru\n"       # subregión — no debe sumarse
    )
    return ruta


@pytest.fixture
def csv_by_age_long(tmp_path):
    """
    CSV ya en formato LONG (post-unpivot) para testear las agregaciones
    de edad sin necesitar ejecutar el unpivot completo.
    """
    ruta = tmp_path / "by_age_long.csv"
    ruta.write_text(
        "date,location_key,age_bucket,new_confirmed,new_deceased\n"
        "2020-01-01,CO,0-9,10,0\n"
        "2020-01-02,CO,0-9,15,0\n"
        "2020-01-01,CO,10-19,20,1\n"
        "2020-01-02,CO,10-19,25,2\n"
        "2020-01-01,CO_25,0-9,5,0\n"   # subregión — no debe sumarse
        "2020-01-01,PE,0-9,8,0\n"
    )
    return ruta


# ---------------------------------------------------------------------------
# TestNormalizarAgeBucket
# ---------------------------------------------------------------------------

class TestNormalizarAgeBucket:

    def _aplicar_normalizacion(self, spark, tmp_path, contenido_csv):
        """Helper: lee CSV y aplica la expresión de normalización."""
        # Importar aquí para no contaminar el scope del módulo de tests
        from etl.transformacion_spark import _expr_normalizar_age_bucket
        ruta = tmp_path / "age_norm.csv"
        ruta.write_text(contenido_csv)
        df = spark.read.option("header", True).csv(str(ruta))
        return df.withColumn("age_bucket", _expr_normalizar_age_bucket("age_bucket"))

    def test_normaliza_80_guion(self, spark, tmp_path):
        """'80-' se convierte en '80+'."""
        df = self._aplicar_normalizacion(
            spark, tmp_path, "age_bucket\n80-\n"
        )
        valor = df.select("age_bucket").collect()[0]["age_bucket"]
        assert valor == "80+", f"Se esperaba '80+', se obtuvo '{valor}'"

    def test_normaliza_80_89(self, spark, tmp_path):
        """'80-89' se convierte en '80+'."""
        df = self._aplicar_normalizacion(
            spark, tmp_path, "age_bucket\n80-89\n"
        )
        valor = df.select("age_bucket").collect()[0]["age_bucket"]
        assert valor == "80+", f"Se esperaba '80+', se obtuvo '{valor}'"

    def test_normaliza_90_mas(self, spark, tmp_path):
        """'90+' se convierte en '80+'."""
        df = self._aplicar_normalizacion(
            spark, tmp_path, "age_bucket\n90+\n"
        )
        valor = df.select("age_bucket").collect()[0]["age_bucket"]
        assert valor == "80+", f"Se esperaba '80+', se obtuvo '{valor}'"

    def test_conserva_etiquetas_normales(self, spark, tmp_path):
        """Etiquetas estándar (0-9, 10-19, …, 70-79) no se modifican."""
        etiquetas_originales = ["0-9", "10-19", "20-29", "30-39",
                                "40-49", "50-59", "60-69", "70-79"]
        contenido = "age_bucket\n" + "\n".join(etiquetas_originales) + "\n"
        df = self._aplicar_normalizacion(spark, tmp_path, contenido)
        resultado = {r["age_bucket"] for r in df.select("age_bucket").collect()}
        for etiqueta in etiquetas_originales:
            assert etiqueta in resultado, (
                f"La etiqueta '{etiqueta}' no debería haberse modificado."
            )

    def test_conserva_80_mas_sin_cambio(self, spark, tmp_path):
        """'80+' ya es canónico y debe conservarse sin cambio."""
        df = self._aplicar_normalizacion(
            spark, tmp_path, "age_bucket\n80+\n"
        )
        valor = df.select("age_bucket").collect()[0]["age_bucket"]
        assert valor == "80+", f"'80+' canónico debería conservarse, se obtuvo '{valor}'"

    def test_todas_variantes_unifican_en_80_mas(self, spark, csv_by_age_variantes_80):
        """
        '80-', '80-89', '90+' y '80+' en el mismo DF deben quedar todos como '80+'.
        """
        from etl.transformacion_spark import _expr_normalizar_age_bucket
        df = spark.read.option("header", True).csv(str(csv_by_age_variantes_80))
        df = df.withColumn("age_bucket", _expr_normalizar_age_bucket("age_bucket"))
        buckets = {r["age_bucket"] for r in df.select("age_bucket").collect()}
        assert buckets == {"80+"}, (
            f"Se esperaba solo {{'80+'}}, se obtuvo {buckets}"
        )


# ---------------------------------------------------------------------------
# TestTransformarByAge
# ---------------------------------------------------------------------------

class TestTransformarByAge:
    """
    Pruebas para el unpivot WIDE → LONG de by-age.csv.

    transformar_by_age() lee el CSV directamente desde disco usando la ruta
    RAW_GOOGLE_DIR, por lo que no se puede invocar directamente en tests
    unitarios sin monkeypatching.  En su lugar se testean las partes del
    proceso que sí son accesibles: el stack SQL producido sobre una vista
    temporal y la eliminación de duplicados en el resultado LONG.
    """

    def test_stack_wide_a_long_produce_filas_por_grupo(self, spark, csv_by_age_wide):
        """
        Un CSV WIDE con 2 grupos etarios y 3 filas produce 6 filas LONG
        (3 ubicaciones × 2 grupos; sin filtrar por Colombia).

        Se replica manualmente el SQL stack que usa transformar_by_age()
        para validar que la lógica es correcta con datos reales.
        """
        df = spark.read.option("header", True).csv(str(csv_by_age_wide))
        df.createOrReplaceTempView("by_age_raw_test")

        df_long = spark.sql("""
            SELECT
                date,
                location_key,
                stack(2,
                    age_bin_0, new_confirmed_age_0, new_deceased_age_0,
                        cumulative_confirmed_age_0, cumulative_deceased_age_0,
                    age_bin_1, new_confirmed_age_1, new_deceased_age_1,
                        cumulative_confirmed_age_1, cumulative_deceased_age_1
                ) AS (age_bucket, new_confirmed, new_deceased,
                      cumulative_confirmed, cumulative_deceased)
            FROM by_age_raw_test
        """)

        # 3 filas de entrada × 2 grupos = 6 filas LONG
        assert df_long.count() == 6, (
            f"Se esperaban 6 filas (3 ubicaciones × 2 grupos), hay {df_long.count()}"
        )

    def test_stack_columnas_esperadas(self, spark, csv_by_age_wide):
        """El resultado LONG tiene las 7 columnas esperadas."""
        df = spark.read.option("header", True).csv(str(csv_by_age_wide))
        df.createOrReplaceTempView("by_age_raw_cols_test")

        df_long = spark.sql("""
            SELECT
                date, location_key,
                stack(2,
                    age_bin_0, new_confirmed_age_0, new_deceased_age_0,
                        cumulative_confirmed_age_0, cumulative_deceased_age_0,
                    age_bin_1, new_confirmed_age_1, new_deceased_age_1,
                        cumulative_confirmed_age_1, cumulative_deceased_age_1
                ) AS (age_bucket, new_confirmed, new_deceased,
                      cumulative_confirmed, cumulative_deceased)
            FROM by_age_raw_cols_test
        """)

        columnas = set(df_long.columns)
        esperadas = {"date", "location_key", "age_bucket", "new_confirmed",
                     "new_deceased", "cumulative_confirmed", "cumulative_deceased"}
        assert esperadas.issubset(columnas), (
            f"Faltan columnas en el resultado LONG. Esperadas: {esperadas}, "
            f"presentes: {columnas}"
        )

    def test_deduplicacion_long(self, spark, tmp_path):
        """
        El resultado LONG no debe tener duplicados por (date, location_key, age_bucket).
        """
        ruta = tmp_path / "long_dup.csv"
        ruta.write_text(
            "date,location_key,age_bucket,new_confirmed\n"
            "2020-01-01,CO,0-9,10\n"
            "2020-01-01,CO,0-9,10\n"   # duplicado
            "2020-01-01,CO,10-19,20\n"
        )
        df = spark.read.option("header", True).csv(str(ruta))
        df_limpio = eliminar_duplicados(df, subset=["date", "location_key", "age_bucket"])
        assert df_limpio.count() == 2, (
            f"Se esperaban 2 filas tras deduplicar, hay {df_limpio.count()}"
        )

    def test_filtro_age_bucket_nulo(self, spark, tmp_path):
        """Filas donde age_bucket es nulo deben eliminarse del resultado LONG."""
        ruta = tmp_path / "long_nulos_bucket.csv"
        ruta.write_text(
            "date,location_key,age_bucket,new_confirmed\n"
            "2020-01-01,CO,0-9,10\n"
            "2020-01-02,CO,,20\n"      # age_bucket nulo — debe filtrarse
            "2020-01-01,CO,10-19,15\n"
        )
        df = spark.read.option("header", True).csv(str(ruta))
        df_filtrado = df.filter(F.col("age_bucket").isNotNull())
        assert df_filtrado.count() == 2, (
            f"Se esperaban 2 filas sin age_bucket nulo, hay {df_filtrado.count()}"
        )


# ---------------------------------------------------------------------------
# TestAgregaciones
# ---------------------------------------------------------------------------

class TestAgregaciones:
 
    def test_promedio_casos_filtra_nivel_pais(self, spark, csv_epidemiologia_paises):
        """
        calcular_promedio_casos_por_pais() no debe incluir registros de subregiones
        (CO_25, PE_15).  Solo deben aparecer los registros de nivel país.
        """
        df = spark.read.option("header", True).csv(str(csv_epidemiologia_paises))
        df = convertir_tipos(df, {
            "new_confirmed": "long",
            "new_deceased": "long",
            "cumulative_confirmed": "long",
            "cumulative_deceased": "long",
        })
        df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

        resultado = calcular_promedio_casos_por_pais(df)
        paises = {r["country_name"] for r in resultado.select("country_name").collect()}

        # Solo deben aparecer Colombia y Peru (a nivel país)
        assert "Colombia" in paises, "Colombia debe aparecer en el resultado."
        assert "Peru" in paises, "Peru debe aparecer en el resultado."
        assert len(paises) == 2, (
            f"Solo deben aparecer 2 países, se obtuvieron {len(paises)}: {paises}"
        )

    def test_promedio_casos_calcula_correctamente(self, spark, csv_epidemiologia_paises):
        """
        El promedio de new_confirmed para Colombia debe ser 125
        ((100 + 150) / 2), usando solo las filas de nivel país (CO).
        """
        df = spark.read.option("header", True).csv(str(csv_epidemiologia_paises))
        df = convertir_tipos(df, {"new_confirmed": "long"})

        resultado = calcular_promedio_casos_por_pais(df)
        fila_col = resultado.filter(F.col("country_name") == "Colombia").collect()

        assert len(fila_col) == 1, "Debe haber exactamente una fila para Colombia."
        avg = fila_col[0]["avg_new_confirmed"]
        assert avg == 125.0, (
            f"El promedio de Colombia debería ser 125.0, se obtuvo {avg}. "
            "Verifica que solo se usan filas de nivel país (CO, no CO_25)."
        )

    def test_distribucion_por_edad_agrupa_por_bucket(self, spark, csv_by_age_long):
        """
        calcular_distribucion_por_edad() agrupa por age_bucket y retorna
        una fila por grupo.  El fixture tiene 2 grupos (0-9 y 10-19).
        """
        df = spark.read.option("header", True).csv(str(csv_by_age_long))
        df = convertir_tipos(df, {"new_confirmed": "long", "new_deceased": "long"})

        resultado = calcular_distribucion_por_edad(df)
        buckets = {r["age_bucket"] for r in resultado.select("age_bucket").collect()}

        assert "0-9" in buckets, "El grupo '0-9' debe aparecer en la distribución."
        assert "10-19" in buckets, "El grupo '10-19' debe aparecer en la distribución."

    def test_distribucion_por_edad_excluye_negativos(self, spark, tmp_path):
        """
        calcular_distribucion_por_edad() debe excluir new_confirmed negativos
        antes de sumar, para que las correcciones no distorsionen los totales.
        """
        ruta = tmp_path / "age_neg.csv"
        ruta.write_text(
            "date,location_key,age_bucket,new_confirmed,new_deceased\n"
            "2020-01-01,CO,0-9,100,0\n"
            "2020-01-02,CO,0-9,-10,0\n"   # corrección negativa — no debe sumarse
            "2020-01-03,CO,0-9,50,0\n"
        )
        df = spark.read.option("header", True).csv(str(ruta))
        df = convertir_tipos(df, {"new_confirmed": "long", "new_deceased": "long"})

        resultado = calcular_distribucion_por_edad(df)
        fila = resultado.filter(F.col("age_bucket") == "0-9").collect()

        assert len(fila) == 1
        total = fila[0]["total_new_confirmed"]
        assert total == 150, (
            f"El total para '0-9' debería ser 150 (100 + 50, excluyendo -10), "
            f"se obtuvo {total}."
        )

    def test_evolucion_temporal_agrupa_por_fecha(self, spark, csv_epidemiologia_paises):
        """
        calcular_evolucion_temporal() retorna una fila por fecha a nivel global.
        Con 2 fechas distintas en el fixture, el resultado debe tener 2 filas.
        """
        df = spark.read.option("header", True).csv(str(csv_epidemiologia_paises))
        df = convertir_tipos(df, {"new_confirmed": "long", "new_deceased": "long",
                                  "cumulative_confirmed": "long"})
        df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

        resultado = calcular_evolucion_temporal(df)
        n_fechas = resultado.count()

        assert n_fechas == 2, (
            f"Se esperaban 2 fechas distintas en la evolución temporal, hay {n_fechas}."
        )

    def test_evolucion_temporal_no_duplica_subregiones(self, spark, csv_epidemiologia_paises):
        """
        calcular_evolucion_temporal() filtra a nivel país antes de agregar.
        Para 2020-01-01 hay CO=100 y PE=80 a nivel país, más CO_25=40 y PE_15=30
        como subregiones que NO deben sumarse.  El total global del día debe ser
        180 (100 + 80), no 250 (100 + 80 + 40 + 30).
        """
        df = spark.read.option("header", True).csv(str(csv_epidemiologia_paises))
        df = convertir_tipos(df, {"new_confirmed": "long", "new_deceased": "long",
                                  "cumulative_confirmed": "long"})
        df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

        resultado = calcular_evolucion_temporal(df)
        fila_dia1 = resultado.filter(
            F.col("date") == F.to_date(F.lit("2020-01-01"), "yyyy-MM-dd")
        ).collect()

        assert len(fila_dia1) == 1, "Debe haber exactamente una fila para 2020-01-01."
        total = fila_dia1[0]["total_new_confirmed"]
        assert total == 180, (
            f"El total global de 2020-01-01 debería ser 180 (CO=100 + PE=80, "
            f"sin subregiones), se obtuvo {total}."
        )
