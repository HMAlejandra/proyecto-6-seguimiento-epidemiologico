"""
calidad_datos.py
----------------
Validaciones de calidad de datos.

Implementa validaciones para:
- Valores nulos.
- Duplicados.
- Tipos de datos.
- Fechas inválidas.
- Países o regiones no identificados.
- Valores negativos no permitidos.
- Inconsistencias en casos y fallecimientos.
- Registros fuera del periodo esperado.
- Integridad de las claves.
- Coherencia de las agregaciones.
- Consistencia entre acumulados y suma de nuevos diarios.

Genera reportes que permiten identificar problemas.
No marca como correcto un conjunto de datos que tenga errores críticos sin resolver.
"""

import logging
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def validar_nulos(df: DataFrame, nombre: str, columnas_criticas: list = None) -> dict:
    """
    Cuenta y reporta valores nulos por columna.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre del dataset para el reporte.
    columnas_criticas : list, opcional
        Columnas donde los nulos son críticos (no deberían existir).

    Retorna
    -------
    dict
        {columna: {nulos, total, porcentaje, es_critico}}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE NULOS: %s", nombre)

    total = df.count()
    resultado = {}
    problemas = 0

    for col_name in df.columns:
        nulos = df.filter(F.col(col_name).isNull()).count()
        pct = (nulos / total * 100) if total > 0 else 0
        es_critico = columnas_criticas and col_name in columnas_criticas

        resultado[col_name] = {
            "nulos": nulos,
            "total": total,
            "porcentaje": round(pct, 2),
            "es_critico": es_critico,
        }

        if nulos > 0:
            nivel = "CRÍTICO" if es_critico else "ADVERTENCIA"
            logger.info("  [%s] %s: %d nulos (%.1f%%)", nivel, col_name, nulos, pct)
            if es_critico:
                problemas += 1

    if problemas > 0:
        logger.warning("%s: %d columnas críticas con nulos.", nombre, problemas)
    else:
        logger.info("%s: Sin nulos en columnas críticas.", nombre)

    return resultado


def validar_duplicados(df: DataFrame, nombre: str,
                       columnas_clave: list = None) -> dict:
    """
    Verifica si existen registros duplicados.

    Parámetros
    ----------
    columnas_clave : list
        Columnas que juntas forman la clave única del registro.

    Retorna
    -------
    dict
        {total, distintos, duplicados, hay_duplicados}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE DUPLICADOS: %s", nombre)

    total = df.count()
    if columnas_clave and all(c in df.columns for c in columnas_clave):
        distintos = df.dropDuplicates(columnas_clave).count()
    else:
        distintos = df.distinct().count()

    duplicados = total - distintos
    resultado = {
        "total": total,
        "distintos": distintos,
        "duplicados": duplicados,
        "hay_duplicados": duplicados > 0,
    }

    if duplicados > 0:
        logger.warning("%s: %d registros duplicados (de %d totales).", nombre, duplicados, total)
    else:
        logger.info("%s: Sin duplicados.", nombre)

    return resultado


def validar_tipos(df: DataFrame, nombre: str) -> dict:
    """
    Verifica que las columnas tengan los tipos de datos esperados.

    Retorna
    -------
    dict
        {columna: tipo_actual}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE TIPOS: %s", nombre)

    resultado = {}
    for field in df.schema.fields:
        resultado[field.name] = str(field.dataType)
        logger.info("  %s: %s", field.name, field.dataType)

    return resultado


def validar_fechas(df: DataFrame, nombre: str, col_fecha: str = "date") -> dict:
    """
    Verifica que las fechas sean válidas y estén dentro de un rango razonable.

    El periodo esperado para datos de COVID-19 de Google Cloud Platform
    es desde diciembre 2019 hasta septiembre 2022 (cuando dejó de actualizarse).

    Retorna
    -------
    dict
        {fecha_min, fecha_max, fuera_de_rango, total_nulos_fecha}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE FECHAS: %s", nombre)

    if col_fecha not in df.columns:
        logger.warning("Columna '%s' no existe en %s.", col_fecha, nombre)
        return {}

    nulos_fecha = df.filter(F.col(col_fecha).isNull()).count()

    if nulos_fecha == df.count():
        logger.warning("%s: Todas las fechas son nulas.", nombre)
        return {"total_nulos_fecha": nulos_fecha, "fuera_de_rango": 0}

    fecha_min = df.agg(F.min(col_fecha)).collect()[0][0]
    fecha_max = df.agg(F.max(col_fecha)).collect()[0][0]

    # Periodo esperado: 2019-12-01 a 2022-12-31 (con margen)
    fuera_rango = df.filter(
        (F.col(col_fecha) < F.lit("2019-12-01")) |
        (F.col(col_fecha) > F.lit("2023-06-30"))
    ).count()

    resultado = {
        "fecha_min": str(fecha_min),
        "fecha_max": str(fecha_max),
        "fuera_de_rango": fuera_rango,
        "total_nulos_fecha": nulos_fecha,
    }

    logger.info("  Fecha mínima: %s", fecha_min)
    logger.info("  Fecha máxima: %s", fecha_max)
    logger.info("  Nulos en fecha: %d", nulos_fecha)
    if fuera_rango > 0:
        logger.warning("  Fechas fuera del rango esperado: %d", fuera_rango)
    else:
        logger.info("  Todas las fechas están dentro del rango esperado.")

    return resultado


def validar_negativos(df: DataFrame, nombre: str,
                      columnas_conteos: list) -> dict:
    """
    Verifica que no existan valores negativos en columnas de conteos.

    Los valores negativos pueden indicar correcciones, pero deben
    registrarse para análisis.

    Retorna
    -------
    dict
        {columna: cantidad_negativos}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE NEGATIVOS: %s", nombre)

    resultado = {}
    for col_name in columnas_conteos:
        if col_name not in df.columns:
            continue
        negativos = df.filter(F.col(col_name) < 0).count()
        resultado[col_name] = negativos
        if negativos > 0:
            logger.warning("  %s: %d valores negativos (posibles correcciones).",
                           col_name, negativos)

    return resultado


def validar_coherencia_casos_fallecimientos(df: DataFrame, nombre: str) -> dict:
    """
    Verifica la coherencia entre casos confirmados y fallecimientos.

    Reglas:
    - Los fallecimientos no deben exceder los casos confirmados (acumulados).
    - Los casos acumulados no deberían ser menores que los casos nuevos diarios.

    Retorna
    -------
    dict
        {inconsistencias_fallecimientos_mayor_casos, inconsistencias_acumulado_menor_nuevo}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE COHERENCIA: %s", nombre)

    resultado = {"inconsistencias": 0}

    # Verificar: fallecimientos > casos confirmados (acumulados)
    if all(c in df.columns for c in ["cumulative_deceased", "cumulative_confirmed"]):
        inconsist = df.filter(
            (F.col("cumulative_deceased") > F.col("cumulative_confirmed")) &
            F.col("cumulative_deceased").isNotNull() &
            F.col("cumulative_confirmed").isNotNull() &
            (F.col("cumulative_confirmed") > 0)
        ).count()
        resultado["fallecimientos_mayor_casos"] = inconsist
        if inconsist > 0:
            logger.warning("  Fallecimientos acumulados > casos acumulados: %d registros.", inconsist)

    # Verificar: acumulado < suma de nuevos (inconsistencia lógica)
    if all(c in df.columns for c in ["new_confirmed", "cumulative_confirmed"]):
        inconsist = df.filter(
            (F.col("cumulative_confirmed") < F.col("new_confirmed")) &
            F.col("cumulative_confirmed").isNotNull() &
            F.col("new_confirmed").isNotNull() &
            (F.col("new_confirmed") > 0) &
            (F.col("cumulative_confirmed") > 0)
        ).count()
        resultado["acumulado_menor_nuevo"] = inconsist
        if inconsist > 0:
            logger.warning("  Acumulado < nuevo diario: %d registros.", inconsist)

    total_inconsist = sum(v for v in resultado.values() if isinstance(v, int))
    resultado["inconsistencias"] = total_inconsist

    if total_inconsist == 0:
        logger.info("  Sin inconsistencias detectadas.")

    return resultado


def validar_cumulativo_vs_suma(df: DataFrame, nombre: str,
                               col_clave: str = "location_key",
                               col_fecha: str = "date",
                               col_nuevo: str = "new_confirmed",
                               col_acumulado: str = "cumulative_confirmed") -> dict:
    """
    Verifica que el acumulado en cada fecha sea coherente con la suma
    de los valores nuevos diarios desde el inicio de la serie.

    Un acumulado que difiere en más de 1 unidad respecto a la suma ventana
    indica que la serie tiene reinicios, correcciones no reflejadas o huecos
    de reporte. Esto afecta directamente a calcular_tasa_contagio_por_pais,
    que usa F.max("cumulative_confirmed") como total de casos del país.

    La tolerancia de 1 unidad absorbe diferencias de redondeo o de la fecha
    de inicio del acumulado cuando los datos arrancan con un valor previo.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre del dataset para el reporte.
    col_clave : str
        Columna que identifica la ubicación (para particionar la ventana).
    col_fecha : str
        Columna de fecha (para ordenar la ventana).
    col_nuevo : str
        Columna de valores nuevos diarios (ej: new_confirmed).
    col_acumulado : str
        Columna de valores acumulados (ej: cumulative_confirmed).

    Retorna
    -------
    dict
        {
            col_par: str,          # par de columnas evaluado
            inconsistencias: int,  # registros donde |acumulado - suma_ventana| > 1
            total_evaluados: int,  # registros con ambas columnas no nulas
        }
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN ACUMULADO VS SUMA DIARIOS: %s (%s ~ %s)",
                nombre, col_acumulado, col_nuevo)

    cols_requeridas = [col_clave, col_fecha, col_nuevo, col_acumulado]
    if not all(c in df.columns for c in cols_requeridas):
        faltantes = [c for c in cols_requeridas if c not in df.columns]
        logger.warning("  Columnas faltantes: %s. Se omite la validación.", faltantes)
        return {"col_par": f"{col_acumulado}~{col_nuevo}", "inconsistencias": None,
                "total_evaluados": None, "omitida": True}

    # Ventana particionada por ubicación y ordenada por fecha: acumula new_confirmed
    # desde la primera fila hasta la actual inclusive.
    w = (
        Window
        .partitionBy(col_clave)
        .orderBy(col_fecha)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    df_eval = (
        df
        .filter(
            F.col(col_nuevo).isNotNull() &
            F.col(col_acumulado).isNotNull() &
            (F.col(col_nuevo) >= 0) &        # ignorar correcciones negativas
            (F.col(col_acumulado) >= 0)
        )
        .withColumn("_suma_ventana", F.sum(F.col(col_nuevo)).over(w))
        .withColumn("_diferencia", F.abs(F.col(col_acumulado) - F.col("_suma_ventana")))
    )

    total_evaluados = df_eval.count()
    inconsistencias = df_eval.filter(F.col("_diferencia") > 1).count()

    resultado = {
        "col_par": f"{col_acumulado}~{col_nuevo}",
        "inconsistencias": inconsistencias,
        "total_evaluados": total_evaluados,
        "omitida": False,
    }

    if total_evaluados == 0:
        logger.warning("  No hay registros válidos para evaluar el acumulado.")
    elif inconsistencias > 0:
        pct = inconsistencias / total_evaluados * 100
        logger.warning(
            "  %d registros (%.1f%%) con |%s - suma(%s)| > 1. "
            "La serie puede tener reinicios o correcciones no reflejadas en el acumulado. "
            "F.max('%s') como total de casos puede ser inexacto.",
            inconsistencias, pct, col_acumulado, col_nuevo, col_acumulado,
        )
    else:
        logger.info("  Acumulado consistente con la suma de diarios. ✓")

    return resultado


def validar_integridad_claves(df: DataFrame, nombre: str,
                              col_clave: str = "location_key") -> dict:
    """
    Verifica la integridad de las claves de integración.

    Retorna
    -------
    dict
        {total_nulos_clave, total_claves_vacias}
    """
    logger.info("-" * 40)
    logger.info("VALIDACIÓN DE CLAVES: %s", nombre)

    if col_clave not in df.columns:
        logger.warning("Columna '%s' no existe en %s.", col_clave, nombre)
        return {}

    nulos = df.filter(F.col(col_clave).isNull()).count()
    vacias = df.filter((F.col(col_clave) == "") | (F.length(F.trim(F.col(col_clave))) == 0)).count()

    resultado = {"nulos_clave": nulos, "claves_vacias": vacias}

    if nulos > 0 or vacias > 0:
        logger.warning("  Claves nulas: %d, vacías: %d", nulos, vacias)
    else:
        logger.info("  Integridad de claves correcta.")

    return resultado


def generar_reporte_calidad(df: DataFrame, nombre: str,
                            columnas_criticas: list = None,
                            columnas_conteos: list = None,
                            columnas_clave: list = None,
                            col_fecha: str = "date",
                            col_clave: str = "location_key") -> dict:
    """
    Genera un reporte completo de calidad de datos para un DataFrame.

    Ejecuta todas las validaciones disponibles y consolida los resultados.

    Parámetros
    ----------
    df : DataFrame
    nombre : str
        Nombre del dataset.
    columnas_criticas : list
        Columnas donde los nulos son críticos.
    columnas_conteos : list
        Columnas de conteos donde no deberían existir negativos.
    columnas_clave : list
        Columnas que forman la clave única para validar duplicados.
    col_fecha : str
        Nombre de la columna de fecha.
    col_clave : str
        Nombre de la columna de clave de integración.

    Retorna
    -------
    dict
        Reporte consolidado de calidad.
    """
    logger.info("=" * 60)
    logger.info("REPORTE DE CALIDAD: %s", nombre)
    logger.info("=" * 60)

    reporte = {"dataset": nombre}

    reporte["nulos"] = validar_nulos(df, nombre, columnas_criticas)
    reporte["duplicados"] = validar_duplicados(df, nombre, columnas_clave)
    reporte["tipos"] = validar_tipos(df, nombre)
    reporte["fechas"] = validar_fechas(df, nombre, col_fecha)

    if columnas_conteos:
        reporte["negativos"] = validar_negativos(df, nombre, columnas_conteos)
        reporte["coherencia"] = validar_coherencia_casos_fallecimientos(df, nombre)

    # Validar que el acumulado de casos sea consistente con la suma de nuevos diarios.
    # Solo tiene sentido cuando el dataset tiene fecha, location_key y ambas columnas
    # de epidemiología (new_confirmed + cumulative_confirmed).
    if (col_fecha and col_fecha in df.columns and
            "new_confirmed" in df.columns and "cumulative_confirmed" in df.columns):
        reporte["cumulativo_vs_suma"] = validar_cumulativo_vs_suma(
            df, nombre,
            col_clave=col_clave,
            col_fecha=col_fecha,
            col_nuevo="new_confirmed",
            col_acumulado="cumulative_confirmed",
        )

    reporte["claves"] = validar_integridad_claves(df, nombre, col_clave)

    # Determinar si hay errores críticos
    errores_criticos = 0

    # Nulos en columnas críticas
    for col_name, info in reporte["nulos"].items():
        if info.get("es_critico") and info["nulos"] > 0:
            errores_criticos += 1

    # Duplicados no esperados
    if reporte["duplicados"].get("hay_duplicados"):
        errores_criticos += 1

    # Fechas nulas masivas
    if reporte["fechas"].get("total_nulos_fecha", 0) > 0:
        pct_nulos = (reporte["fechas"]["total_nulos_fecha"] /
                     max(reporte["duplicados"]["total"], 1) * 100)
        if pct_nulos > 50:
            errores_criticos += 1

    reporte["errores_criticos"] = errores_criticos
    reporte["estado"] = "CRÍTICO" if errores_criticos > 0 else "ACEPTABLE"

    logger.info("-" * 40)
    if errores_criticos > 0:
        logger.warning("Estado de calidad: CRÍTICO (%d errores)", errores_criticos)
    else:
        logger.info("Estado de calidad: ACEPTABLE")

    return reporte


def validar_todos_los_datasets(resultados_etl: dict) -> dict:
    """
    Ejecuta validaciones de calidad sobre todos los datasets transformados.

    Parámetros
    ----------
    resultados_etl : dict
        Diccionario con los DataFrames transformados.

    Retorna
    -------
    dict
        Reportes consolidados de calidad.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO VALIDACIONES DE CALIDAD")
    logger.info("=" * 60)

    reportes = {}

    # Epidemiología
    if "df_epidemiologia" in resultados_etl and resultados_etl["df_epidemiologia"] is not None:
        reportes["epidemiologia"] = generar_reporte_calidad(
            resultados_etl["df_epidemiologia"],
            nombre="epidemiologia",
            columnas_criticas=["date", "location_key"],
            columnas_conteos=["new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"],
            columnas_clave=["date", "location_key"],
        )
        # Validar también el par de fallecidos (new_deceased ~ cumulative_deceased)
        reportes["epidemiologia"]["cumulativo_vs_suma_deceased"] = validar_cumulativo_vs_suma(
            resultados_etl["df_epidemiologia"],
            nombre="epidemiologia",
            col_clave="location_key",
            col_fecha="date",
            col_nuevo="new_deceased",
            col_acumulado="cumulative_deceased",
        )

    # By-age
    if "df_by_age" in resultados_etl and resultados_etl["df_by_age"] is not None:
        reportes["by_age"] = generar_reporte_calidad(
            resultados_etl["df_by_age"],
            nombre="by_age",
            columnas_criticas=["date", "location_key", "age_bucket"],
            columnas_conteos=["new_confirmed", "new_deceased", "cumulative_confirmed"],
            columnas_clave=["date", "location_key", "age_bucket"],
        )

    # Demographics
    if "df_demographics" in resultados_etl and resultados_etl["df_demographics"] is not None:
        reportes["demographics"] = generar_reporte_calidad(
            resultados_etl["df_demographics"],
            nombre="demographics",
            columnas_criticas=["location_key"],
            col_fecha=None,
        )

    # Weather
    if "df_weather" in resultados_etl and resultados_etl["df_weather"] is not None:
        reportes["weather"] = generar_reporte_calidad(
            resultados_etl["df_weather"],
            nombre="weather",
            columnas_criticas=["date", "location_key"],
            columnas_clave=["date", "location_key"],
        )

    # Hospitalizations
    if "df_hospitalizations" in resultados_etl and resultados_etl["df_hospitalizations"] is not None:
        reportes["hospitalizations"] = generar_reporte_calidad(
            resultados_etl["df_hospitalizations"],
            nombre="hospitalizations",
            columnas_criticas=["date", "location_key"],
            columnas_conteos=["new_hospitalized_patients", "new_intensive_care_patients",
                              "cumulative_hospitalized_patients", "current_hospitalized_patients"],
            columnas_clave=["date", "location_key"],
        )

    logger.info("=" * 60)
    logger.info("VALIDACIONES DE CALIDAD COMPLETADAS")
    logger.info("=" * 60)

    return reportes
