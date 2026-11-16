"""
anova.py
--------
Análisis estadístico ANOVA para comparar tasas de contagio entre grupos etarios.

Pregunta de investigación:
    ¿Existen diferencias estadísticamente significativas entre las tasas
    de contagio de los diferentes grupos etarios?

Hipótesis:
    H0: Las medias de las tasas de contagio de los diferentes grupos etarios
        son iguales.
    H1: Al menos una de las medias de las tasas de contagio de los grupos
        etarios es diferente.

Nivel de significancia: α = 0.05

Pruebas ejecutadas:
1. Shapiro-Wilk para evaluar normalidad de los residuos.
2. Levene para evaluar homogeneidad de varianzas.
3. ANOVA clásico (one-way) cuando se cumplen los supuestos.
4. Welch ANOVA cuando hay heterogeneidad de varianzas.
5. Kruskal-Wallis cuando el análisis no paramétrico es más apropiado.
6. Tukey HSD para comparaciones posteriores (cuando corresponde).

NO se inventan resultados, valores F, valores p ni conclusiones.
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

logger = logging.getLogger(__name__)

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)


def _guardar_resultado(nombre: str, contenido: str):
    """Guarda un resultado en un archivo de texto en la carpeta resultados/."""
    ruta = RESULTADOS_DIR / f"{nombre}.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    logger.info("Resultado guardado en: %s", ruta)


def probar_normalidad(grupos: dict) -> dict:
    """
    Evalúa la normalidad de los residuos usando Shapiro-Wilk.

    Como las muestras pueden ser grandes, Shapiro-Wilk puede ser muy sensible.
    Se aplica a una muestra aleatoria de como máximo 5000 observaciones por grupo
    (Shapiro-Wilk pierde potencia con muestras muy grandes).

    Parámetros
    ----------
    grupos : dict
        {nombre_grupo: numpy.array de valores}

    Retorna
    -------
    dict
        {grupo: {estadistico_shapiro, p_valor, es_normal}}
    """
    logger.info("-" * 40)
    logger.info("PRUEBA DE NORMALIDAD (Shapiro-Wilk)")

    resultados = {}
    max_muestra_shapiro = 5000  # Shapiro-Wilk pierde potencia con muestras grandes

    for nombre, valores in grupos.items():
        if len(valores) < 3:
            logger.warning("  Grupo '%s': muy pocos datos para Shapiro-Wilk (n=%d).",
                          nombre, len(valores))
            resultados[nombre] = {"estadistico": None, "p_valor": None, "es_normal": False}
            continue

        # Tomar muestra aleatoria si es muy grande
        if len(valores) > max_muestra_shapiro:
            np.random.seed(42)
            muestra = np.random.choice(valores, size=max_muestra_shapiro, replace=False)
        else:
            muestra = valores

        try:
            estadistico, p_valor = stats.shapiro(muestra)
            es_normal = p_valor > 0.05
            resultados[nombre] = {
                "estadistico": float(estadistico),
                "p_valor": float(p_valor),
                "es_normal": es_normal,
                "n_muestra": len(muestra),
            }
            logger.info("  Grupo '%s': W=%.4f, p=%.4f, normal=%s (n=%d)",
                        nombre, estadistico, p_valor, es_normal, len(muestra))
        except Exception as e:
            logger.error("  Error en Shapiro-Wilk para '%s': %s", nombre, e)
            resultados[nombre] = {"estadistico": None, "p_valor": None, "es_normal": False}

    return resultados


def probar_homogeneidad_varianzas(grupos: dict) -> dict:
    """
    Evalúa la homogeneidad de varianzas usando la prueba de Levene.

    Parámetros
    ----------
    grupos : dict
        {nombre_grupo: numpy.array de valores}

    Retorna
    -------
    dict
        {estadistico_levene, p_valor, varianzas_homogeneas}
    """
    logger.info("-" * 40)
    logger.info("PRUEBA DE HOMOGENEIDAD DE VARIANZAS (Levene)")

    valores_grupos = list(grupos.values())

    if len(valores_grupos) < 2:
        logger.warning("Se necesitan al menos 2 grupos para Levene.")
        return {"estadistico": None, "p_valor": None, "varianzas_homogeneas": False}

    try:
        estadistico, p_valor = stats.levene(*valores_grupos, center="median")
        varianzas_homogeneas = p_valor > 0.05
        resultado = {
            "estadistico": float(estadistico),
            "p_valor": float(p_valor),
            "varianzas_homogeneas": varianzas_homogeneas,
        }
        logger.info("  Levene: W=%.4f, p=%.4f, homogeneas=%s",
                    estadistico, p_valor, varianzas_homogeneas)
        return resultado
    except Exception as e:
        logger.error("  Error en Levene: %s", e)
        return {"estadistico": None, "p_valor": None, "varianzas_homogeneas": False}


def ejecutar_anova_clasico(grupos: dict) -> dict:
    """
    Ejecuta ANOVA clásico de un factor (one-way ANOVA).

    Parámetros
    ----------
    grupos : dict
        {nombre_grupo: numpy.array de valores}

    Retorna
    -------
    dict
        {estadistico_F, p_valor, grados_libertad_entre, grados_libertad_dentro}
    """
    logger.info("-" * 40)
    logger.info("ANOVA CLÁSICO (One-Way)")

    try:
        f_stat, p_valor = stats.f_oneway(*grupos.values())

        # Calcular grados de libertad
        k = len(grupos)  # número de grupos
        n_total = sum(len(v) for v in grupos.values())  # total de observaciones
        gl_entre = k - 1
        gl_dentro = n_total - k

        resultado = {
            "estadistico_F": float(f_stat),
            "p_valor": float(p_valor),
            "grados_libertad_entre": gl_entre,
            "grados_libertad_dentro": gl_dentro,
            "n_total": n_total,
            "n_grupos": k,
        }

        logger.info("  F=%.4f, p=%.6f, gl(%d, %d)",
                    f_stat, p_valor, gl_entre, gl_dentro)

        return resultado

    except Exception as e:
        logger.error("  Error en ANOVA clásico: %s", e)
        return {}


def ejecutar_welch_anova(grupos: dict) -> dict:
    """
    Ejecuta Welch ANOVA (robusto a heterogeneidad de varianzas).

    Retorna
    -------
    dict
        {estadistico_F, p_valor}
    """
    logger.info("-" * 40)
    logger.info("WELCH ANOVA")

    try:
        # Welch ANOVA usando scipy
        from scipy.stats import f_oneway

        # Implementación de Welch ANOVA manual
        k = len(grupos)
        n_total = sum(len(v) for v in grupos.values())

        # Medias y varianzas por grupo
        medias = [np.mean(v) for v in grupos.values()]
        varianzas = [np.var(v, ddof=1) for v in grupos.values()]
        ns = [len(v) for v in grupos.values()]

        # Media ponderada
        w = [n / var for n, var in zip(ns, varianzas) if var > 0]
        if not w:
            logger.error("  Varianzas son cero. No se puede ejecutar Welch ANOVA.")
            return {}

        media_ponderada = sum(wi * mi for wi, mi in zip(w, medias)) / sum(w)

        # Estadístico de Welch
        numerador = sum(wi * (mi - media_ponderada) ** 2 for wi, mi in zip(w, medias)) / (k - 1)
        denominador = (1 + (2 * (k - 2) / (k ** 2 - 1)) *
                       (sum((1 - wi / sum(w)) ** 2 / (ni - 1) for wi, ni in zip(w, ns))))

        if denominador == 0:
            logger.error("  Denominador es cero en Welch ANOVA.")
            return {}

        f_stat = numerador / denominador

        # Grados de libertad
        gl1 = k - 1
        gl2 = (k ** 2 - 1) / (3 * sum((1 - wi / sum(w)) ** 2 / (ni - 1) for wi, ni in zip(w, ns)))

        p_valor = 1 - stats.f.cdf(f_stat, gl1, gl2)

        resultado = {
            "estadistico_F": float(f_stat),
            "p_valor": float(p_valor),
            "gl1": float(gl1),
            "gl2": float(gl2),
        }

        logger.info("  Welch F=%.4f, p=%.6f, gl(%.1f, %.1f)",
                    f_stat, p_valor, gl1, gl2)

        return resultado

    except Exception as e:
        logger.error("  Error en Welch ANOVA: %s", e)
        return {}


def ejecutar_kruskal_wallis(grupos: dict) -> dict:
    """
    Ejecuta la prueba no paramétrica de Kruskal-Wallis.

    Retorna
    -------
    dict
        {estadistico_H, p_valor}
    """
    logger.info("-" * 40)
    logger.info("KRUSKAL-WALLIS (No Paramétrico)")

    try:
        h_stat, p_valor = stats.kruskal(*grupos.values())
        resultado = {
            "estadistico_H": float(h_stat),
            "p_valor": float(p_valor),
        }
        logger.info("  H=%.4f, p=%.6f", h_stat, p_valor)
        return resultado
    except Exception as e:
        logger.error("  Error en Kruskal-Wallis: %s", e)
        return {}


def ejecutar_tukey_hsd(grupos: dict) -> dict:
    """
    Ejecuta comparaciones posteriores con Tukey HSD.

    Parámetros
    ----------
    grupos : dict
        {nombre_grupo: numpy.array de valores}

    Retorna
    -------
    dict
        Resultados de las comparaciones pairwise.
    """
    logger.info("-" * 40)
    logger.info("COMPARACIONES POSTERIORES (Tukey HSD)")

    try:
        # Preparar datos para Tukey
        todos_valores = []
        todos_grupos = []
        for nombre, valores in grupos.items():
            for v in valores:
                todos_valores.append(v)
                todos_grupos.append(nombre)

        tukey = pairwise_tukeyhsd(todos_valores, todos_grupos, alpha=0.05)

        resultado = {
            "tabla": str(tukey),
            "resumen": tukey.summary().as_text(),
        }
        logger.info("\n%s", tukey.summary())
        return resultado

    except Exception as e:
        logger.error("  Error en Tukey HSD: %s", e)
        return {}


def ejecutar_anova(datos_preparados: dict) -> dict:
    """
    Ejecuta el análisis ANOVA completo con verificación de supuestos.

    Flujo:
    1. Verificar que hay datos suficientes.
    2. Probar normalidad (Shapiro-Wilk).
    3. Probar homogeneidad de varianzas (Levene).
    4. Si se cumplen supuestos → ANOVA clásico + Tukey HSD.
    5. Si no hay homogeneidad → Welch ANOVA.
    6. Si no hay normalidad → Kruskal-Wallis.
    7. Guardar resultados.

    Parámetros
    ----------
    datos_preparados : dict
        Diccionario con 'grupos' y 'metadatos' de preparar_datos_anova().

    Retorna
    -------
    dict
        Resultados completos del análisis.
    """
    logger.info("=" * 60)
    logger.info("ANÁLISIS ESTADÍSTICO ANOVA")
    logger.info("=" * 60)

    grupos = datos_preparados["grupos"]
    metadatos = datos_preparados["metadatos"]

    # Información del análisis
    info = f"""
================================================================
ANÁLISIS ESTADÍSTICO ANOVA — PROYECTO 6
================================================================

Pregunta de investigación:
    ¿Existen diferencias estadísticamente significativas entre las
    tasas de contagio de los diferentes grupos etarios?

Hipótesis nula (H0):
    Las medias de las tasas de contagio de los diferentes grupos
    etarios son iguales.

Hipótesis alternativa (H1):
    Al menos una de las medias de las tasas de contagio de los grupos
    etarios es diferente.

Nivel de significancia: α = 0.05

Variable analizada: {metadatos['variable']}
Tipo de variable: {metadatos['tipo']}
Número de grupos: {metadatos['n_grupos']}
Observaciones totales: {metadatos['n_observaciones_total']}

Tamaño de muestra por grupo:
"""
    for nombre, info_grupo in metadatos["grupos_info"].items():
        info += f"    {nombre}: n={info_grupo['n_observaciones']}, media={info_grupo['media']:.2f}, std={info_grupo['desviacion_estandar']:.2f}\n"

    info += "\nLimitaciones:\n"
    for lim in metadatos["limitaciones"]:
        info += f"    - {lim}\n"

    logger.info(info)

    resultados = {
        "info": info,
        "hipotesis": {
            "h0": "Las medias de las tasas de contagio de los diferentes grupos etarios son iguales.",
            "h1": "Al menos una de las medias de las tasas de contagio de los grupos etarios es diferente.",
            "alfa": 0.05,
        },
        "metadatos": metadatos,
    }

    # 1. Prueba de normalidad
    resultados["normalidad"] = probar_normalidad(grupos)
    todos_normales = all(
        r.get("es_normal", False) for r in resultados["normalidad"].values()
    )

    # 2. Prueba de homogeneidad de varianzas
    resultados["homogeneidad"] = probar_homogeneidad_varianzas(grupos)
    varianzas_homogeneas = resultados["homogeneidad"].get("varianzas_homogeneas", False)

    # 3. Decidir qué prueba usar
    if todos_normales and varianzas_homogeneas:
        # ANOVA clásico + Tukey HSD
        logger.info("Supuestos cumplidos. Ejecutando ANOVA clásico.")
        resultados["anova_clasico"] = ejecutar_anova_clasico(grupos)
        resultados["tukey_hsd"] = ejecutar_tukey_hsd(grupos)
        resultados["prueba_principal"] = "ANOVA clásico"
    elif todos_normales and not varianzas_homogeneas:
        # Welch ANOVA
        logger.info("Varianzas no homogéneas. Ejecutando Welch ANOVA.")
        resultados["welch_anova"] = ejecutar_welch_anova(grupos)
        resultados["anova_clasico"] = ejecutar_anova_clasico(grupos)  # También para comparar
        resultados["prueba_principal"] = "Welch ANOVA"
    else:
        # Kruskal-Wallis
        logger.info("Supuestos no cumplidos. Ejecutando Kruskal-Wallis.")
        resultados["kruskal_wallis"] = ejecutar_kruskal_wallis(grupos)
        # También ejecutar ANOVA clásico y Welch para referencia
        resultados["anova_clasico"] = ejecutar_anova_clasico(grupos)
        resultados["welch_anova"] = ejecutar_welch_anova(grupos)
        resultados["prueba_principal"] = "Kruskal-Wallis"

    # 4. Decisión estadística
    prueba_principal = resultados["prueba_principal"]
    p_valor_principal = None

    if prueba_principal == "ANOVA clásico" and "anova_clasico" in resultados:
        p_valor_principal = resultados["anova_clasico"].get("p_valor")
    elif prueba_principal == "Welch ANOVA" and "welch_anova" in resultados:
        p_valor_principal = resultados["welch_anova"].get("p_valor")
    elif prueba_principal == "Kruskal-Wallis" and "kruskal_wallis" in resultados:
        p_valor_principal = resultados["kruskal_wallis"].get("p_valor")

    if p_valor_principal is not None:
        decision = "Rechazar H0" if p_valor_principal < 0.05 else "No rechazar H0"
        resultados["decision"] = {
            "prueba": prueba_principal,
            "p_valor": p_valor_principal,
            "decision": decision,
            "interpretacion": (
                f"Con un nivel de significancia α=0.05, el valor p={p_valor_principal:.6f} "
                f"{'es menor que' if p_valor_principal < 0.05 else 'es mayor que'} α, "
                f"por lo que {'se rechaza' if p_valor_principal < 0.05 else 'no se rechaza'} "
                f"la hipótesis nula. "
                + ("Existen diferencias estadísticamente significativas entre las medias "
                   "de los grupos etarios." if p_valor_principal < 0.05 else
                   "No hay evidencia suficiente para afirmar que existen diferencias "
                   "significativas entre las medias de los grupos etarios.")
            ),
        }
        logger.info("DECISIÓN: %s (p=%.6f)", decision, p_valor_principal)
    else:
        resultados["decision"] = {"error": "No se pudo calcular el valor p principal."}

    # 5. Advertencia metodológica sobre conteos vs tasas.
    # Solo se muestra si preparacion_anova no pudo calcular tasas reales
    # (usar_tasas=False). Si se calcularon tasas aproximadas con demographics,
    # emitir esta advertencia sería factualmente incorrecto.
    usar_tasas = metadatos.get("usar_tasas", False)
    if not usar_tasas:
        advertencia_conteos = (
            "\n⚠ ADVERTENCIA METODOLÓGICA — CONTEOS VS TASAS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Este análisis usa new_confirmed (conteos absolutos de casos), NO tasas de contagio.\n"
            "La pregunta de investigación menciona 'tasas de contagio por grupo etario', pero\n"
            "el cruce con demographics no fue posible o no produjo denominadores válidos.\n\n"
            "Implicación: grupos etarios más numerosos (ej. adultos de 30-50 años) tendrán\n"
            "sistemáticamente más casos que grupos menos numerosos (ej. mayores de 80),\n"
            "incluso si su riesgo relativo de contagio es menor. Un resultado estadísticamente\n"
            "significativo puede deberse a diferencias de tamaño poblacional entre grupos,\n"
            "no a diferencias reales de riesgo epidemiológico.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        logger.warning(advertencia_conteos)
        resultados["advertencia_metodologica"] = advertencia_conteos.strip()
    else:
        nota_tasas = (
            "\nℹ VARIABLE ANALIZADA: tasas aproximadas por 100 000 hab.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Se calcularon tasas aproximadas usando denominadores de demographics.csv.\n"
            "Los denominadores son proporciones de la población total del país, NO conteos\n"
            "absolutos por grupo etario; las tasas son aproximaciones documentadas.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        logger.info(nota_tasas)
        resultados["advertencia_metodologica"] = nota_tasas.strip()

    # 6. Guardar resultados
    _guardar_resultado_completo(resultados)

    logger.info("=" * 60)
    logger.info("ANÁLISIS ANOVA COMPLETADO")
    logger.info("=" * 60)

    return resultados


def _guardar_resultado_completo(resultados: dict):
    """Guarda todos los resultados en un archivo de texto."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"resultados_anova_{timestamp}"

    contenido = []
    contenido.append(resultados.get("info", ""))
    contenido.append("\n" + "=" * 60 + "\n")
    contenido.append("RESULTADOS DETALLADOS\n")
    contenido.append("=" * 60 + "\n\n")

    # Normalidad
    contenido.append("1. PRUEBA DE NORMALIDAD (Shapiro-Wilk)\n")
    contenido.append("-" * 40 + "\n")
    for grupo, res in resultados.get("normalidad", {}).items():
        if res.get("estadistico") is not None:
            contenido.append(
                f"  {grupo}: W={res['estadistico']:.4f}, "
                f"p={res['p_valor']:.4f}, "
                f"normal={'Sí' if res['es_normal'] else 'No'} "
                f"(n_muestra={res.get('n_muestra', '?')})\n"
            )
        else:
            contenido.append(f"  {grupo}: No se pudo calcular\n")
    contenido.append("\n")

    # Homogeneidad
    contenido.append("2. PRUEBA DE HOMOGENEIDAD DE VARIANZAS (Levene)\n")
    contenido.append("-" * 40 + "\n")
    hom = resultados.get("homogeneidad", {})
    if hom.get("estadistico") is not None:
        contenido.append(
            f"  W={hom['estadistico']:.4f}, "
            f"p={hom['p_valor']:.4f}, "
            f"varianzas_homogeneas={'Sí' if hom['varianzas_homogeneas'] else 'No'}\n"
        )
    else:
        contenido.append("  No se pudo calcular\n")
    contenido.append("\n")

    # ANOVA clásico
    contenido.append("3. ANOVA CLÁSICO (One-Way)\n")
    contenido.append("-" * 40 + "\n")
    anova_c = resultados.get("anova_clasico", {})
    if anova_c:
        contenido.append(
            f"  F={anova_c['estadistico_F']:.4f}, "
            f"p={anova_c['p_valor']:.6f}, "
            f"gl({anova_c['grados_libertad_entre']}, {anova_c['grados_libertad_dentro']})\n"
        )
    else:
        contenido.append("  No se ejecutó\n")
    contenido.append("\n")

    # Welch ANOVA
    if "welch_anova" in resultados and resultados["welch_anova"]:
        contenido.append("4. WELCH ANOVA\n")
        contenido.append("-" * 40 + "\n")
        welch = resultados["welch_anova"]
        if welch:
            contenido.append(
                f"  F={welch['estadistico_F']:.4f}, "
                f"p={welch['p_valor']:.6f}, "
                f"gl({welch['gl1']:.1f}, {welch['gl2']:.1f})\n"
            )
        else:
            contenido.append("  No se pudo calcular\n")
        contenido.append("\n")

    # Kruskal-Wallis
    if "kruskal_wallis" in resultados and resultados["kruskal_wallis"]:
        contenido.append("5. KRUSKAL-WALLIS\n")
        contenido.append("-" * 40 + "\n")
        kw = resultados["kruskal_wallis"]
        if kw:
            contenido.append(
                f"  H={kw['estadistico_H']:.4f}, "
                f"p={kw['p_valor']:.6f}\n"
            )
        else:
            contenido.append("  No se pudo calcular\n")
        contenido.append("\n")

    # Tukey HSD
    if "tukey_hsd" in resultados and resultados["tukey_hsd"]:
        contenido.append("6. COMPARACIONES POSTERIORES (Tukey HSD)\n")
        contenido.append("-" * 40 + "\n")
        contenido.append(resultados["tukey_hsd"]["resumen"])
        contenido.append("\n")

    # Decisión
    contenido.append("=" * 60 + "\n")
    contenido.append("DECISIÓN ESTADÍSTICA\n")
    contenido.append("=" * 60 + "\n\n")
    decision = resultados.get("decision", {})
    contenido.append(f"Prueba principal: {resultados.get('prueba_principal', '?')}\n")
    if decision.get("p_valor") is not None:
        contenido.append(f"Valor p: {decision['p_valor']:.6f}\n")
        contenido.append(f"Decisión: {decision['decision']}\n")
        contenido.append(f"\nInterpretación:\n{decision['interpretacion']}\n")
    elif decision.get("error"):
        contenido.append(f"Error: {decision['error']}\n")

    contenido.append("\n" + "=" * 60 + "\n")
    contenido.append("ADVERTENCIA METODOLÓGICA\n")
    contenido.append("=" * 60 + "\n")
    contenido.append(
        "Este análisis usa new_confirmed (conteos absolutos), NO tasas de contagio.\n"
        "by-age.csv no incluye denominadores poblacionales por grupo etario.\n"
        "Un resultado significativo puede deberse a diferencias de tamaño poblacional\n"
        "entre grupos, no a diferencias reales de riesgo epidemiológico.\n"
    )

    contenido.append("\n" + "=" * 60 + "\n")
    contenido.append("LIMITACIONES\n")
    contenido.append("=" * 60 + "\n")
    for lim in resultados.get("metadatos", {}).get("limitaciones", []):
        contenido.append(f"- {lim}\n")

    # Guardar
    texto = "".join(contenido)
    _guardar_resultado(nombre_archivo, texto)
    _guardar_resultado("resultados_anova_ultimo", texto)

    logger.info("Resultados guardados en: %s/%s.txt", RESULTADOS_DIR, nombre_archivo)
