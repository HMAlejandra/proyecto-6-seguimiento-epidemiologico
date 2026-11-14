"""
extraccion.py
-------------
Descarga de datos desde las fuentes oficiales.

Fuentes principales:
- Google Cloud Platform — COVID-19 Open Data (v3)
  Repositorio histórico, dejó de actualizarse en septiembre de 2022.

Responsabilidades:
1. Descargar los archivos de Google Cloud Platform.
2. Comprobar que las descargas sean exitosas.
3. Guardar los archivos originales en data/raw/google/.
4. Evitar descargar nuevamente archivos si ya existen y son válidos.
5. Manejar errores de conexión.
6. Registrar las descargas en los logs.
"""

import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas y configuración
# ---------------------------------------------------------------------------

# Raíz del proyecto (sube dos niveles desde etl/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorios de datos crudos
RAW_GOOGLE_DIR = PROJECT_ROOT / "data" / "raw" / "google"
RAW_WHO_DIR = PROJECT_ROOT / "data" / "raw" / "who"
LOGS_DIR = PROJECT_ROOT / "logs"

# URLs de Google Cloud Platform — COVID-19 Open Data (v3)
# Repositorio: https://github.com/GoogleCloudPlatform/covid-19-open-data
# Estos archivos dejaron de actualizarse en septiembre de 2022.
GOOGLE_SOURCES = {
    "epidemiology": {
        "url": "https://storage.googleapis.com/covid19-open-data/v3/epidemiology.csv",
        "description": "Datos epidemiológicos: casos nuevos, acumulados, fallecimientos.",
    },
    "by-age": {
        "url": "https://storage.googleapis.com/covid19-open-data/v3/by-age.csv",
        "description": "Casos y fallecimientos desglosados por grupo etario.",
    },
    "demographics": {
        "url": "https://storage.googleapis.com/covid19-open-data/v3/demographics.csv",
        "description": "Datos demográficos por ubicación: población, densidad, etc.",
    },
    "index": {
        "url": "https://storage.googleapis.com/covid19-open-data/v3/index.csv",
        "description": "Índice geográfico: códigos de ubicación, nombres de países/regiones.",
    },
    "weather": {
        "url": "https://storage.googleapis.com/covid19-open-data/v3/weather.csv",
        "description": "Variables climáticas: temperatura, precipitación, humedad.",
    },
    "hospitalizations": {
        "url": "https://storage.googleapis.com/covid19-open-data/v3/hospitalizations.csv",
        "description": "Hospitalizaciones, UCI, casos graves.",
    },
}

# Tamaño mínimo aceptable para considerar un archivo como válido (bytes).
# Los archivos de esta fuente son grandes; si uno pesa menos de 1 KB
# probablemente sea un error de descarga.
MIN_FILE_SIZE_BYTES = 1024  # 1 KB

# Tiempo máximo de espera por respuesta (segundos).
REQUEST_TIMEOUT = 300  # 5 minutos — algunos archivos son muy grandes


def _ensure_dirs():
    """Crea los directorios necesarios si no existen."""
    RAW_GOOGLE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_WHO_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _is_file_valid(filepath: Path) -> bool:
    """
    Verifica si un archivo ya existe y tiene un tamaño razonable.

    Parámetros
    ----------
    filepath : Path
        Ruta del archivo a verificar.

    Retorna
    -------
    bool
        True si el archivo existe y su tamaño es >= MIN_FILE_SIZE_BYTES.
    """
    if not filepath.exists():
        return False
    size = filepath.stat().st_size
    if size < MIN_FILE_SIZE_BYTES:
        logger.warning("Archivo %s existe pero pesa solo %d bytes (muy pequeño).", filepath, size)
        return False
    return True


def descargar_archivo(nombre: str, url: str, destino: Path, forzar: bool = False) -> bool:
    """
    Descarga un archivo desde una URL y lo guarda en destino.

    Si el archivo ya existe y es válido, no lo descarga nuevamente
    (salvo que forzar=True).

    Parámetros
    ----------
    nombre : str
        Nombre identificador del archivo (para logs).
    url : str
        URL de descarga.
    destino : Path
        Ruta local donde guardar el archivo.
    forzar : bool
        Si True, descarga incluso si el archivo ya existe y es válido.

    Retorna
    -------
    bool
        True si el archivo existe y es válido al finalizar, False si falló.
    """
    # Verificar si ya existe y es válido
    if not forzar and _is_file_valid(destino):
        size_mb = destino.stat().st_size / (1024 * 1024)
        logger.info("Archivo '%s' ya existe (%.2f MB). Se omite la descarga.", nombre, size_mb)
        return True

    logger.info("Descargando '%s' desde %s ...", nombre, url)

    try:
        respuesta = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        respuesta.raise_for_status()

        # Escribir el archivo en bloques para no cargar todo en memoria
        with open(destino, "wb") as f:
            for chunk in respuesta.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = destino.stat().st_size / (1024 * 1024)
        logger.info("Descarga exitosa: '%s' -> %s (%.2f MB)", nombre, destino, size_mb)
        return True

    except requests.exceptions.ConnectionError as e:
        logger.error("Error de conexión al descargar '%s': %s", nombre, e)
        return False
    except requests.exceptions.HTTPError as e:
        logger.error("Error HTTP al descargar '%s': %s", nombre, e)
        return False
    except requests.exceptions.Timeout:
        logger.error("Timeout al descargar '%s' (límite: %d segundos).", nombre, REQUEST_TIMEOUT)
        return False
    except OSError as e:
        logger.error("Error de escritura al guardar '%s': %s", nombre, e)
        return False


def descargar_google(forzar: bool = False) -> dict:
    """
    Descarga todos los archivos de Google Cloud Platform.

    Parámetros
    ----------
    forzar : bool
        Si True, descarga nuevamente incluso los archivos que ya existen.

    Retorna
    -------
    dict
        Diccionario con el estado de cada archivo:
        {nombre: {"exitoso": bool, "ruta": str, "tamano_mb": float}}
    """
    _ensure_dirs()

    resultados = {}

    for nombre, info in GOOGLE_SOURCES.items():
        url = info["url"]
        destino = RAW_GOOGLE_DIR / f"{nombre}.csv"

        exitoso = descargar_archivo(nombre, url, destino, forzar=forzar)

        tamano_mb = 0.0
        if destino.exists():
            tamano_mb = destino.stat().st_size / (1024 * 1024)

        resultados[nombre] = {
            "exitoso": exitoso,
            "ruta": str(destino),
            "tamano_mb": round(tamano_mb, 2),
            "descripcion": info["description"],
        }

    # Resumen en log
    exitosos = sum(1 for r in resultados.values() if r["exitoso"])
    total = len(resultados)
    logger.info("Descarga Google: %d/%d archivos exitosos.", exitosos, total)

    return resultados


def investigar_who() -> dict:
    """
    Investiga la disponibilidad de datos descargables de la OMS.

    La OMS publica datos de COVID-19 en:
    https://data.who.int/dashboards/covid19/data

    Tras investigación, la OMS ofrece datos a través de su API:
    https://ghoapi.azureedge.net/api/Indicator
    y dashboards interactivos, pero no ofrece archivos CSV descargables
    directamente con la misma estructura que Google Cloud Platform.

    La integración con Google Cloud Platform no es directa porque:
    - Las definiciones de variables difieren.
    - La granularidad temporal y geográfica es distinta.
    - Las claves de integración (códigos de país) usan sistemas diferentes.

    Por lo tanto, se documenta la limitación y se mantiene Google Cloud
    Platform como fuente principal.

    Retorna
    -------
    dict
        Diccionario con el resultado de la investigación.
    """
    logger.info("Investigando disponibilidad de datos de la OMS...")

    resultado = {
        "fuente": "Organización Mundial de la Salud (OMS)",
        "url_portal": "https://data.who.int/dashboards/covid19/data",
        "encontrada": False,
        "motivo": (
            "La OMS ofrece un dashboard interactivo y una API (GHO API), "
            "pero no proporciona archivos CSV descargables directamente con "
            "una estructura compatible con Google Cloud Platform. "
            "Las definiciones de variables, granularidad y claves de "
            "integración difieren entre fuentes. "
            "Se mantiene Google Cloud Platform como fuente principal."
        ),
        "archivos_descargados": [],
    }

    logger.warning("OMS: %s", resultado["motivo"])

    return resultado


def descargar_todas_las_fuentes(forzar: bool = False) -> dict:
    """
    Coordina la descarga de todas las fuentes disponibles.

    Parámetros
    ----------
    forzar : bool
        Si True, descarga nuevamente incluso los archivos que ya existen.

    Retorna
    -------
    dict
        Resultado consolidado de todas las descargas.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO DESCARGA DE FUENTES DE DATOS")
    logger.info("=" * 60)

    resultados = {
        "google": descargar_google(forzar=forzar),
        "who": investigar_who(),
    }

    # Resumen general
    google_ok = sum(1 for r in resultados["google"].values() if r["exitoso"])
    google_total = len(resultados["google"])

    logger.info("-" * 60)
    logger.info("RESUMEN DE DESCARGAS:")
    logger.info("  Google Cloud Platform: %d/%d archivos exitosos", google_ok, google_total)
    logger.info("  OMS: Sin archivos descargables (ver logs)")
    logger.info("=" * 60)

    return resultados


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configurar logging básico para ejecución directa
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(LOGS_DIR / "extraccion.log"), encoding="utf-8"),
        ],
    )

    descargar_todas_las_fuentes()
