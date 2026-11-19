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
   
    if not filepath.exists():
        return False
    size = filepath.stat().st_size
    if size < MIN_FILE_SIZE_BYTES:
        logger.warning("Archivo %s existe pero pesa solo %d bytes (muy pequeño).", filepath, size)
        return False
    return True


def descargar_archivo(nombre: str, url: str, destino: Path, forzar: bool = False) -> bool:
    
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


def descargar_todas_las_fuentes(forzar: bool = False) -> dict:
    
    logger.info("=" * 60)
    logger.info("INICIANDO DESCARGA DE FUENTES DE DATOS")
    logger.info("=" * 60)

    resultados = {
        "google": descargar_google(forzar=forzar)
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
