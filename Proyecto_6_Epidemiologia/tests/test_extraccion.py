"""
test_extraccion.py
-------------------
Pruebas automatizadas para el módulo de extracción.

Verifica:
- Que los archivos puedan leerse.
- Que las rutas de descarga sean correctas.
- Que la función de validación de archivos funcione.
- Que el manejo de errores funcione correctamente.
"""

import os
import tempfile
from pathlib import Path

import pytest

from etl.extraccion import (
    GOOGLE_SOURCES,
    _is_file_valid,
    descargar_archivo,
    RAW_GOOGLE_DIR,
    _ensure_dirs,
)


class TestFuentes:
    """Pruebas sobre la definición de fuentes."""

    def test_google_sources_tiene_6_fuentes(self):
        """Verifica que hay exactamente 6 fuentes de Google."""
        assert len(GOOGLE_SOURCES) == 6, f"Se esperaban 6 fuentes, hay {len(GOOGLE_SOURCES)}"

    def test_google_sources_tienen_url(self):
        """Verifica que todas las fuentes tienen URL."""
        for nombre, info in GOOGLE_SOURCES.items():
            assert "url" in info, f"Fuente '{nombre}' no tiene URL"
            assert info["url"].startswith("https://"), f"URL de '{nombre}' no es HTTPS"

    def test_google_sources_tienen_descripcion(self):
        """Verifica que todas las fuentes tienen descripción."""
        for nombre, info in GOOGLE_SOURCES.items():
            assert "description" in info, f"Fuente '{nombre}' no tiene descripción"

    def test_nombres_fuentes_esperados(self):
        """Verifica que los nombres de las fuentes son los esperados."""
        esperados = {"epidemiology", "by-age", "demographics", "index", "weather", "hospitalizations"}
        assert set(GOOGLE_SOURCES.keys()) == esperados


class TestValidacionArchivos:
    """Pruebas de validación de archivos."""

    def test_archivo_inexistente_no_es_valido(self):
        """Un archivo que no existe no debe ser válido."""
        ruta = Path(tempfile.gettempdir()) / "archivo_inexistente_test.csv"
        if ruta.exists():
            ruta.unlink()
        assert _is_file_valid(ruta) is False

    def test_archivo_pequeno_no_es_valido(self):
        """Un archivo muy pequeño (< 1KB) no debe ser válido."""
        ruta = Path(tempfile.gettempdir()) / "archivo_pequeno_test.csv"
        with open(ruta, "w") as f:
            f.write("x")  # Solo 1 byte
        assert _is_file_valid(ruta) is False
        ruta.unlink()

    def test_archivo_valido_si_suficientemente_grande(self):
        """Un archivo >= 1KB debe ser válido."""
        ruta = Path(tempfile.gettempdir()) / "archivo_valido_test.csv"
        with open(ruta, "w") as f:
            f.write("a" * 2048)  # 2KB
        assert _is_file_valid(ruta) is True
        ruta.unlink()


class TestDescarga:
    """Pruebas del proceso de descarga."""

    def test_ensure_dirs_crea_directorios(self):
        """Verifica que _ensure_dirs crea los directorios necesarios."""
        _ensure_dirs()
        assert RAW_GOOGLE_DIR.exists(), "Directorio raw/google no existe"
