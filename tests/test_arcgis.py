"""Pruebas del cliente REST de ArcGIS y de la descarga de capas.

Ninguna prueba toca la red: las respuestas del servidor se simulan
reemplazando urllib.request.urlopen o las funciones del cliente.

    python -m pytest tests/test_arcgis.py -v
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from src.extraccion import arcgis


# ----------------------------------------------------------------- auxiliares


class RespuestaFalsa(io.BytesIO):
    """Imita lo que devuelve urlopen: un archivo de bytes usable con `with`."""

    def __init__(self, cuerpo: dict):
        super().__init__(json.dumps(cuerpo).encode("utf-8"))


@pytest.fixture
def sin_espera(monkeypatch):
    """Anula time.sleep para que los reintentos no hagan esperar la prueba."""
    esperas: list[float] = []
    monkeypatch.setattr(arcgis.time, "sleep", esperas.append)
    return esperas


# ------------------------------------------------------------------ _pedir


def test_pedir_agrega_formato_json_a_la_consulta(monkeypatch):
    urls: list[str] = []

    def urlopen(url, timeout):
        urls.append(url)
        return RespuestaFalsa({"count": 5})

    monkeypatch.setattr(arcgis.urllib.request, "urlopen", urlopen)
    assert arcgis._pedir("https://servidor/consulta", {"where": "1=1"}) == {"count": 5}
    assert "f=json" in urls[0]
    assert "where=1%3D1" in urls[0]


def test_pedir_reintenta_tras_un_fallo_de_red(monkeypatch, sin_espera):
    respuestas = [urllib.error.URLError("caido"), RespuestaFalsa({"count": 7})]

    def urlopen(url, timeout):
        siguiente = respuestas.pop(0)
        if isinstance(siguiente, Exception):
            raise siguiente
        return siguiente

    monkeypatch.setattr(arcgis.urllib.request, "urlopen", urlopen)
    assert arcgis._pedir("https://servidor/consulta", {}) == {"count": 7}
    assert sin_espera == [arcgis.ESPERA_BASE]


def test_pedir_trata_el_error_del_servicio_como_fallo(monkeypatch, sin_espera):
    respuestas = [RespuestaFalsa({"error": {"code": 500}}), RespuestaFalsa({"count": 1})]
    monkeypatch.setattr(
        arcgis.urllib.request, "urlopen", lambda url, timeout: respuestas.pop(0)
    )
    assert arcgis._pedir("https://servidor/consulta", {}) == {"count": 1}
    assert len(sin_espera) == 1


def test_pedir_se_rinde_despues_de_todos_los_reintentos(monkeypatch, sin_espera):
    def urlopen(url, timeout):
        raise urllib.error.URLError("caido")

    monkeypatch.setattr(arcgis.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="No se pudo consultar"):
        arcgis._pedir("https://servidor/consulta", {})
    # La espera se duplica en cada intento: 3, 6, 12, 24 segundos.
    assert sin_espera == [arcgis.ESPERA_BASE * 2**i for i in range(arcgis.REINTENTOS)]
