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


def pagina(desde: int, cantidad: int, excede: bool) -> dict:
    """Bloque de rasgos como los que entrega el servicio en /query."""
    return {
        "features": [
            {
                "attributes": {"OBJECTID": desde + i, "CIV": 1000 + desde + i},
                "geometry": {"paths": [[[0, 0], [1, 1]]]},
            }
            for i in range(cantidad)
        ],
        "exceededTransferLimit": excede,
    }


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


# ------------------------------------------------------------------ contar


def test_contar_devuelve_un_entero(monkeypatch):
    llamadas: list[tuple] = []

    def pedir(url, parametros):
        llamadas.append((url, parametros))
        return {"count": "139107"}

    monkeypatch.setattr(arcgis, "_pedir", pedir)
    assert arcgis.contar("Malla_Vial_Integral") == 139107
    url, parametros = llamadas[0]
    assert url.endswith("/Malla_Vial_Integral/FeatureServer/0/query")
    assert parametros["returnCountOnly"] == "true"


# ----------------------------------------------------------------- paginar


def test_paginar_recorre_todas_las_paginas(monkeypatch):
    bloques = [
        pagina(0, arcgis.PAGINA, excede=True),
        pagina(arcgis.PAGINA, arcgis.PAGINA, excede=True),
        pagina(2 * arcgis.PAGINA, 150, excede=False),
    ]
    desplazamientos: list[int] = []

    def pedir(url, parametros):
        desplazamientos.append(parametros["resultOffset"])
        return bloques.pop(0)

    monkeypatch.setattr(arcgis, "_pedir", pedir)
    registros = list(arcgis.paginar("Servicio"))
    assert len(registros) == 2 * arcgis.PAGINA + 150
    assert desplazamientos == [0, arcgis.PAGINA, 2 * arcgis.PAGINA]
    assert [r["OBJECTID"] for r in registros] == list(range(2 * arcgis.PAGINA + 150))


def test_paginar_se_detiene_con_una_pagina_vacia(monkeypatch):
    bloques = [pagina(0, arcgis.PAGINA, excede=True), {"features": []}]
    monkeypatch.setattr(arcgis, "_pedir", lambda url, parametros: bloques.pop(0))
    assert len(list(arcgis.paginar("Servicio"))) == arcgis.PAGINA
    assert bloques == []


def test_paginar_sin_geometria_no_la_pide_ni_la_entrega(monkeypatch):
    pedidos: list[dict] = []

    def pedir(url, parametros):
        pedidos.append(parametros)
        return pagina(0, 3, excede=False)

    monkeypatch.setattr(arcgis, "_pedir", pedir)
    registros = list(arcgis.paginar("Servicio"))
    assert pedidos[0]["returnGeometry"] == "false"
    assert "outSR" not in pedidos[0]
    assert all("geometry" not in r for r in registros)


def test_paginar_con_geometria_la_pide_reproyectada(monkeypatch):
    pedidos: list[dict] = []

    def pedir(url, parametros):
        pedidos.append(parametros)
        return pagina(0, 3, excede=False)

    monkeypatch.setattr(arcgis, "_pedir", pedir)
    registros = list(arcgis.paginar("Servicio", con_geometria=True, srs_salida=3116))
    assert pedidos[0]["returnGeometry"] == "true"
    assert pedidos[0]["outSR"] == 3116
    assert all(r["geometry"] == {"paths": [[[0, 0], [1, 1]]]} for r in registros)
