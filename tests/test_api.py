"""Pruebas de la API. Requieren las tablas del tablero:

    python -m src.modelos.provisional
    python -m src.modelos.factores
    python -m src.tablero.datos
"""

from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "data" / "procesado" / "tablero_segmentos.parquet"
pytestmark = pytest.mark.skipif(not RUTA.exists(), reason="Corre primero src.tablero.datos")


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)


def test_salud(cliente):
    r = cliente.get("/salud").json()
    assert r["estado"] == "ok" and r["segmentos"] == 116_322 and 2021 in r["anios"]


def test_ranking_ordenado_y_del_tamano_pedido(cliente):
    filas = cliente.get("/ranking", params={"top": 50}).json()
    assert len(filas) == 50
    assert [f["rango"] for f in filas] == sorted(f["rango"] for f in filas)
    assert filas[0]["esperado_eb"] >= filas[-1]["esperado_eb"]


def test_ranking_filtra_por_localidad(cliente):
    filas = cliente.get("/ranking", params={"top": 20, "localidad": "Suba"}).json()
    assert filas and all(f["localidad"] == "SUBA" for f in filas)


def test_geojson_trae_geometria_en_bogota(cliente):
    fc = cliente.get("/segmentos/top", params={"n": 10}).json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 10
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert -74.3 < lon < -73.9 and 4.4 < lat < 4.9


def test_segmento_inexistente_da_404(cliente):
    assert cliente.get("/segmento/1").status_code == 404


def test_factores_trae_las_cinco_categorias(cliente):
    assert len(cliente.get("/factores").json()["factores"]) == 5
