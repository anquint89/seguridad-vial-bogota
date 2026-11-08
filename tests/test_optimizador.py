"""Pruebas del what-if y la mochila con datos simulados (no necesitan red ni datos)."""

import pandas as pd
import pytest

from src.tablero.optimizador import armar_paquete, opciones, resumen

FACTORES = [
    {"clave": "barata", "categoria": "Barata", "factor": -0.10, "ic90": [-0.2, 0.05],
     "costo_unitario_millones": 1.0, "unidad_costo": "segmento"},
    {"clave": "cara", "categoria": "Cara", "factor": -0.50, "ic90": [-0.6, -0.4],
     "costo_unitario_millones": 10.0, "unidad_costo": "segmento"},
    {"clave": "por_km", "categoria": "Por km", "factor": -0.20, "ic90": [-0.3, -0.1],
     "costo_unitario_millones": 4.0, "unidad_costo": "km"},
]
TODAS = {"barata", "cara"}


@pytest.fixture
def ranking():
    return pd.DataFrame({
        "CIV": [1, 2, 3],
        "esperado_eb": [10.0, 4.0, 1.0],
        "long_m": [1000.0, 500.0, 2000.0],
        "localidad": ["A", "B", "A"],
    })


def paquete(ranking, activas, presupuesto, costos=None):
    return armar_paquete(ranking, opciones(ranking, FACTORES, activas, costos or {}), presupuesto)


def test_con_poca_plata_va_la_opcion_barata(ranking):
    t = paquete(ranking, TODAS, 3.0)
    assert t.en_paquete.all() and (t.intervencion == "Barata").all()
    assert t.costo.sum() == pytest.approx(3.0)


def test_con_mas_plata_mejora_al_segmento_de_mayor_riesgo(ranking):
    # 1 + 1 + 1 = 3; mejorar el CIV 1 a "Cara" cuesta 9 mas y evita 4 mas
    t = paquete(ranking, TODAS, 12.0)
    assert t.intervencion.tolist() == ["Cara", "Barata", "Barata"]
    assert t[t.en_paquete].costo.sum() <= 12.0


def test_fuera_del_paquete_muestra_la_recomendada(ranking):
    t = paquete(ranking, TODAS, 1.0)
    assert t.en_paquete.tolist() == [True, False, False]
    assert t.intervencion.tolist() == ["Barata"] * 3


def test_apagar_una_categoria(ranking):
    t = paquete(ranking, {"cara"}, 100.0)
    assert (t.intervencion == "Cara").all() and t.reduccion.iloc[0] == pytest.approx(5.0)


def test_costo_editado_y_costo_por_km(ranking):
    t = paquete(ranking, {"por_km"}, 100.0, {"por_km": 2.0})
    assert t.costo.tolist() == pytest.approx([2.0, 1.0, 4.0])  # 2 M/km x 1; 0,5; 2 km


def test_sin_categorias_activas_no_hay_paquete(ranking):
    t = paquete(ranking, set(), 1e9)
    assert not t.en_paquete.any()


def test_resumen(ranking):
    r = resumen(paquete(ranking, {"cara"}, 20.0))
    assert r["segmentos"] == 2 and r["costo"] == pytest.approx(20.0)
    assert r["evitados"] == pytest.approx(7.0) and r["costo_por_evitado"] == pytest.approx(20 / 7)
    assert r["evitados_bajo"] <= r["evitados"] <= r["evitados_alto"]
