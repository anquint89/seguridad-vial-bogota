"""Pruebas del grafo de vecindad."""

import json

import pandas as pd

from src.modelos.vecindad import aristas, nodos_por_civ


def _linea(civ: int, inicio: tuple[float, float], fin: tuple[float, float]) -> dict:
    return {"MVICIV": civ, "geometry": json.dumps({"paths": [[list(inicio), list(fin)]]})}


def test_dos_segmentos_que_comparten_extremo_son_vecinos():
    malla = pd.DataFrame(
        [_linea(10, (0, 0), (100, 0)), _linea(20, (100, 0), (100, 80))]
    )
    tabla = aristas(nodos_por_civ(malla))
    assert set(map(tuple, tabla.to_numpy())) == {(10, 20), (20, 10)}


def test_un_segmento_suelto_no_genera_aristas():
    malla = pd.DataFrame(
        [_linea(10, (0, 0), (100, 0)), _linea(20, (900, 900), (950, 900))]
    )
    assert aristas(nodos_por_civ(malla)).empty


def test_ningun_civ_es_vecino_de_si_mismo():
    """Dos registros del mismo CIV que se tocan no generan un lazo."""
    malla = pd.DataFrame(
        [_linea(10, (0, 0), (100, 0)), _linea(10, (100, 0), (200, 0))]
    )
    assert aristas(nodos_por_civ(malla)).empty


def test_el_grafo_es_simetrico():
    malla = pd.DataFrame(
        [
            _linea(10, (0, 0), (100, 0)),
            _linea(20, (100, 0), (100, 80)),
            _linea(30, (100, 0), (200, 0)),
        ]
    )
    tabla = aristas(nodos_por_civ(malla))
    pares = set(map(tuple, tabla.to_numpy()))
    assert all((b, a) in pares for a, b in pares)
    assert len(pares) == 6  # tres CIV en un nodo: 3 x 2 direcciones


def test_el_civ_cero_se_ignora():
    """CIV 0 significa sin llave, no un segmento real."""
    malla = pd.DataFrame(
        [_linea(0, (0, 0), (100, 0)), _linea(20, (100, 0), (100, 80))]
    )
    assert aristas(nodos_por_civ(malla)).empty


def test_el_nodo_se_redondea_al_metro():
    """Coordenadas que difieren en centimetros son el mismo nodo: hace de
    tolerancia de union entre trazos que la fuente no cierra exacto."""
    malla = pd.DataFrame(
        [_linea(10, (0, 0), (100.2, 0.1)), _linea(20, (99.9, -0.2), (100, 80))]
    )
    assert not aristas(nodos_por_civ(malla)).empty
