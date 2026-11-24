"""La metrica de negocio y el desempate, que es donde se escondia el problema.

La linea base del metodo vigente son conteos enteros, y en el corte del 5 %
empatan 1.881 segmentos. Ordenar y cortar da un numero que depende del orden
del algoritmo de ordenamiento. Aqui se fija el comportamiento correcto.
"""

import numpy as np
import pytest

from src.modelos.metricas import captura_top, captura_top_alineada, empates_en_el_corte
import pandas as pd


def test_sin_empates_es_la_captura_de_toda_la_vida():
    puntaje = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    graves = np.array([10.0, 0.0, 0.0, 0.0, 10.0])
    # Los dos primeros capturan 10 de 20.
    assert captura_top(puntaje, graves, 2) == pytest.approx(50.0)


def test_el_grupo_empatado_entra_por_la_parte_proporcional():
    # Cuatro empatados en 1.0 con 8 graves entre ellos, y caben dos.
    puntaje = np.array([9.0, 1.0, 1.0, 1.0, 1.0])
    graves = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    # El de 9.0 entra entero (2 graves) y de los empatados caben 2 de 4,
    # o sea la mitad de sus 8 graves.
    assert captura_top(puntaje, graves, 3) == pytest.approx(100 * (2 + 4) / 10)


def test_el_resultado_no_depende_del_orden_de_entrada():
    rng = np.random.default_rng(3)
    puntaje = np.repeat([3.0, 2.0, 1.0], 40)
    graves = rng.integers(0, 4, size=120).astype(float)
    base = captura_top(puntaje, graves, 50)
    for _ in range(10):
        idx = rng.permutation(len(puntaje))
        assert captura_top(puntaje[idx], graves[idx], 50) == pytest.approx(base)


def test_coincide_con_la_media_del_desempate_aleatorio():
    rng = np.random.default_rng(11)
    puntaje = np.repeat([4.0, 2.0, 2.0, 1.0], 25)
    graves = rng.integers(0, 5, size=100).astype(float)
    k = 60
    exacta = captura_top(puntaje, graves, k)

    muestras = []
    for _ in range(400):
        orden = np.lexsort((rng.random(len(puntaje)), -puntaje))
        muestras.append(100 * graves[orden[:k]].sum() / graves.sum())
    assert exacta == pytest.approx(float(np.mean(muestras)), abs=0.5)


def test_declara_cuantos_empatan_en_el_corte():
    puntaje = np.array([9.0, 1.0, 1.0, 1.0, 1.0])
    d = empates_en_el_corte(puntaje, 3)
    assert d == {"valor_de_corte": 1.0, "empatados": 4, "cupo": 2}


def test_alineada_usa_el_indice_y_no_la_posicion():
    graves = pd.Series([5.0, 0.0, 5.0], index=[30, 10, 20])
    puntaje = pd.Series([1.0, 9.0, 0.0], index=[10, 20, 30])
    # Por indice, el mejor puntaje es el CIV 20, que tiene 5 graves de 10.
    assert captura_top_alineada(puntaje, graves, 1) == pytest.approx(50.0)


def test_un_civ_sin_puntaje_cuenta_como_cero():
    graves = pd.Series([1.0, 1.0], index=[1, 2])
    puntaje = pd.Series([7.0], index=[1])
    assert captura_top_alineada(puntaje, graves, 1) == pytest.approx(50.0)
