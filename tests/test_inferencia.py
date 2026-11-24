"""Pruebas de la inferencia en vivo de la API."""

import numpy as np
import pandas as pd
import pytest

from src.api import inferencia

pytestmark = pytest.mark.skipif(
    not inferencia.disponible(),
    reason="faltan los artefactos del modelo, corre python -m src.modelos.empaquetar",
)


def test_la_api_declara_que_infiere_y_con_que_modelo():
    f = inferencia.ficha()
    assert f["inferencia"] == "en vivo"
    assert set(f["modelos"]) == {"conteo", "clasificacion"}
    for corrida in f["modelos"].values():
        assert corrida  # el vinculo con la corrida de MLflow no se pierde


def test_predecir_devuelve_el_contrato_que_espera_el_tablero():
    d = inferencia.predecir(2021)
    assert list(d.columns) == ["CIV", "esperado", "esperado_eb", "prob_grave"]
    assert len(d) > 0
    assert d.CIV.is_unique


def test_las_predicciones_son_numeros_validos():
    d = inferencia.predecir(2021)
    assert (d.esperado >= 0).all()
    assert (d.esperado_eb >= 0).all()
    assert d.prob_grave.between(0, 1).all()
    assert np.isfinite(d[["esperado", "esperado_eb", "prob_grave"]].to_numpy()).all()


def test_pedir_unos_pocos_segmentos_da_lo_mismo_que_pedirlos_todos():
    """La inferencia no depende de cuantos segmentos se pidan a la vez."""
    todos = inferencia.predecir(2021).set_index("CIV")
    unos = inferencia.predecir(2021, todos.index[:20].tolist()).set_index("CIV")
    assert len(unos) == 20
    pd.testing.assert_frame_equal(unos, todos.loc[unos.index], check_exact=False)


def test_los_dos_anios_del_tablero_tienen_prediccion():
    assert inferencia.anios() == [2021, 2022]
    for anio in inferencia.anios():
        assert not inferencia.predecir(anio).empty


def test_un_anio_sin_caracteristicas_no_revienta():
    assert inferencia.predecir(1999).empty


def test_un_civ_inexistente_no_revienta():
    assert inferencia.predecir(2021, [-1]).empty


def test_el_empirical_bayes_mueve_el_esperado():
    """El factor es un parametro del modelo, no un uno constante: si no
    moviera nada, el posproceso no estaria aplicandose."""
    d = inferencia.predecir(2021)
    assert not np.allclose(d.esperado, d.esperado_eb)
