"""Pruebas de la rejilla de experimentos."""

import pandas as pd

from src.modelos.experimentos import (
    CARACTERES_PROHIBIDOS,
    PERCENTILES,
    captura_por_corte,
    nombre_de_modelo,
)


def test_el_nombre_del_modelo_pierde_los_caracteres_que_mlflow_rechaza():
    """Un nombre de corrida con punto, como lr0.05, hacia fallar el registro
    del modelo sin tumbar la corrida: el modelo se perdia en silencio."""
    assert nombre_de_modelo("lightgbm_poisson_h31_lr0.05") == "lightgbm_poisson_h31_lr0_05"
    sucio = "a/b:c.d%e\"f'g"
    limpio = nombre_de_modelo(sucio)
    assert not any(malo in limpio for malo in CARACTERES_PROHIBIDOS)


def test_un_nombre_limpio_no_se_toca():
    assert nombre_de_modelo("regresion_logistica") == "regresion_logistica"


def test_la_captura_cubre_los_cuatro_cortes():
    ranking = pd.Series(range(100, 0, -1), index=range(1, 101)).astype(float)
    graves = pd.Series([1] * 100, index=range(1, 101))
    metricas = captura_por_corte(ranking, graves, universo=100)
    assert set(metricas) == {f"captura_top_{int(q * 100)}_pct" for q in PERCENTILES}
    # Ranking perfecto sobre graves uniformes: el top 10 % captura el 10 %.
    assert metricas["captura_top_10_pct"] == 10.0


def test_la_captura_premia_el_orden_correcto():
    """Dos rankings opuestos sobre los mismos graves: el bueno captura mas."""
    graves = pd.Series([0] * 90 + [1] * 10, index=range(1, 101))
    bueno = pd.Series(range(1, 101), index=range(1, 101)).astype(float)
    malo = pd.Series(range(100, 0, -1), index=range(1, 101)).astype(float)
    assert (
        captura_por_corte(bueno, graves, 100)["captura_top_10_pct"]
        > captura_por_corte(malo, graves, 100)["captura_top_10_pct"]
    )
