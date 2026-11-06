"""Pruebas del modelo de conteo: dispersion, Empirical Bayes y matriz de diseno."""

import numpy as np
import pandas as pd
import pytest

from src.modelos.conteo import empirical_bayes, estimar_alpha, matriz_diseno, preparar


def test_alpha_es_cero_cuando_los_datos_son_poisson():
    """Sin sobredispersion, la regresion auxiliar devuelve alpha cercano a cero."""
    rng = np.random.default_rng(0)
    mu = np.full(200_000, 0.5)
    y = rng.poisson(mu).astype(float)
    assert abs(estimar_alpha(y, mu)) < 0.02


def test_alpha_recupera_la_dispersion_conocida():
    """Con una mezcla gamma-Poisson de alpha 0,5 se recupera ese valor."""
    rng = np.random.default_rng(1)
    alpha = 0.5
    mu = np.full(200_000, 1.0)
    lam = rng.gamma(shape=mu / alpha, scale=alpha)
    y = rng.poisson(lam).astype(float)
    assert estimar_alpha(y, mu) == pytest.approx(alpha, abs=0.05)


def test_empirical_bayes_queda_entre_lo_esperado_y_lo_observado():
    mu = pd.Series([1.0, 1.0, 1.0])
    obs = pd.Series([0.0, 1.0, 5.0])
    eb = empirical_bayes(mu, obs, alpha=0.5)
    assert (eb >= np.minimum(mu, obs)).all()
    assert (eb <= np.maximum(mu, obs)).all()
    assert eb[1] == pytest.approx(1.0)  # sin discrepancia no hay correccion


def test_empirical_bayes_encoge_mas_cuando_hay_menos_historia():
    """Con mu pequeno el peso del modelo es mayor: la observacion se encoge mas."""
    obs = pd.Series([10.0, 10.0])
    poca = empirical_bayes(pd.Series([0.1]), pd.Series([10.0]), alpha=0.5)[0]
    mucha = empirical_bayes(pd.Series([8.0]), pd.Series([10.0]), alpha=0.5)[0]
    assert poca < mucha < obs[0]


def _panel_minimo() -> pd.DataFrame:
    civ = np.repeat([1, 2], 7)
    anio = np.tile(range(2015, 2022), 2)
    n = len(civ)
    return pd.DataFrame(
        {
            "CIV": civ,
            "anio": anio,
            "graves": np.arange(n) % 3,
            "graves_acum_previo": np.arange(n),
            "graves_rezago1": np.r_[np.nan, np.arange(n - 1)],
            "long_m": np.full(n, 100.0),
            "exposicion_m": np.full(n, 200.0),
            "log_long_km": np.full(n, np.log(0.1)),
            "carriles": np.full(n, 2),
            "clase_via": np.tile([1, 3], n // 2 + 1)[:n],
            "vel_reg": np.full(n, 30.0),
            "vel_faltante": np.zeros(n, dtype="int8"),
            "tratado_senalizacion": np.zeros(n, dtype="int8"),
            "post_senalizacion": np.zeros(n, dtype="int8"),
        }
    )


def test_la_matriz_no_repite_el_termino_lineal_del_anio():
    """n_prev y tendencia son ambas lineales en el anio: solo puede entrar una,
    o la matriz queda deficiente de rango contra el intercepto."""
    X = matriz_diseno(preparar(_panel_minimo()))
    assert "tendencia" in X.columns
    assert "n_prev" not in X.columns


def test_la_matriz_deja_fuera_la_clase_de_referencia():
    X = matriz_diseno(preparar(_panel_minimo()))
    assert "clase_1" in X.columns
    assert "clase_3" not in X.columns
    assert "const" in X.columns


def test_la_historia_tiene_ventana_comparable():
    """hist_media es la media por anio previo, no el acumulado crudo: en 2021
    un CIV con un grave por anio desde 2015 tiene media 1, no 6."""
    p = preparar(_panel_minimo())
    fila = p[(p.CIV == 1) & (p.anio == 2021)].iloc[0]
    assert fila.n_prev == 6
    assert fila.hist_media == pytest.approx(fila.graves_acum_previo / 6)
