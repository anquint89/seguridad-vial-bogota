"""Pruebas del modelo de conteo: dispersion y Empirical Bayes.

Lo relativo a la matriz de diseno vive en test_caracteristicas.py, porque esa
capa la comparten los dos modelos.
"""

import numpy as np
import pandas as pd
import pytest

from src.modelos.conteo import empirical_bayes, estimar_alpha


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
    poca = empirical_bayes(pd.Series([0.1]), pd.Series([10.0]), alpha=0.5)[0]
    mucha = empirical_bayes(pd.Series([8.0]), pd.Series([10.0]), alpha=0.5)[0]
    assert poca < mucha < 10.0
