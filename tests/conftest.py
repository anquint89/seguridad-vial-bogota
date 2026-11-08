"""Datos de juguete compartidos por las pruebas de los modelos."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel_minimo() -> pd.DataFrame:
    """Dos CIV con los siete anios, con la forma del panel real."""
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


@pytest.fixture
def aristas_minimas() -> pd.DataFrame:
    """Los dos CIV del panel de juguete son vecinos entre si."""
    return pd.DataFrame({"CIV": [1, 2], "vecino": [2, 1]})
