"""Pruebas de la exposicion (siempre) y del panel (si ya existe panel.parquet).

Para las del panel, correr antes:  python -m src.preparacion.panel
Cifras de control: salen del perfilamiento.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.preparacion.exposicion import exposicion_ampliada, offset_log_longitud

RUTA = Path(__file__).resolve().parents[1] / "data" / "procesado" / "panel.parquet"
requiere_panel = pytest.mark.skipif(not RUTA.exists(), reason="Corre primero src.preparacion.panel")


def test_exposicion_pondera_por_carriles_y_clase():
    r = exposicion_ampliada(pd.Series([1000.0, 500.0, 200.0]), pd.Series([2, 1, 3]), pd.Series([1, 2, 9]))
    assert r.tolist() == [6000.0, 1000.0, 600.0]  # clase 9 sin peso -> 1


def test_offset_con_piso_para_longitud_cero():
    r = offset_log_longitud(pd.Series([0.0, 1000.0]))
    assert r.notna().all() and r.iloc[1] == pytest.approx(0.0)  # log(1 km) = 0


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(RUTA)


@requiere_panel
def test_rejilla_y_totales(panel):
    assert len(panel) == 116_322 * 7 and not panel.duplicated(["CIV", "anio"]).any()
    assert panel.graves.sum() == 71_268
    assert panel.loc[panel.anio == 2021, "graves"].sum() == 10_717


@requiere_panel
def test_historia_sin_fuga(panel):
    assert panel.loc[panel.anio == 2015, "graves_rezago1"].isna().all()
    assert (panel.graves_acum_previo + panel.graves == panel.groupby("CIV").graves.cumsum()).all()