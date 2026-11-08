"""Pruebas de la capa de caracteristicas compartida."""

import numpy as np
import pytest

from src.modelos.caracteristicas import matriz_diseno, particion, preparar


def test_la_historia_tiene_ventana_comparable(panel_minimo):
    """hist_media es la media por anio previo, no el acumulado crudo: en 2021
    un CIV con seis anios de historia divide entre seis, no suma."""
    p = preparar(panel_minimo)
    fila = p[(p.CIV == 1) & (p.anio == 2021)].iloc[0]
    assert fila.n_prev == 6
    assert fila.hist_media == pytest.approx(fila.graves_acum_previo / 6)


def test_el_rezago_del_primer_anio_queda_en_cero(panel_minimo):
    """2015 no tiene anio anterior. Se rellena con cero y la particion lo deja fuera."""
    p = preparar(panel_minimo)
    assert p[(p.CIV == 1) & (p.anio == 2015)].rezago1.iloc[0] == 0.0
    entrena, _ = particion(p)
    assert not p[entrena].anio.eq(2015).any()


def test_la_matriz_no_repite_el_termino_lineal_del_anio(panel_minimo):
    """n_prev y tendencia son ambas lineales en el anio: solo puede entrar una,
    o la matriz queda deficiente de rango contra el intercepto."""
    X = matriz_diseno(preparar(panel_minimo))
    assert "tendencia" in X.columns
    assert "n_prev" not in X.columns

    # Las columnas constantes del panel de juguete (obras y vecindad en cero)
    # bajan el rango por si solas y no son el problema que se vigila aqui.
    variables = X.loc[:, X.nunique() > 1]
    assert np.linalg.matrix_rank(variables.to_numpy(float)) == variables.shape[1]


def test_la_matriz_deja_fuera_la_clase_de_referencia(panel_minimo):
    X = matriz_diseno(preparar(panel_minimo))
    assert "clase_1" in X.columns
    assert "clase_3" not in X.columns
    assert "const" in X.columns


def test_la_particion_no_solapa_ni_toca_el_anio_de_validacion(panel_minimo):
    p = preparar(panel_minimo)
    entrena, valida = particion(p)
    assert not (entrena & valida).any()
    assert p[entrena].anio.max() == 2020
    assert set(p[valida].anio) == {2021}


def test_sin_grafo_las_variables_de_vecindad_quedan_en_cero(panel_minimo):
    p = preparar(panel_minimo, aristas=None)
    assert (p.vecinos_hist == 0).all()
    assert (p.vecinos_n == 0).all()
    assert "vecinos_hist_log" in matriz_diseno(p).columns


def test_la_vecindad_promedia_la_historia_del_vecino(panel_minimo, aristas_minimas):
    """Con dos CIV vecinos entre si, la historia de vecinos de uno es la
    historia propia del otro."""
    p = preparar(panel_minimo, aristas_minimas)
    for anio in range(2015, 2022):
        uno = p[(p.CIV == 1) & (p.anio == anio)].iloc[0]
        dos = p[(p.CIV == 2) & (p.anio == anio)].iloc[0]
        assert uno.vecinos_hist == pytest.approx(dos.hist_media)
        assert dos.vecinos_hist == pytest.approx(uno.hist_media)
    assert (p.vecinos_n == 1).all()


def test_la_vecindad_no_mira_el_futuro(panel_minimo, aristas_minimas):
    """La historia del vecino en el anio t solo usa anios anteriores a t, asi
    que cambiar los graves de 2021 no puede mover ninguna variable de 2021."""
    p = preparar(panel_minimo, aristas_minimas)
    alterado = panel_minimo.copy()
    ultimo = (alterado.anio == 2021)
    alterado.loc[ultimo, "graves"] = 99
    q = preparar(alterado, aristas_minimas)
    columnas = ["hist_media", "rezago1", "vecinos_hist", "vecinos_rezago1"]
    assert np.allclose(
        p.loc[p.anio == 2021, columnas].to_numpy(float),
        q.loc[q.anio == 2021, columnas].to_numpy(float),
    )
