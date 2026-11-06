"""Pruebas del clasificador de ocurrencia."""

import numpy as np
import pandas as pd
import pytest

from src.modelos import caracteristicas, clasificador


def test_la_respuesta_es_ocurrencia_no_conteo():
    """Un segmento con tres graves en el anio cuenta igual que uno con uno."""
    graves = pd.Series([0, 1, 3, 0, 7])
    y = (graves >= 1).astype(float)
    assert list(y) == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_los_dos_modelos_ven_la_misma_matriz():
    """El clasificador reutiliza la capa compartida a proposito: si cada modelo
    construyera la suya, la comparacion dejaria de ser de forma funcional."""
    assert clasificador.matriz_diseno is caracteristicas.matriz_diseno
    assert clasificador.particion is caracteristicas.particion


def test_el_brier_se_omite_cuando_el_puntaje_no_es_probabilidad():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    conteo = np.array([0.0, 5.0, 1.0, 9.0, 7.0, 0.0, 2.0, 4.0])
    ficha = clasificador._desempeno(y, conteo, es_probabilidad=False)
    assert "brier" not in ficha
    assert "auc_roc" in ficha and "auc_pr" in ficha


def test_el_brier_se_calcula_cuando_si_lo_es():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    prob = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.05, 0.3, 0.6])
    ficha = clasificador._desempeno(y, prob)
    # La ficha redondea a cinco decimales: 0,0553125 -> 0,05531.
    assert ficha["brier"] == round(float(np.mean((prob - y) ** 2)), 5)
    assert ficha["brier"] == 0.05531
    assert ficha["auc_roc"] == pytest.approx(1.0)


def test_la_ventana_coincide_con_la_del_conteo():
    """Las dos empiezan en 2016: desde 2017 post_ y tratado_ son identicas."""
    assert clasificador.ANIO_INICIO == caracteristicas.ANIO_INICIO == 2016
    assert clasificador.ANIO_FIN_ENTRENA == caracteristicas.ANIO_FIN_ENTRENA == 2020
    assert clasificador.ANIO_VALIDA == caracteristicas.ANIO_VALIDA == 2021
