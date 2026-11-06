"""Comprueba que los JSON de los modelos digan lo mismo que las predicciones.

Las cifras del reporte salen de reportes/modelos/*.json. Aqui se recalculan
desde data/procesado/*.parquet para que un cambio en los modelos no pase
inadvertido. Necesita los datos de DVC; si no estan, las pruebas se omiten.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
PRED_CONTEO = RAIZ / "data" / "procesado" / "prediccion_conteo.parquet"
PRED_CLASES = RAIZ / "data" / "procesado" / "prediccion_clasificador.parquet"
JSON_CONTEO = RAIZ / "reportes" / "modelos" / "conteo.json"
JSON_CLASES = RAIZ / "reportes" / "modelos" / "clasificador.json"

CORTES = (1, 2, 5, 10)

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (PRED_CONTEO, PRED_CLASES, JSON_CONTEO, JSON_CLASES)),
    reason="faltan las predicciones o los JSON; corre el pipeline o dvc pull",
)


@pytest.fixture(scope="module")
def conteo():
    return pd.read_parquet(PRED_CONTEO)


@pytest.fixture(scope="module")
def clases():
    return pd.read_parquet(PRED_CLASES)


@pytest.fixture(scope="module")
def jconteo():
    return json.loads(JSON_CONTEO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def jclases():
    return json.loads(JSON_CLASES.read_text(encoding="utf-8"))


def capturados(df, score, corte):
    """Graves de 2021 en el primer `corte` % del ranking ordenado por `score`."""
    k = int(round(corte / 100 * len(df)))
    orden = df.sort_values(score, ascending=False, kind="stable")
    return int(orden["graves_2021"].head(k).sum())


def test_los_dos_modelos_puntuan_el_mismo_universo(conteo, clases, jconteo):
    assert len(conteo) == len(clases)
    assert len(conteo) == jconteo["ventana"]["universo_civ"]
    assert conteo["CIV"].nunique() == len(conteo)


def test_la_ventana_cuadra_con_el_universo(conteo, jconteo):
    v = jconteo["ventana"]
    assert v["entrena"] == [2016, 2020]
    assert v["valida"] == 2021
    assert v["filas_valida"] == len(conteo)
    assert v["filas_entrena"] == 5 * len(conteo)


def test_el_denominador_son_los_graves_de_2021(conteo, jconteo, jclases):
    graves = int(conteo["graves_2021"].sum())
    for seccion in (jconteo["evaluacion_2021"], jclases["desempeno_2021"]):
        for modelo in seccion.values():
            for corte in CORTES:
                assert modelo[f"top_{corte}_pct"]["de"] == graves
                assert modelo[f"top_{corte}_pct"]["segmentos"] == int(round(corte / 100 * len(conteo)))


def test_captura_de_los_modelos_de_conteo(conteo, jconteo):
    for columna, modelo in (("mu_2021", "binomial_negativa"), ("eb_2021", "binomial_negativa_eb")):
        for corte in CORTES:
            publicado = jconteo["evaluacion_2021"][modelo][f"top_{corte}_pct"]["capturados"]
            assert capturados(conteo, columna, corte) == publicado


def test_captura_de_los_clasificadores(clases, jclases):
    for columna, modelo in (("prob_logit", "regresion_logistica"), ("prob_lgb", "lightgbm")):
        for corte in CORTES:
            publicado = jclases["desempeno_2021"][modelo][f"top_{corte}_pct"]["capturados"]
            assert capturados(clases, columna, corte) == publicado


def test_captura_de_la_linea_base(conteo, jconteo):
    """El conteo historico empata en el corte, asi que se compara el porcentaje."""
    graves = int(conteo["graves_2021"].sum())
    for corte in CORTES:
        publicado = jconteo["evaluacion_2021"]["linea_base_conteo_historico"][f"top_{corte}_pct"]
        pct = 100 * capturados(conteo, "conteo_2015_2020", corte) / graves
        assert pct == pytest.approx(publicado["pct"], abs=1.0)


def test_discriminacion_de_los_clasificadores(clases, jclases):
    y = clases["ocurrencia_2021"]
    for columna, modelo in (("prob_logit", "regresion_logistica"), ("prob_lgb", "lightgbm")):
        publicado = jclases["desempeno_2021"][modelo]
        assert round(roc_auc_score(y, clases[columna]), 4) == publicado["auc_roc"]
        assert round(average_precision_score(y, clases[columna]), 4) == publicado["auc_pr"]
        assert round(brier_score_loss(y, clases[columna]), 5) == publicado["brier"]


def test_los_modelos_discriminan_mejor_que_la_linea_base(clases, conteo, jclases):
    base = conteo.set_index("CIV").loc[clases["CIV"], "conteo_2015_2020"]
    publicado = jclases["desempeno_2021"]["linea_base_conteo_historico"]
    assert round(roc_auc_score(clases["ocurrencia_2021"], base), 4) == publicado["auc_roc"]
    assert publicado["auc_roc"] < jclases["desempeno_2021"]["lightgbm"]["auc_roc"]


def test_los_positivos_del_clasificador(clases, jclases):
    r = jclases["respuesta"]
    assert int(clases["ocurrencia_2021"].sum()) == r["positivos_valida"]
    assert (clases["ocurrencia_2021"] == (clases["graves_2021"] > 0)).all()


def test_calibracion_del_conteo(conteo, jconteo):
    for columna, modelo in (("mu_2021", "binomial_negativa"), ("eb_2021", "binomial_negativa_eb")):
        publicado = jconteo["calibracion_2021"][modelo]
        suma = float(conteo[columna].sum())
        assert round(suma, 1) == publicado["suma_predicha"]
        assert round(float(conteo[columna].max()), 2) == publicado["maximo_predicho"]
        assert round(suma / publicado["suma_observada"], 3) == publicado["razon_suma"]


def test_el_conteo_sigue_sobrestimando_un_14_por_ciento(jconteo):
    """Si esto cambia, el parrafo del reporte que lo declara hay que reescribirlo."""
    assert 1.10 < jconteo["calibracion_2021"]["binomial_negativa"]["razon_suma"] < 1.20


def test_la_binomial_negativa_mejora_el_aic_sobre_poisson(jconteo):
    assert jconteo["ajuste"]["binomial_negativa"]["aic"] < jconteo["ajuste"]["poisson"]["aic"]


def test_la_elasticidad_de_la_longitud_sigue_liberada(jconteo):
    """Forzarla a 1 era el error que inflaba la prediccion a 17.413 graves."""
    assert 0.2 < jconteo["elasticidad_longitud"] < 0.4
