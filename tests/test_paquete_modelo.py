"""El paquete instalable modelo-riesgo, que es lo que corre dentro de la API.

Se omiten si el paquete no esta instalado, porque el repositorio se puede usar
sin el para trabajar en el pipeline.

    pip install model-pkg/modelo_riesgo-*.whl
"""

import json
from pathlib import Path

import pytest

modelo_riesgo = pytest.importorskip("modelo_riesgo", reason="falta instalar la rueda de model-pkg")

import pandas as pd  # noqa: E402

from modelo_riesgo.config.core import DATASETS, config  # noqa: E402
from modelo_riesgo.predict import make_prediction  # noqa: E402
from modelo_riesgo.processing.validation import columnas_requeridas  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SELECCION = RAIZ / "reportes" / "modelos" / "seleccion.json"


@pytest.fixture(scope="module")
def muestra() -> pd.DataFrame:
    return pd.read_csv(DATASETS / config["test_data_file"])


def test_el_paquete_trae_las_dos_tareas():
    assert set(config["modelos"]) == {"conteo", "clasificacion"}


def test_sirve_las_corridas_que_elige_la_seleccion():
    """Lo que la API ejecuta tiene que ser lo que el reporte dice que se eligio."""
    if not SELECCION.exists():
        pytest.skip("no esta reportes/modelos/seleccion.json")
    eleccion = json.loads(SELECCION.read_text(encoding="utf-8"))
    for tarea in ("conteo", "clasificacion"):
        assert config["modelos"][tarea]["corrida"] == eleccion[tarea]["seleccion"]["elegida"]


def test_predice_las_tres_columnas(muestra):
    r = make_prediction(input_data=muestra)
    assert r["errors"] is None
    salida = pd.DataFrame(r["predictions"])
    assert {"esperado", "esperado_eb", "prob_grave"} <= set(salida.columns)
    assert len(salida) == len(muestra)


def test_el_esperado_es_positivo_y_la_probabilidad_cabe_en_cero_uno(muestra):
    salida = pd.DataFrame(make_prediction(input_data=muestra)["predictions"])
    assert (salida.esperado > 0).all()
    assert salida.prob_grave.between(0, 1).all()


def test_el_orden_de_las_columnas_no_cambia_el_resultado(muestra):
    normal = make_prediction(input_data=muestra)["predictions"]
    volteada = make_prediction(input_data=muestra[list(reversed(muestra.columns))])["predictions"]
    assert normal == volteada


def test_avisa_de_la_columna_que_falta(muestra):
    falta = columnas_requeridas()[3]
    r = make_prediction(input_data=muestra.drop(columns=[falta]))
    assert r["predictions"] is None
    assert json.loads(r["errors"]) == {"faltan_columnas": [falta]}


def test_avisa_del_valor_que_no_es_numerico(muestra):
    mala = muestra.copy()
    columna = columnas_requeridas()[3]
    mala[columna] = mala[columna].astype(object)
    mala.loc[0, columna] = "no es un numero"
    r = make_prediction(input_data=mala)
    assert r["predictions"] is None
    assert json.loads(r["errors"]) == {"valores_vacios_o_no_numericos": [columna]}


def test_sin_factor_eb_el_esperado_corregido_es_el_esperado(muestra):
    salida = pd.DataFrame(
        make_prediction(input_data=muestra.drop(columns=["factor_eb"]))["predictions"]
    )
    assert salida.esperado.equals(salida.esperado_eb)


def test_el_desplazamiento_multiplica_la_prediccion(muestra):
    """Doblar la longitud del segmento tiene que doblar los graves esperados."""
    import numpy as np

    base = pd.DataFrame(make_prediction(input_data=muestra.head(20))["predictions"])
    doble = muestra.head(20).copy()
    doble[config["offset"]] = doble[config["offset"]] + np.log(2)
    doblada = pd.DataFrame(make_prediction(input_data=doble)["predictions"])
    assert np.allclose(doblada.esperado, 2 * base.esperado)
