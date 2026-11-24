"""Los dos modelos elegidos, servidos desde el paquete.

El de conteo responde cuantos siniestros graves se esperan al anio en el
segmento, y el de clasificacion que tan probable es que ocurra al menos uno.
Los dos corren sobre la misma matriz de caracteristicas y la misma particion,
que es lo que los hace comparables.

El factor de Empirical Bayes viaja en los datos de entrada, no en el modelo:
es un parametro ajustado por segmento. Si no viene, esperado_eb sale igual a
esperado.
"""

import numpy as np
import pandas as pd

from modelo_riesgo import __version__ as _version
from modelo_riesgo.config.core import config
from modelo_riesgo.processing.data_manager import load_model
from modelo_riesgo.processing.validation import validate_inputs

FACTOR_EB = "factor_eb"

_modelos = {tarea: load_model(file_name=d["archivo"]) for tarea, d in config["modelos"].items()}


def modelo(tarea: str):
    """El Booster de LightGBM de una de las dos tareas, ya cargado."""
    return _modelos[tarea]


def make_prediction(*, input_data) -> dict:
    """Riesgo esperado y probabilidad de siniestro grave por fila.

    Devuelve un dict con `predictions`, que es una lista de registros con
    esperado, esperado_eb y prob_grave, mas la version del paquete y los
    errores de validacion si los hubo.
    """
    data = pd.DataFrame(input_data)
    validados, errores = validate_inputs(input_data=data)
    if errores is not None:
        return {"predictions": None, "version": _version, "errors": errores}

    X = np.ascontiguousarray(validados[config["features"]].to_numpy(np.float64))
    # El objetivo poisson ya devuelve la exponencial: el desplazamiento se
    # multiplica aparte, igual que al entrenar.
    esperado = _modelos["conteo"].predict(X) * np.exp(validados[config["offset"]].to_numpy())
    prob = _modelos["clasificacion"].predict(X)

    if FACTOR_EB in data.columns:
        factor = pd.to_numeric(data[FACTOR_EB], errors="coerce").fillna(1.0).to_numpy()
    else:
        factor = np.ones(len(esperado))

    salida = pd.DataFrame(
        {"esperado": esperado, "esperado_eb": esperado * factor, "prob_grave": prob}
    )
    if "CIV" in data.columns:
        salida.insert(0, "CIV", data["CIV"].to_numpy())

    return {
        "predictions": salida.to_dict(orient="records"),
        "version": _version,
        "errors": None,
    }
