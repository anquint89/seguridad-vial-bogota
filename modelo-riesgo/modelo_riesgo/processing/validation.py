import json

import pandas as pd

from modelo_riesgo.config.core import config


def validate_inputs(*, input_data: pd.DataFrame):
    """Devuelve (datos, errores). errores es None si todo está bien, o un JSON con el detalle."""
    columnas = config["features"]

    faltan = [c for c in columnas if c not in input_data.columns]
    if faltan:
        return input_data, json.dumps({"faltan_columnas": faltan})

    datos = input_data[columnas].apply(pd.to_numeric, errors="coerce")
    malas = [c for c in columnas if datos[c].isna().any()]
    if malas:
        return input_data, json.dumps({"valores_vacios_o_no_numericos": malas})

    return datos, None