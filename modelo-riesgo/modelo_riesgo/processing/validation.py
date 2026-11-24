import json

import pandas as pd

from modelo_riesgo.config.core import config


def columnas_requeridas() -> list[str]:
    """Las variables del modelo mas el desplazamiento, que no es una variable."""
    return [*config["features"], config["offset"]]


def validate_inputs(*, input_data: pd.DataFrame):
    """Devuelve (datos, errores). errores es None si todo esta bien, o un JSON
    con el detalle de lo que falta o no es numerico.

    Las columnas sobrantes se ignoran y el orden no importa: se reindexa por
    el nombre, que es el mismo con el que se entreno.
    """
    columnas = columnas_requeridas()

    faltan = [c for c in columnas if c not in input_data.columns]
    if faltan:
        return input_data, json.dumps({"faltan_columnas": faltan})

    datos = input_data[columnas].apply(pd.to_numeric, errors="coerce")
    malas = [c for c in columnas if datos[c].isna().any()]
    if malas:
        return input_data, json.dumps({"valores_vacios_o_no_numericos": malas})

    return datos, None
