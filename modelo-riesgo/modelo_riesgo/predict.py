import numpy as np
import pandas as pd

from modelo_riesgo import __version__ as _version
from modelo_riesgo.config.core import config
from modelo_riesgo.processing.data_manager import load_model
from modelo_riesgo.processing.validation import validate_inputs

_modelo = load_model(file_name=f"{config['pipeline_save_file']}{_version}.txt")


def make_prediction(*, input_data) -> dict:
    """Graves esperados al año para cada fila de características."""
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        matriz = np.ascontiguousarray(validated_data.to_numpy(np.float64))
        # El modelo se entrenó con log_km como desplazamiento: se multiplica al final.
        mu = _modelo.predict(matriz) * np.exp(validated_data[config["offset"]].to_numpy())
        results = {"predictions": mu.tolist(), "version": _version, "errors": errors}

    return results