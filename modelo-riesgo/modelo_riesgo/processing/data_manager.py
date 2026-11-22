import lightgbm as lgb
import pandas as pd

from modelo_riesgo.config.core import DATASETS, TRAINED


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Carga un archivo de la carpeta datasets del paquete."""
    return pd.read_csv(DATASETS / file_name)


def load_model(*, file_name: str) -> lgb.Booster:
    """Carga el modelo entrenado de la carpeta trained del paquete."""
    return lgb.Booster(model_file=str(TRAINED / file_name))