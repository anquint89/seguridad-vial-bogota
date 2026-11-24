"""Inferencia en vivo dentro de la API.

La API calcula la prediccion en cada peticion con el paquete `modelo-riesgo`,
en vez de devolver una tabla de predicciones ya calculada. Los modelos los
trae el paquete instalado, y lo que se lee de disco son las caracteristicas
del segmento, que son la entrada.

ENTRA : el paquete modelo-riesgo instalado (pip install modelo_riesgo-*.whl)
        data/procesado/caracteristicas_inferencia.parquet
  La carpeta de datos se cambia con DATOS_DIR.

Los artefactos del paquete los produce `python -m src.modelos.empaquetar`, que
los saca de MLflow siguiendo reportes/modelos/seleccion.json. El vinculo con
la corrida queda escrito en el config.yml del paquete.

Sobre 116.322 segmentos la prediccion tarda unas decimas de segundo, asi que
se recalcula en cada peticion y no hace falta cachearla.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import lightgbm  # noqa: F401,I001  debe cargar antes que pandas/pyarrow

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARACTERISTICAS = "caracteristicas_inferencia.parquet"


def carpeta_datos() -> Path:
    return Path(os.environ.get("DATOS_DIR", RAIZ / "data" / "procesado"))


def disponible() -> bool:
    """Si falta cualquier pieza, la API lo dice en /salud en vez de reventar."""
    if not (carpeta_datos() / CARACTERISTICAS).exists():
        return False
    try:
        import modelo_riesgo.predict  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def motor() -> dict:
    """Carga las caracteristicas una vez por proceso. Los modelos los tiene ya
    cargados el paquete."""
    from modelo_riesgo import __version__ as version
    from modelo_riesgo.config.core import config

    return {
        "rasgos": pd.read_parquet(carpeta_datos() / CARACTERISTICAS),
        "config": config,
        "version": version,
    }


def anios() -> list[int]:
    return sorted(int(a) for a in motor()["rasgos"].anio_ref.unique())


def predecir(anio: int, civs: list[int] | None = None) -> pd.DataFrame:
    """Corre los dos modelos sobre los segmentos pedidos.

    Devuelve CIV, esperado, esperado_eb y prob_grave, que son las mismas
    columnas que antes se leian de la tabla.
    """
    from modelo_riesgo.predict import make_prediction

    rasgos = motor()["rasgos"]
    sel = rasgos[rasgos.anio_ref == anio]
    if civs is not None:
        sel = sel[sel.CIV.isin(civs)]
    vacio = pd.DataFrame(columns=["CIV", "esperado", "esperado_eb", "prob_grave"])
    if sel.empty:
        return vacio

    salida = make_prediction(input_data=sel)
    if salida["errors"] is not None:
        raise RuntimeError(f"el modelo rechazo las caracteristicas: {salida['errors']}")
    return pd.DataFrame(salida["predictions"])


def ficha() -> dict:
    """Lo que /salud cuenta sobre los modelos que esta sirviendo."""
    if not disponible():
        return {"inferencia": "no disponible", "motivo": "faltan los artefactos del modelo"}
    cfg = motor()["config"]
    return {
        "inferencia": "en vivo",
        "paquete": f"{cfg['package_name']} {motor()['version']}",
        "empaquetado": cfg.get("empaquetado"),
        "experimento": cfg.get("experimento"),
        "modelos": {t: d["corrida"] for t, d in cfg.get("modelos", {}).items()},
    }
