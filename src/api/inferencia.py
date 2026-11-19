"""Inferencia en vivo dentro de la API.

La API calcula la prediccion en cada peticion con el modelo empaquetado, en
vez de devolver una tabla de predicciones ya calculada. Lo que lee de disco
son las caracteristicas del segmento, que son la entrada del modelo, y el
modelo mismo.

ENTRA : modelos/conteo.txt, modelos/clasificacion.txt, modelos/metadatos.json
        data/procesado/caracteristicas_inferencia.parquet
  Las rutas se cambian con MODELOS_DIR y DATOS_DIR.

Los artefactos los produce `python -m src.modelos.empaquetar`, que los saca de
MLflow. El vinculo con la corrida queda en metadatos.json.

Sobre 116.322 segmentos la prediccion tarda unas decimas de segundo, asi que
se recalcula en cada peticion y no hace falta cachearla.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb  # noqa: I001  debe ir antes que pandas/pyarrow

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
NO_SON_VARIABLES = ("CIV", "anio_ref", "log_long_km", "factor_eb")


def carpeta_modelos() -> Path:
    return Path(os.environ.get("MODELOS_DIR", RAIZ / "modelos"))


def carpeta_datos() -> Path:
    return Path(os.environ.get("DATOS_DIR", RAIZ / "data" / "procesado"))


def disponible() -> bool:
    """Si falta cualquier pieza, la API lo dice en /salud en vez de reventar."""
    m, d = carpeta_modelos(), carpeta_datos()
    return (
        (m / "conteo.txt").exists()
        and (m / "clasificacion.txt").exists()
        and (d / "caracteristicas_inferencia.parquet").exists()
    )


@lru_cache(maxsize=1)
def motor() -> dict:
    """Carga los modelos y las caracteristicas una vez por proceso."""
    m, d = carpeta_modelos(), carpeta_datos()
    rasgos = pd.read_parquet(d / "caracteristicas_inferencia.parquet")
    metadatos = json.loads((m / "metadatos.json").read_text(encoding="utf-8"))
    return {
        "conteo": lgb.Booster(model_file=str(m / "conteo.txt")),
        "clasificacion": lgb.Booster(model_file=str(m / "clasificacion.txt")),
        "rasgos": rasgos,
        "variables": [c for c in rasgos.columns if c not in NO_SON_VARIABLES],
        "metadatos": metadatos,
    }


def anios() -> list[int]:
    return sorted(int(a) for a in motor()["rasgos"].anio_ref.unique())


def predecir(anio: int, civs: list[int] | None = None) -> pd.DataFrame:
    """Corre los dos modelos sobre los segmentos pedidos.

    Devuelve CIV, esperado, esperado_eb y prob_grave, que son las mismas
    columnas que antes se leian de la tabla.
    """
    m = motor()
    rasgos = m["rasgos"]
    sel = rasgos[rasgos.anio_ref == anio]
    if civs is not None:
        sel = sel[sel.CIV.isin(civs)]
    if sel.empty:
        return pd.DataFrame(columns=["CIV", "esperado", "esperado_eb", "prob_grave"])

    X = np.ascontiguousarray(sel[m["variables"]].to_numpy(np.float64))
    # El objetivo poisson ya devuelve la exponencial: el desplazamiento se
    # multiplica aparte, igual que al entrenar.
    esperado = m["conteo"].predict(X) * np.exp(sel.log_long_km.to_numpy(np.float64))
    return pd.DataFrame(
        {
            "CIV": sel.CIV.to_numpy(),
            "esperado": esperado,
            "esperado_eb": esperado * sel.factor_eb.to_numpy(np.float64),
            "prob_grave": m["clasificacion"].predict(X),
        }
    )


def ficha() -> dict:
    """Lo que /salud cuenta sobre el modelo que esta sirviendo."""
    if not disponible():
        return {"inferencia": "no disponible", "motivo": "faltan los artefactos del modelo"}
    meta = motor()["metadatos"]
    return {
        "inferencia": "en vivo",
        "empaquetado": meta.get("empaquetado"),
        "experimento": meta.get("experimento"),
        "modelos": {t: d["corrida"] for t, d in meta.get("modelos", {}).items()},
    }
