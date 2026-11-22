"""Entrena lightgbm_poisson_h31_lr0.1 y lo deja listo para empaquetar.

    python -m src.modelos.empaquetar
"""
import lightgbm as lgb  # antes que pandas (nota de experimentos.py)

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelos.caracteristicas import cargar, matriz_diseno, particion
from src.modelos.experimentos import LGB_COMUN

RAIZ = Path(__file__).resolve().parents[2]
VERSION = "0.0.1"
PARAMETROS = {**LGB_COMUN, "num_leaves": 31, "learning_rate": 0.10, "objective": "poisson"}
RONDAS = 200


def contiguo(x):
    return np.ascontiguousarray(x.to_numpy(np.float64))


def main() -> int:
    p = cargar()
    X = matriz_diseno(p)
    y = p.graves.astype(float)
    entrena, valida = particion(p)

    ds = lgb.Dataset(contiguo(X[entrena]), label=contiguo(y[entrena]),
                     init_score=contiguo(X.log_km[entrena]), feature_name=list(X.columns))
    arbol = lgb.train(PARAMETROS, ds, num_boost_round=RONDAS)

    destino = RAIZ / "modelo-riesgo" / "modelo_riesgo" / "trained"
    destino.mkdir(parents=True, exist_ok=True)
    arbol.save_model(str(destino / f"modelo-riesgo-output{VERSION}.txt"))
    (destino / "columnas.json").write_text(json.dumps(list(X.columns)), encoding="utf-8")

    feats = X[valida].copy()
    feats.insert(0, "CIV", p.CIV[valida].to_numpy())
    feats.to_parquet(RAIZ / "data" / "procesado" / "caracteristicas_2021.parquet", index=False)

    # Control: debe dar cerca de 45,87 % (el valor de MLflow).
    mu = arbol.predict(contiguo(X[valida])) * np.exp(X.log_km[valida].to_numpy())
    real = pd.Series(y[valida].to_numpy(), index=p.CIV[valida].to_numpy())
    mu = pd.Series(mu, index=real.index)
    k = int(round(len(mu) * 0.05))
    captura = 100 * real.loc[mu.sort_values(ascending=False).head(k).index].sum() / real.sum()
    print(f"suma predicha {mu.sum():,.0f} contra {real.sum():,.0f} observados | captura top 5 %: {captura:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())