"""Entrena lightgbm_poisson_h31_lr0.1 y lo deja listo para empaquetar.

    python -m src.modelos.empaquetar
"""
import lightgbm as lgb  # antes que pandas (nota de experimentos.py)

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from src.modelos.caracteristicas import cargar, matriz_diseno, particion
from src.modelos.experimentos import LGB_COMUN

RAIZ = Path(__file__).resolve().parents[2]
PAQUETE = RAIZ / "modelo-riesgo" / "modelo_riesgo"
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

    (PAQUETE / "trained").mkdir(parents=True, exist_ok=True)
    (PAQUETE / "datasets").mkdir(parents=True, exist_ok=True)
    arbol.save_model(str(PAQUETE / "trained" / f"modelo-riesgo-output{VERSION}.txt"))

    config = {
        "package_name": "modelo-riesgo",
        "pipeline_save_file": "modelo-riesgo-output",
        "test_data_file": "muestra_2021.csv",
        "modelo": "lightgbm_poisson_h31_lr0.1",
        "target": "graves",
        "offset": "log_km",
        "anio_entrenamiento": [2016, 2020],
        "anio_validacion": 2021,
        "num_boost_round": RONDAS,
        "parametros": PARAMETROS,
        "features": list(X.columns),
    }
    (PAQUETE / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


    feats = X[valida].copy()
    feats.insert(0, "CIV", p.CIV[valida].to_numpy())
    feats.to_parquet(RAIZ / "data" / "procesado" / "caracteristicas_2021.parquet", index=False)
    feats.sample(200, random_state=7).to_csv(PAQUETE / "datasets" / "muestra_2021.csv", index=False)

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
