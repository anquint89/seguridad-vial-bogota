"""Clasificador de ocurrencia: probabilidad de al menos un grave en el anio.

ENTRA : data/procesado/panel.parquet
SALE  : reportes/modelos/clasificador.json
        data/procesado/prediccion_clasificador.parquet

POR QUE UN SEGUNDO MODELO
  El de conteo responde "cuantos graves se esperan". Este responde "cual es la
  probabilidad de que ocurra al menos uno". Para priorizar con presupuesto
  limitado la segunda pregunta es la que importa en los tramos de riesgo bajo,
  donde el conteo esperado es una fraccion y no se puede leer como decision.

QUE COMPARA
  Regresion logistica, que da razones de momios interpretables, contra
  LightGBM, que captura no linealidades e interacciones. Las dos sobre la
  matriz de caracteristicas.py, la misma que usa el modelo de conteo, para que
  la comparacion sea de forma funcional y no de informacion disponible.

VENTANA Y VARIABLES
  Las define caracteristicas.py: entrena 2016-2020, valida 2021, e incluye la
  historia de los segmentos vecinos.

NOTA DE ENTORNO
  lightgbm se importa ANTES que pandas. Si pyarrow carga primero su runtime de
  OpenMP, la DLL nativa de lightgbm se cae con una violacion de acceso al
  construir el Dataset.

    python -m src.modelos.clasificador
"""

from __future__ import annotations

import lightgbm as lgb  # noqa: I001  debe ir antes que pandas/pyarrow

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.exploracion.perfilamiento import captura
from src.modelos.caracteristicas import (
    ANIO_FIN_ENTRENA,
    ANIO_INICIO,
    ANIO_VALIDA,
    PANEL,
    VECINDAD,
    cargar,
    matriz_diseno,
    particion,
)

RAIZ = Path(__file__).resolve().parents[2]
PERCENTILES = (0.01, 0.02, 0.05, 0.10)
DECILES = 10
PARAMETROS_LGB = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
}
RONDAS = 400


def _desempeno(y: np.ndarray, puntaje: np.ndarray, es_probabilidad: bool = True) -> dict:
    """AUC y precision media aceptan cualquier puntaje monotono. El Brier no:
    exige una probabilidad, asi que no aplica a la linea base, que es un conteo."""
    ficha = {
        "auc_roc": round(float(roc_auc_score(y, puntaje)), 4),
        "auc_pr": round(float(average_precision_score(y, puntaje)), 4),
    }
    if es_probabilidad:
        ficha["brier"] = round(float(brier_score_loss(y, puntaje)), 5)
    return ficha


def _calibracion(prob: pd.Series, y: pd.Series) -> dict:
    d = pd.DataFrame({"prob": prob, "y": y})
    d["decil"] = pd.qcut(d.prob.rank(method="first"), DECILES, labels=False)
    tabla = d.groupby("decil").agg(
        probabilidad_media=("prob", "mean"), tasa_observada=("y", "mean"), segmentos=("y", "size")
    )
    return {
        "probabilidad_media": round(float(prob.mean()), 5),
        "tasa_observada": round(float(y.mean()), 5),
        "por_decil": {
            int(i): {
                "probabilidad_media": round(float(r.probabilidad_media), 5),
                "tasa_observada": round(float(r.tasa_observada), 5),
                "segmentos": int(r.segmentos),
            }
            for i, r in tabla.iterrows()
        },
    }


def _ranking(prob: pd.Series, graves: pd.Series, universo: int) -> dict:
    """Misma metrica de negocio que el conteo: graves de 2021 capturados al
    ordenar por probabilidad y quedarse con los primeros k."""
    ficha = {}
    for q in PERCENTILES:
        k = int(round(universo * q))
        cap, tot, pct = captura(prob, graves, k)
        ficha[f"top_{int(q * 100)}_pct"] = {
            "segmentos": k,
            "capturados": cap,
            "de": tot,
            "pct": round(pct, 1),
        }
    return ficha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--vecindad", type=Path, default=VECINDAD)
    ap.add_argument("--salida", type=Path, default=RAIZ / "reportes" / "modelos")
    ap.add_argument("--predicciones", type=Path, default=RAIZ / "data" / "procesado")
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    a.predicciones.mkdir(parents=True, exist_ok=True)

    p = cargar(a.panel, a.vecindad)
    universo = int(p.CIV.nunique())

    X = matriz_diseno(p)
    y = (p.graves >= 1).astype(float)  # ocurrencia, no conteo
    entrena, valida = particion(p)
    yv = y[valida].to_numpy()

    # --- Regresion logistica ----------------------------------------------
    logit = sm.GLM(y[entrena], X[entrena], family=sm.families.Binomial()).fit()
    prob_logit = np.asarray(logit.predict(X[valida]), dtype=float)

    # --- LightGBM ---------------------------------------------------------
    columnas = list(X.columns)
    ds = lgb.Dataset(
        np.ascontiguousarray(X[entrena].to_numpy(np.float64)),
        label=np.ascontiguousarray(y[entrena].to_numpy(np.float64)),
        feature_name=columnas,
    )
    arbol = lgb.train(PARAMETROS_LGB, ds, num_boost_round=RONDAS)
    prob_lgb = arbol.predict(np.ascontiguousarray(X[valida].to_numpy(np.float64)))

    civ = p.CIV[valida].to_numpy()
    graves = pd.Series(p.graves[valida].to_numpy(), index=civ)
    pred = pd.DataFrame(
        {"CIV": civ, "ocurrencia_2021": yv.astype(int), "graves_2021": p.graves[valida].to_numpy(),
         "prob_logit": prob_logit, "prob_lgb": prob_lgb}
    ).set_index("CIV")

    # Referencia: ordenar por conteo historico, como hace el metodo vigente.
    historico = pd.Series(
        pred.index.map(p[p.anio <= ANIO_FIN_ENTRENA].groupby("CIV").graves.sum()), index=pred.index
    )

    importancia = pd.Series(arbol.feature_importance("gain"), index=columnas).sort_values(
        ascending=False
    )

    ficha = {
        "ventana": {
            "entrena": [ANIO_INICIO, ANIO_FIN_ENTRENA],
            "valida": ANIO_VALIDA,
            "filas_entrena": int(entrena.sum()),
            "filas_valida": int(valida.sum()),
            "universo_civ": universo,
        },
        "respuesta": {
            "definicion": "al menos un siniestro grave en el anio",
            "positivos_entrena": int(y[entrena].sum()),
            "tasa_entrena": round(float(y[entrena].mean()), 5),
            "positivos_valida": int(yv.sum()),
            "tasa_valida": round(float(yv.mean()), 5),
        },
        "desempeno_2021": {
            "regresion_logistica": {
                **_desempeno(yv, prob_logit),
                **_ranking(pred.prob_logit, graves, universo),
            },
            "lightgbm": {**_desempeno(yv, prob_lgb), **_ranking(pred.prob_lgb, graves, universo)},
            "linea_base_conteo_historico": {
                **_desempeno(yv, historico.to_numpy(dtype=float), es_probabilidad=False),
                **_ranking(historico, graves, universo),
            },
        },
        "calibracion_2021": {
            "regresion_logistica": _calibracion(pred.prob_logit, pred.ocurrencia_2021),
            "lightgbm": _calibracion(pred.prob_lgb, pred.ocurrencia_2021),
        },
        "razones_de_momios": {
            nombre: {
                "coef": round(float(c), 4),
                "razon_de_momios": round(float(np.exp(c)), 4),
                "ee": round(float(e), 4),
                "p": round(float(pv), 6),
            }
            for nombre, c, e, pv in zip(logit.params.index, logit.params, logit.bse, logit.pvalues)
        },
        "importancia_lightgbm": {
            nombre: round(float(100 * v / importancia.sum()), 2)
            for nombre, v in importancia.head(15).items()
        },
    }

    (a.salida / "clasificador.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pred.reset_index().to_parquet(a.predicciones / "prediccion_clasificador.parquet", index=False)

    d = ficha["desempeno_2021"]
    print(f"positivos: {int(yv.sum()):,} de {len(yv):,} ({100 * yv.mean():.2f} %)")
    for nombre, clave in (
        ("linea base", "linea_base_conteo_historico"),
        ("logistica", "regresion_logistica"),
        ("lightgbm", "lightgbm"),
    ):
        f = d[clave]
        brier = f"{f['brier']:.5f}" if "brier" in f else "   n/a"
        print(
            f"  {nombre:11s} AUC {f['auc_roc']:.4f}  PR {f['auc_pr']:.4f}  Brier {brier}"
            f"  | top5% {f['top_5_pct']['pct']:4.1f}  top10% {f['top_10_pct']['pct']:4.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
