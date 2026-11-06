"""Modelo de conteo de siniestros graves por segmento y anio.

ENTRA : data/procesado/panel.parquet
SALE  : reportes/modelos/conteo.json              metricas, calibracion y coeficientes
        data/procesado/prediccion_conteo.parquet  riesgo esperado por CIV

QUE HACE
  1. Binomial negativa con enlace logaritmico. El desplazamiento es log(km),
     pero log(km) entra ADEMAS como regresor libre: forzar elasticidad 1 en la
     longitud descalibra el modelo (ver nota de la elasticidad mas abajo).
  2. Empirical Bayes de Hauer como posproceso, para corregir la regresion a la
     media, que es la primera de las dos fallas del metodo vigente.
  3. Evalua con la misma funcion captura() que fija la linea base, para que el
     44,5 % del metodo vigente y el del modelo sean comparables.

VENTANA
  Entrena 2016-2020 y valida 2021. 2015 no entra: el rezago lo deja vacio.
  La ventana TIENE que empezar en 2016. Todas las obras utilizables estan
  marcadas en 2016, asi que desde 2017 las columnas post_ y tratado_ son
  identicas y el efecto de la intervencion deja de estar identificado.

HISTORIA DE VENTANA COMPARABLE
  No se usa el acumulado crudo: su ventana crece con el anio (un anio de
  historia en 2016, cinco en 2021), asi que el coeficiente mezcla nivel de
  riesgo con tiempo transcurrido y se dispara. Se usa la media por anio
  previo, que significa lo mismo en toda fila, con n_prev como control.

ELASTICIDAD DE LA LONGITUD
  Con el desplazamiento solo, el modelo predice 17.413 graves para 2021 contra
  10.717 observados, con un maximo de 132 en un segmento donde se observaron 11.
  Liberando log(km) la elasticidad estimada baja de 1 a ~0,4: los siniestros no
  escalan con la longitud del tramo.

    python -m src.modelos.conteo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from src.exploracion.perfilamiento import captura

RAIZ = Path(__file__).resolve().parents[2]
ANIO_INICIO = 2016
ANIO_FIN_ENTRENA = 2020
ANIO_VALIDA = 2021
ANIO_CENTRO = 2019
CLASE_REFERENCIA = 3
PERCENTILES = (0.01, 0.02, 0.05, 0.10)
DECILES = 10


def preparar(panel: pd.DataFrame) -> pd.DataFrame:
    """Anade las derivadas que el modelo necesita. No toca el panel en disco."""
    p = panel.sort_values(["CIV", "anio"]).reset_index(drop=True)
    p["n_prev"] = (p.anio - 2015).astype(float)
    p["hist_media"] = p.graves_acum_previo / p.n_prev.clip(lower=1)
    p["hist_media_log"] = np.log1p(p.hist_media)
    p["rezago1_log"] = np.log1p(p.groupby("CIV").graves.shift(1))
    p["log_km"] = p.log_long_km
    return p


def matriz_diseno(p: pd.DataFrame) -> pd.DataFrame:
    """Matriz sobre el panel completo, para que las indicadoras de clase de via
    sean las mismas en entrenamiento y en validacion."""
    post = sorted(c for c in p.columns if c.startswith("post_"))
    tratado = sorted(c for c in p.columns if c.startswith("tratado_"))

    X = pd.DataFrame(index=p.index)
    X["hist_media_log"] = p.hist_media_log.astype(float)
    X["rezago1_log"] = p.rezago1_log.fillna(0.0).astype(float)
    # n_prev NO entra: es anio-2015 y tendencia es anio-2019, asi que difieren
    # en una constante y la matriz queda deficiente de rango contra el intercepto.
    X["log_km"] = p.log_km.astype(float)
    X["carriles"] = p.carriles.astype(float)
    X["vel_reg"] = p.vel_reg.fillna(p.vel_reg.median()).astype(float)
    X["vel_faltante"] = p.vel_faltante.astype(float)
    X["tendencia"] = (p.anio - ANIO_CENTRO).astype(float)
    X["pandemia"] = (p.anio == 2020).astype(float)
    for col in post + tratado:
        X[col] = p[col].astype(float)

    clase = pd.get_dummies(p.clase_via.astype(int), prefix="clase", dtype=float)
    clase = clase.drop(columns=[f"clase_{CLASE_REFERENCIA}"], errors="ignore")
    return sm.add_constant(pd.concat([X, clase], axis=1), has_constant="add")


def estimar_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Dispersion NB2 por la regresion auxiliar de Cameron y Trivedi:
    ((y - mu)^2 - y) / mu  regresada en mu sin intercepto."""
    z = ((y - mu) ** 2 - y) / mu
    return float((mu * z).sum() / (mu**2).sum())


def empirical_bayes(mu_hist: pd.Series, obs_hist: pd.Series, alpha: float) -> pd.Series:
    """Hauer. w = 1/(1 + alpha*mu): cuanta menos historia, mas pesa el modelo."""
    w = 1.0 / (1.0 + alpha * mu_hist)
    return w * mu_hist + (1.0 - w) * obs_hist


def _punto(y: pd.Series, pred: pd.Series) -> dict:
    err = pred - y
    return {
        "mae": round(float(err.abs().mean()), 4),
        "rmse": round(float(np.sqrt((err**2).mean())), 4),
        "spearman": round(float(spearmanr(pred, y).statistic), 4),
    }


def _ranking(ranking: pd.Series, validacion: pd.Series, universo: int) -> dict:
    ficha = {}
    for q in PERCENTILES:
        k = int(round(universo * q))
        cap, tot, pct = captura(ranking, validacion, k)
        ficha[f"top_{int(q * 100)}_pct"] = {
            "segmentos": k,
            "capturados": cap,
            "de": tot,
            "pct": round(pct, 1),
        }
    return ficha


def _calibracion(pred: pd.Series, obs: pd.Series) -> dict:
    """Suma, maximo y tabla por decil de prediccion. Un modelo util para el
    tablero tiene que acertar el nivel, no solo el orden."""
    d = pd.DataFrame({"pred": pred, "obs": obs})
    d["decil"] = pd.qcut(d.pred.rank(method="first"), DECILES, labels=False)
    tabla = d.groupby("decil").agg(
        predicho_medio=("pred", "mean"), observado_medio=("obs", "mean"), segmentos=("obs", "size")
    )
    return {
        "suma_predicha": round(float(pred.sum()), 1),
        "suma_observada": int(obs.sum()),
        "razon_suma": round(float(pred.sum() / obs.sum()), 3),
        "maximo_predicho": round(float(pred.max()), 2),
        "maximo_observado": int(obs.max()),
        "por_decil": {
            int(i): {
                "predicho_medio": round(float(r.predicho_medio), 4),
                "observado_medio": round(float(r.observado_medio), 4),
                "segmentos": int(r.segmentos),
            }
            for i, r in tabla.iterrows()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=RAIZ / "data" / "procesado" / "panel.parquet")
    ap.add_argument("--salida", type=Path, default=RAIZ / "reportes" / "modelos")
    ap.add_argument("--predicciones", type=Path, default=RAIZ / "data" / "procesado")
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    a.predicciones.mkdir(parents=True, exist_ok=True)

    p = preparar(pd.read_parquet(a.panel))
    universo = int(p.CIV.nunique())

    X = matriz_diseno(p)
    y = p.graves.astype(float)
    desplazamiento = p.log_long_km.astype(float)
    entrena = p.anio.between(ANIO_INICIO, ANIO_FIN_ENTRENA).to_numpy()
    valida = (p.anio == ANIO_VALIDA).to_numpy()

    # --- Poisson, solo para medir la sobredispersion y sacar alpha --------
    poisson = sm.GLM(
        y[entrena], X[entrena], family=sm.families.Poisson(), offset=desplazamiento[entrena]
    ).fit()
    alpha = estimar_alpha(y[entrena].to_numpy(), np.asarray(poisson.fittedvalues, dtype=float))

    # --- Binomial negativa, el modelo -------------------------------------
    nb = sm.GLM(
        y[entrena],
        X[entrena],
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=desplazamiento[entrena],
    ).fit()

    mu = pd.Series(np.asarray(nb.predict(X, offset=desplazamiento), dtype=float), index=p.index)

    # Reescalado de nivel: el orden no cambia, el nivel queda creible para el
    # tablero. El factor se estima SOLO con los anios de entrenamiento.
    escala = float(y[entrena].sum() / mu[entrena].sum())
    mu = mu * escala

    civ = p.CIV
    mu_hist = mu[entrena].groupby(civ[entrena]).sum()
    obs_hist = y[entrena].groupby(civ[entrena]).sum()
    factor_eb = empirical_bayes(mu_hist, obs_hist, alpha) / mu_hist

    pred = pd.DataFrame(
        {
            "CIV": civ[valida].to_numpy(),
            "graves_2021": y[valida].to_numpy().astype(int),
            "mu_2021": mu[valida].to_numpy(),
        }
    ).set_index("CIV")
    pred["eb_2021"] = pred.mu_2021 * pred.index.map(factor_eb)
    pred["conteo_2015_2020"] = pred.index.map(p[p.anio <= 2020].groupby("CIV").graves.sum())

    obs = pred.graves_2021
    historico = pred.conteo_2015_2020
    anios_hist = ANIO_FIN_ENTRENA - ANIO_INICIO + 1

    ficha = {
        "ventana": {
            "entrena": [ANIO_INICIO, ANIO_FIN_ENTRENA],
            "valida": ANIO_VALIDA,
            "filas_entrena": int(entrena.sum()),
            "filas_valida": int(valida.sum()),
            "universo_civ": universo,
            "factor_de_escala": round(escala, 4),
        },
        "sobredispersion": {
            "media_segmento_anio": round(float(y[entrena].mean()), 4),
            "varianza_segmento_anio": round(float(y[entrena].var()), 4),
            "indice_de_dispersion": round(float(y[entrena].var() / y[entrena].mean()), 2),
            "alpha_nb2": round(alpha, 4),
        },
        "ajuste": {
            "poisson": {"log_verosimilitud": round(float(poisson.llf), 1), "aic": round(float(poisson.aic), 1)},
            "binomial_negativa": {"log_verosimilitud": round(float(nb.llf), 1), "aic": round(float(nb.aic), 1)},
        },
        "elasticidad_longitud": round(1.0 + float(nb.params["log_km"]), 4),
        "coeficientes": {
            nombre: {
                "coef": round(float(c), 4),
                "razon_de_tasas": round(float(np.exp(c)), 4),
                "ee": round(float(e), 4),
                "p": round(float(pv), 6),
            }
            for nombre, c, e, pv in zip(nb.params.index, nb.params, nb.bse, nb.pvalues)
        },
        "calibracion_2021": {
            "binomial_negativa": _calibracion(pred.mu_2021, obs),
            "binomial_negativa_eb": _calibracion(pred.eb_2021, obs),
        },
        "evaluacion_2021": {
            "linea_base_conteo_historico": {
                **_ranking(historico, obs, universo),
                **_punto(obs, historico / anios_hist),
            },
            "binomial_negativa": {**_ranking(pred.mu_2021, obs, universo), **_punto(obs, pred.mu_2021)},
            "binomial_negativa_eb": {**_ranking(pred.eb_2021, obs, universo), **_punto(obs, pred.eb_2021)},
        },
    }

    (a.salida / "conteo.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pred.reset_index().to_parquet(a.predicciones / "prediccion_conteo.parquet", index=False)

    ev = ficha["evaluacion_2021"]
    cal = ficha["calibracion_2021"]["binomial_negativa"]
    print(f"alpha NB2 {alpha:.4f} | elasticidad longitud {ficha['elasticidad_longitud']:.3f} | escala {escala:.3f}")
    print(f"calibracion: suma predicha {cal['suma_predicha']:,.0f} contra {cal['suma_observada']:,} observados "
          f"(razon {cal['razon_suma']}), maximo {cal['maximo_predicho']} contra {cal['maximo_observado']}")
    for nombre, clave in (("linea base", "linea_base_conteo_historico"), ("NB", "binomial_negativa"), ("NB+EB", "binomial_negativa_eb")):
        f = ev[clave]
        print(f"  {nombre:10s} top1% {f['top_1_pct']['pct']:4.1f}  top2% {f['top_2_pct']['pct']:4.1f}  "
              f"top5% {f['top_5_pct']['pct']:4.1f}  top10% {f['top_10_pct']['pct']:4.1f}  rho {f['spearman']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
