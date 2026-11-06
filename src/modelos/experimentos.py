"""Rejilla de experimentos registrada en MLflow.

ENTRA : data/procesado/panel.parquet, data/procesado/vecindad.parquet
SALE  : una corrida de MLflow por modelo, con sus parametros, sus metricas y
        el modelo serializado.

QUE REGISTRA
  Dos tareas sobre la MISMA matriz de caracteristicas y la MISMA particion,
  de modo que las corridas son comparables entre si:

    conteo        cuantos graves se esperan     Poisson, binomial negativa,
                                                 binomial negativa + Empirical
                                                 Bayes, LightGBM Poisson
    clasificacion si ocurre al menos uno        logistica, LightGBM, CatBoost

  Toda corrida mide ademas la metrica de negocio -- que fraccion de los graves
  de 2021 cae en los primeros k segmentos del ranking -- contra la misma linea
  base del metodo vigente, que entra como una corrida mas para poder leerla en
  la misma tabla de MLflow.

DONDE REGISTRA
  Sin argumentos escribe en ./mlruns y se mira con `mlflow ui`. Con
  --tracking-uri http://<ip>:5000 registra contra el servidor de la EC2.

    python -m src.modelos.experimentos
    python -m src.modelos.experimentos --tracking-uri http://54.0.0.1:5000

NOTA DE ENTORNO
  lightgbm se importa ANTES que pandas. Si pyarrow carga primero su runtime de
  OpenMP, la DLL nativa de lightgbm se cae con una violacion de acceso.
"""

from __future__ import annotations

import lightgbm as lgb  # noqa: I001  debe ir antes que pandas/pyarrow

import argparse
from pathlib import Path

import mlflow
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
from src.modelos.conteo import empirical_bayes, estimar_alpha

EXPERIMENTO = "seguridad-vial-bogota"
PERCENTILES = (0.01, 0.02, 0.05, 0.10)

REJILLA_LGB = (
    {"num_leaves": 31, "learning_rate": 0.05, "num_boost_round": 400},
    {"num_leaves": 63, "learning_rate": 0.05, "num_boost_round": 400},
    {"num_leaves": 31, "learning_rate": 0.10, "num_boost_round": 200},
)
LGB_COMUN = {
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
}


def captura_por_corte(ranking: pd.Series, graves: pd.Series, universo: int) -> dict:
    """Metrica de negocio, identica para las dos tareas y para la linea base."""
    metricas = {}
    for q in PERCENTILES:
        k = int(round(universo * q))
        _, _, pct = captura(ranking, graves, k)
        metricas[f"captura_top_{int(q * 100)}_pct"] = round(pct, 2)
    return metricas


def _contiguo(df: pd.DataFrame | pd.Series) -> np.ndarray:
    return np.ascontiguousarray(df.to_numpy(np.float64))


CARACTERES_PROHIBIDOS = ("/", ":", ".", "%", '"', "'")


def nombre_de_modelo(nombre: str) -> str:
    """MLflow rechaza estos caracteres en el nombre de un modelo. Un nombre de
    corrida como lightgbm_poisson_h31_lr0.05 lleva un punto y el registro
    falla, asi que se sanea aqui en vez de confiar en como venga."""
    for malo in CARACTERES_PROHIBIDOS:
        nombre = nombre.replace(malo, "_")
    return nombre


def registrar_statsmodels(resultados):
    """Serializa un GLM de statsmodels SIN los datos de entrenamiento.

    Un objeto de resultados lleva dentro la matriz de diseno completa. Con
    581.610 filas por 28 columnas eso son unos 130 MB por modelo, y cuatro
    modelos dejaron un almacen de 1,3 GB. `remove_data` guarda solo lo que
    hace falta para predecir.
    """
    return lambda nombre: mlflow.statsmodels.log_model(resultados, name=nombre, remove_data=True)


def registrar_lightgbm(arbol):
    return lambda nombre: mlflow.lightgbm.log_model(arbol, name=nombre)


def _corrida(nombre: str, tarea: str, parametros: dict, metricas: dict, registrar=None):
    with mlflow.start_run(run_name=nombre):
        mlflow.set_tags({"tarea": tarea, "proyecto": "seguridad-vial-bogota"})
        mlflow.log_params(
            {
                "ventana_entrena": f"{ANIO_INICIO}-{ANIO_FIN_ENTRENA}",
                "anio_valida": ANIO_VALIDA,
                **parametros,
            }
        )
        mlflow.log_metrics(metricas)
        if registrar is not None:
            try:
                registrar(nombre_de_modelo(nombre))
            except Exception as error:  # el registro del modelo no debe tumbar la corrida
                mlflow.set_tag("modelo_no_serializado", str(error)[:200])
    linea = "  ".join(f"{k} {v}" for k, v in list(metricas.items())[:4])
    print(f"  {nombre:28s} {linea}")


def experimentos_de_conteo(p, X, y, offset, entrena, valida, universo, graves):
    print("conteo:")
    # --- linea base: el metodo vigente, como corrida comparable -----------
    historico = pd.Series(
        graves.index.map(p[p.anio <= ANIO_FIN_ENTRENA].groupby("CIV").graves.sum()),
        index=graves.index,
    ).astype(float)
    anios = ANIO_FIN_ENTRENA - ANIO_INICIO + 1
    _corrida(
        "linea_base_conteo_historico",
        "conteo",
        {"familia": "ninguna", "descripcion": "orden por conteo historico de graves"},
        {
            **captura_por_corte(historico, graves, universo),
            "mae": round(float((historico / anios - graves).abs().mean()), 4),
            "rmse": round(float(np.sqrt(((historico / anios - graves) ** 2).mean())), 4),
        },
    )

    # --- Poisson y binomial negativa --------------------------------------
    poisson = sm.GLM(
        y[entrena], X[entrena], family=sm.families.Poisson(), offset=offset[entrena]
    ).fit()
    alpha = estimar_alpha(y[entrena].to_numpy(), np.asarray(poisson.fittedvalues, dtype=float))
    mu_poisson = pd.Series(
        np.asarray(poisson.predict(X, offset=offset), dtype=float), index=p.index
    )
    _corrida(
        "poisson_glm",
        "conteo",
        {"familia": "Poisson", "enlace": "log", "desplazamiento": "log_km", "alpha": 0.0},
        {
            **captura_por_corte(pd.Series(mu_poisson[valida].to_numpy(), index=graves.index), graves, universo),
            "aic": round(float(poisson.aic), 1),
            "log_verosimilitud": round(float(poisson.llf), 1),
        },
        registrar_statsmodels(poisson),
    )

    nb = sm.GLM(
        y[entrena],
        X[entrena],
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=offset[entrena],
    ).fit()
    mu = pd.Series(np.asarray(nb.predict(X, offset=offset), dtype=float), index=p.index)
    escala = float(y[entrena].sum() / mu[entrena].sum())
    mu = mu * escala
    mu_valida = pd.Series(mu[valida].to_numpy(), index=graves.index)
    _corrida(
        "binomial_negativa_glm",
        "conteo",
        {
            "familia": "BinomialNegativa",
            "enlace": "log",
            "desplazamiento": "log_km",
            "alpha": round(alpha, 4),
            "reescalado": round(escala, 4),
        },
        {
            **captura_por_corte(mu_valida, graves, universo),
            "aic": round(float(nb.aic), 1),
            "log_verosimilitud": round(float(nb.llf), 1),
            "razon_suma_predicha": round(float(mu_valida.sum() / graves.sum()), 3),
            "elasticidad_longitud": round(1.0 + float(nb.params["log_km"]), 4),
        },
        registrar_statsmodels(nb),
    )

    # --- Empirical Bayes como posproceso ----------------------------------
    mu_hist = mu[entrena].groupby(p.CIV[entrena]).sum()
    obs_hist = y[entrena].groupby(p.CIV[entrena]).sum()
    factor = empirical_bayes(mu_hist, obs_hist, alpha) / mu_hist
    eb = mu_valida * mu_valida.index.map(factor)
    _corrida(
        "binomial_negativa_eb",
        "conteo",
        {"familia": "BinomialNegativa", "posproceso": "EmpiricalBayes_Hauer", "alpha": round(alpha, 4)},
        {
            **captura_por_corte(eb, graves, universo),
            "razon_suma_predicha": round(float(eb.sum() / graves.sum()), 3),
        },
    )

    # --- LightGBM con objetivo Poisson ------------------------------------
    for config in REJILLA_LGB:
        rondas = config["num_boost_round"]
        par = {**LGB_COMUN, **{k: v for k, v in config.items() if k != "num_boost_round"},
               "objective": "poisson"}
        ds = lgb.Dataset(
            _contiguo(X[entrena]),
            label=_contiguo(y[entrena]),
            init_score=_contiguo(offset[entrena]),
            feature_name=list(X.columns),
        )
        arbol = lgb.train(par, ds, num_boost_round=rondas)
        # predict ya aplica la exponencial; el desplazamiento se multiplica.
        pr = pd.Series(
            arbol.predict(_contiguo(X[valida])) * np.exp(offset[valida].to_numpy()),
            index=graves.index,
        )
        _corrida(
            f"lightgbm_poisson_h{config['num_leaves']}_lr{config['learning_rate']}",
            "conteo",
            {"familia": "LightGBM", "objetivo": "poisson", **config, **LGB_COMUN},
            {
                **captura_por_corte(pr, graves, universo),
                "razon_suma_predicha": round(float(pr.sum() / graves.sum()), 3),
            },
            registrar_lightgbm(arbol),
        )


def experimentos_de_clasificacion(p, X, y_conteo, entrena, valida, universo, graves):
    print("clasificacion:")
    y = (y_conteo >= 1).astype(float)
    yv = y[valida].to_numpy()

    def metricas(prob) -> dict:
        serie = pd.Series(prob, index=graves.index)
        return {
            "auc_roc": round(float(roc_auc_score(yv, prob)), 4),
            "auc_pr": round(float(average_precision_score(yv, prob)), 4),
            "brier": round(float(brier_score_loss(yv, prob)), 5),
            **captura_por_corte(serie, graves, universo),
        }

    logit = sm.GLM(y[entrena], X[entrena], family=sm.families.Binomial()).fit()
    _corrida(
        "regresion_logistica",
        "clasificacion",
        {"familia": "Binomial", "enlace": "logit"},
        {**metricas(np.asarray(logit.predict(X[valida]), dtype=float)),
         "aic": round(float(logit.aic), 1)},
        registrar_statsmodels(logit),
    )

    for config in REJILLA_LGB:
        rondas = config["num_boost_round"]
        par = {**LGB_COMUN, **{k: v for k, v in config.items() if k != "num_boost_round"},
               "objective": "binary"}
        ds = lgb.Dataset(_contiguo(X[entrena]), label=_contiguo(y[entrena]),
                         feature_name=list(X.columns))
        arbol = lgb.train(par, ds, num_boost_round=rondas)
        _corrida(
            f"lightgbm_binario_h{config['num_leaves']}_lr{config['learning_rate']}",
            "clasificacion",
            {"familia": "LightGBM", "objetivo": "binary", **config, **LGB_COMUN},
            metricas(arbol.predict(_contiguo(X[valida]))),
            registrar_lightgbm(arbol),
        )

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  catboost no instalado, se omite esa familia")
        return
    # allow_writing_files: sin esto CatBoost deja un catboost_info/ de logs
    # en la raiz del repositorio cada vez que entrena.
    gato = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=7,
        allow_writing_files=False,
    )
    gato.fit(X[entrena].to_numpy(np.float64), y[entrena].to_numpy(np.float64))
    prob = gato.predict_proba(X[valida].to_numpy(np.float64))[:, 1]
    _corrida(
        "catboost",
        "clasificacion",
        {"familia": "CatBoost", "iterations": 400, "learning_rate": 0.05, "depth": 6},
        metricas(prob),
        lambda nombre: mlflow.catboost.log_model(gato, name=nombre),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--vecindad", type=Path, default=VECINDAD)
    ap.add_argument("--tracking-uri", default=None,
                    help="http://<ip>:5000 para registrar contra la EC2. Sin esto, ./mlruns")
    ap.add_argument("--experimento", default=EXPERIMENTO)
    a = ap.parse_args()

    if a.tracking_uri:
        mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(a.experimento)
    print(f"registrando en: {mlflow.get_tracking_uri()}  |  experimento: {a.experimento}\n")

    p = cargar(a.panel, a.vecindad)
    X = matriz_diseno(p)
    y = p.graves.astype(float)
    offset = p.log_long_km.astype(float)
    entrena, valida = particion(p)
    universo = int(p.CIV.nunique())
    graves = pd.Series(y[valida].to_numpy(), index=p.CIV[valida].to_numpy())

    experimentos_de_conteo(p, X, y, offset, entrena, valida, universo, graves)
    experimentos_de_clasificacion(p, X, y, entrena, valida, universo, graves)
    print(f"\nlisto. {len(X.columns)} variables, {int(entrena.sum()):,} filas de entrenamiento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
