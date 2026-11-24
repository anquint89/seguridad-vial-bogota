"""Rejilla de experimentos registrada en MLflow.

ENTRA : data/procesado/panel.parquet, data/procesado/vecindad.parquet
SALE  : una corrida de MLflow por modelo, con sus parametros, sus metricas y
        el modelo serializado.

QUE REGISTRA
  Las dos tareas sobre la misma matriz y la misma particion, para que las
  corridas sean comparables. Conteo: Poisson, binomial negativa, con y sin
  Empirical Bayes, y LightGBM. Clasificacion: logistica, LightGBM y CatBoost.
  Todas miden ademas la captura, y la linea base del metodo vigente entra como
  una corrida mas para leerla en la misma tabla.

  El puntaje de validacion de cada corrida se vuelca a un parquet, que es lo
  que luego compara seleccion.py sin reentrenar.

OJO: lightgbm se importa ANTES que pandas. Si pyarrow carga primero su runtime
de OpenMP, la DLL nativa de lightgbm se cae con una violacion de acceso.

    python -m src.modelos.experimentos
    python -m src.modelos.experimentos --tracking-uri http://<ip>:5000

  Con --exportar no entrena: vuelca las corridas ya registradas a
  reportes/modelos/experimentos.json, de donde el reporte cita las cifras.
"""

from __future__ import annotations

import lightgbm as lgb  # noqa: I001  debe ir antes que pandas/pyarrow

import argparse
import json
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

RAIZ = Path(__file__).resolve().parents[2]
EXPORTACION = RAIZ / "reportes" / "modelos" / "experimentos.json"
PUNTAJES_VALIDACION = RAIZ / "data" / "procesado" / "puntajes_validacion.parquet"

EXPERIMENTO = "seguridad-vial-bogota"
PERCENTILES = (0.01, 0.02, 0.05, 0.10)

# La metrica con la que se ordena cada tarea al exportar: la de negocio en el
# conteo, el AUC en la clasificacion.
METRICA_PRINCIPAL = {"conteo": "captura_top_5_pct", "clasificacion": "auc_roc"}

# Rondas por tasa de aprendizaje: cuanto mas baja la tasa, mas arboles hacen
# falta para llegar al mismo sitio. No se usa parada temprana a proposito,
# porque el unico conjunto disponible para vigilarla seria 2021, que es el de
# validacion, y mirarlo durante el entrenamiento lo invalidaria.
RONDAS_POR_TASA = {0.03: 700, 0.05: 500, 0.10: 250}

REJILLA_LGB = tuple(
    {
        "num_leaves": hojas,
        "learning_rate": tasa,
        "min_data_in_leaf": minimo,
        "num_boost_round": RONDAS_POR_TASA[tasa],
    }
    for hojas in (15, 31, 63, 127)
    for tasa in (0.03, 0.05, 0.10)
    for minimo in (100, 300)
)

REJILLA_CATBOOST = tuple(
    {"depth": prof, "learning_rate": tasa, "iterations": iters}
    for prof in (4, 6, 8)
    for tasa, iters in ((0.05, 500), (0.10, 250))
)

# min_data_in_leaf NO va aqui: lo recorre la rejilla. Si estuviera en los dos
# sitios, el diccionario de parametros que se registra en MLflow lo tomaria de
# aqui y quedaria anotado 200 en todas las corridas, cualquiera que fuese el
# valor con el que se entreno de verdad.
LGB_COMUN = {
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
}

# Puntaje de validacion de cada corrida, para poder comparar los modelos entre
# si despues sin reentrenarlos. Lo vuelca main() y lo consume seleccion.py.
PUNTAJES: dict[str, pd.Series] = {}


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


def _corrida(nombre: str, tarea: str, parametros: dict, metricas: dict, registrar=None,
             puntaje: pd.Series | None = None):
    """Registra una corrida. `puntaje` es el orden que el modelo induce sobre
    los segmentos de validacion: se guarda aparte para poder comparar modelos
    con bootstrap sin volver a entrenarlos."""
    if puntaje is not None:
        PUNTAJES[nombre] = puntaje.astype(float)
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
        puntaje=historico,
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
        puntaje=pd.Series(mu_poisson[valida].to_numpy(), index=graves.index),
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
        puntaje=mu_valida,
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
        puntaje=eb,
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
        # predict ya aplica la exponencial. El desplazamiento se multiplica aparte.
        pr = pd.Series(
            arbol.predict(_contiguo(X[valida])) * np.exp(offset[valida].to_numpy()),
            index=graves.index,
        )
        _corrida(
            f"lightgbm_poisson_h{config['num_leaves']}_lr{config['learning_rate']}"
            f"_m{config['min_data_in_leaf']}",
            "conteo",
            {"familia": "LightGBM", "objetivo": "poisson", **LGB_COMUN, **config},
            {
                **captura_por_corte(pr, graves, universo),
                "razon_suma_predicha": round(float(pr.sum() / graves.sum()), 3),
            },
            registrar_lightgbm(arbol),
            puntaje=pr,
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
        {**metricas(prob_logit := np.asarray(logit.predict(X[valida]), dtype=float)),
         "aic": round(float(logit.aic), 1)},
        registrar_statsmodels(logit),
        puntaje=pd.Series(prob_logit, index=graves.index),
    )

    for config in REJILLA_LGB:
        rondas = config["num_boost_round"]
        par = {**LGB_COMUN, **{k: v for k, v in config.items() if k != "num_boost_round"},
               "objective": "binary"}
        ds = lgb.Dataset(_contiguo(X[entrena]), label=_contiguo(y[entrena]),
                         feature_name=list(X.columns))
        arbol = lgb.train(par, ds, num_boost_round=rondas)
        prob = arbol.predict(_contiguo(X[valida]))
        _corrida(
            f"lightgbm_binario_h{config['num_leaves']}_lr{config['learning_rate']}"
            f"_m{config['min_data_in_leaf']}",
            "clasificacion",
            {"familia": "LightGBM", "objetivo": "binary", **LGB_COMUN, **config},
            metricas(prob),
            registrar_lightgbm(arbol),
            puntaje=pd.Series(prob, index=graves.index),
        )

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  catboost no instalado, se omite esa familia")
        return
    Xe = X[entrena].to_numpy(np.float64)
    ye = y[entrena].to_numpy(np.float64)
    Xv = X[valida].to_numpy(np.float64)
    for config in REJILLA_CATBOOST:
        # allow_writing_files: sin esto CatBoost deja un catboost_info/ de logs
        # en la raiz del repositorio cada vez que entrena.
        gato = CatBoostClassifier(
            verbose=0, random_seed=7, allow_writing_files=False, **config
        )
        gato.fit(Xe, ye)
        prob = gato.predict_proba(Xv)[:, 1]
        _corrida(
            f"catboost_d{config['depth']}_lr{config['learning_rate']}",
            "clasificacion",
            {"familia": "CatBoost", **config},
            metricas(prob),
            lambda nombre, m=gato: mlflow.catboost.log_model(m, name=nombre),
            puntaje=pd.Series(prob, index=graves.index),
        )


def exportar(experimento: str, destino: Path) -> int:
    """Vuelca las corridas registradas a un JSON, para citarlas en el reporte.

    Las metricas de la rejilla solo viven en MLflow, y el servidor de la EC2 se
    apaga al terminar la entrega. Esto las deja en el repositorio, ordenadas por
    la metrica principal de cada tarea, para que ninguna cifra se transcriba a
    mano.
    """
    from mlflow.tracking import MlflowClient

    cliente = MlflowClient()
    encontrado = [e for e in cliente.search_experiments() if e.name == experimento]
    if not encontrado:
        print(f"no hay ningun experimento llamado {experimento} en {mlflow.get_tracking_uri()}")
        return 1

    corridas = cliente.search_runs([encontrado[0].experiment_id], max_results=1000)
    ficha = {
        "origen": {
            "tracking_uri": mlflow.get_tracking_uri(),
            "experimento": experimento,
            "corridas": len(corridas),
        }
    }
    for tarea, metrica in METRICA_PRINCIPAL.items():
        de_la_tarea = [c for c in corridas if c.data.tags.get("tarea") == tarea]
        de_la_tarea.sort(key=lambda c: c.data.metrics.get(metrica, -1), reverse=True)
        ficha[tarea] = {
            "ordenado_por": metrica,
            "corridas": [
                {
                    "nombre": c.info.run_name,
                    "familia": c.data.params.get("familia", "linea base"),
                    "estado": c.info.status,
                    "metricas": {k: round(v, 5) for k, v in sorted(c.data.metrics.items())},
                    "parametros": dict(sorted(c.data.params.items())),
                }
                for c in de_la_tarea
            ],
        }

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {destino} con {len(corridas)} corridas")
    for tarea, metrica in METRICA_PRINCIPAL.items():
        mejor = ficha[tarea]["corridas"][0]
        print(f"  mejor en {tarea}: {mejor['nombre']} con {metrica} {mejor['metricas'][metrica]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--vecindad", type=Path, default=VECINDAD)
    ap.add_argument("--tracking-uri", default=None,
                    help="http://<ip>:5000 para registrar contra la EC2. Sin esto, ./mlruns")
    ap.add_argument("--experimento", default=EXPERIMENTO)
    ap.add_argument("--exportar", action="store_true",
                    help="no entrena: vuelca las corridas ya registradas a un JSON")
    ap.add_argument("--salida", type=Path, default=EXPORTACION)
    ap.add_argument("--puntajes", type=Path, default=PUNTAJES_VALIDACION,
                    help="donde volcar el puntaje de validacion de cada corrida")
    a = ap.parse_args()

    if a.tracking_uri:
        mlflow.set_tracking_uri(a.tracking_uri)
    if a.exportar:
        return exportar(a.experimento, a.salida)
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

    # Los puntajes de validacion, para que seleccion.py compare los modelos con
    # bootstrap sin tener que reentrenar la rejilla entera.
    tabla = pd.DataFrame(PUNTAJES)
    tabla.insert(0, "graves_2021", graves.astype(int))
    tabla.index.name = "CIV"
    a.puntajes.parent.mkdir(parents=True, exist_ok=True)
    tabla.reset_index().to_parquet(a.puntajes, index=False)

    print(f"\nlisto. {len(X.columns)} variables, {int(entrena.sum()):,} filas de entrenamiento.")
    print(f"{len(PUNTAJES)} corridas con puntaje volcadas a {a.puntajes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
