"""Predicciones del modelo elegido, cargado desde MLflow.

Escribe predicciones.parquet con el contrato que consume el tablero,
asi que la API y el tablero no cambian, pero las cifras ya salen de los
modelos de la rejilla y no del predictor de grupos de referencia.

ENTRA : reportes/modelos/seleccion.json  que corrida se eligio por tarea
        los modelos serializados en MLflow
        data/procesado/panel.parquet y vecindad.parquet
SALE  : data/procesado/predicciones.parquet
        data/procesado/predicciones_metricas.json

CONTRATO, el de siempre (una fila por CIV x anio_ref):
    CIV, anio_ref, esperado, esperado_eb, prob_grave, modelo

DOS ANIOS
  2021 se predice con historia hasta 2020 y se puede validar, porque lo
  observado existe. 2022 es el pronostico que usa el tablero y se predice con
  historia hasta 2021. Sus filas no estan en el panel, que termina en 2021, y
  se construyen aqui copiando los atributos fijos del segmento y recalculando
  la historia un anio mas adelante.

EMPIRICAL BAYES
  El modelo de conteo da el esperado. La correccion de Hauer se aplica encima
  como posproceso, con la dispersion estimada sobre los anios de
  entrenamiento, igual que en conteo.py.

    python -m src.modelos.predecir
"""

from __future__ import annotations

import lightgbm as lgb  # noqa: I001  debe ir antes que pandas/pyarrow

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.modelos.caracteristicas import (
    ANIO_FIN_ENTRENA,
    ANIO_INICIO,
    ANIO_VALIDA,
    PANEL,
    VECINDAD,
    matriz_diseno,
    preparar,
)
from src.modelos.conteo import empirical_bayes, estimar_alpha
from src.modelos.metricas import captura_top_alineada, empates_en_el_corte

RAIZ = Path(__file__).resolve().parents[2]
SELECCION = RAIZ / "reportes" / "modelos" / "seleccion.json"
SALIDA = RAIZ / "data" / "procesado"
TRACKING = "sqlite:///mlflow.db"
EXPERIMENTO = "seguridad-vial-bogota-e3"
ANIO_PRONOSTICO = ANIO_VALIDA + 1
CORTE = 0.05

CARACTERES_PROHIBIDOS = ("/", ":", ".", "%", '"', "'")


def nombre_en_mlflow(corrida: str) -> str:
    """MLflow guarda el modelo con el nombre saneado de la corrida."""
    for malo in CARACTERES_PROHIBIDOS:
        corrida = corrida.replace(malo, "_")
    return corrida


def cargar_modelo(cliente, experimento_id: str, corrida: str):
    """Trae de MLflow el modelo LightGBM que produjo esa corrida."""
    buscado = nombre_en_mlflow(corrida)
    for m in cliente.search_logged_models([experimento_id]):
        if m.name == buscado:
            return mlflow.lightgbm.load_model(f"models:/{m.model_id}")
    raise SystemExit(f"no esta en MLflow el modelo '{buscado}' de la corrida '{corrida}'")


def filas_de_pronostico(panel: pd.DataFrame) -> pd.DataFrame:
    """Construye el anio que todavia no ocurrio, a partir del ultimo del panel.

    Los atributos del segmento no cambian de un anio al siguiente. Lo que hay
    que mover es la historia: el acumulado previo pasa a incluir el ultimo anio
    observado, y preparar() deriva de ahi el rezago y la media por anio.
    """
    ultimo = panel[panel.anio == ANIO_VALIDA].copy()
    acumulado = panel.groupby("CIV").graves.sum()
    ultimo["anio"] = ANIO_PRONOSTICO
    ultimo["graves"] = 0  # no observado: no se usa para predecir
    ultimo["graves_acum_previo"] = ultimo.CIV.map(acumulado).astype(int)
    return pd.concat([panel, ultimo], ignore_index=True)


def predecir(modelo, X: pd.DataFrame, offset: pd.Series) -> np.ndarray:
    """LightGBM con objetivo poisson ya aplica la exponencial al predecir, asi
    que el desplazamiento se multiplica aparte."""
    bruto = modelo.predict(np.ascontiguousarray(X.to_numpy(np.float64)))
    return bruto * np.exp(offset.to_numpy())


def factor_empirical_bayes(p: pd.DataFrame, esperado: pd.Series, y: pd.Series) -> pd.Series:
    """Correccion de Hauer por segmento, estimada con los anios de
    entrenamiento, que son los unicos con observado comparable.

    Es un parametro ajustado del modelo, no una prediccion, asi que viaja con
    el modelo empaquetado y la API solo lo multiplica.
    """
    entrena = p.anio.between(ANIO_INICIO, ANIO_FIN_ENTRENA).to_numpy()
    alpha = estimar_alpha(y[entrena].to_numpy(), esperado[entrena].to_numpy())
    mu_hist = esperado[entrena].groupby(p.CIV[entrena]).sum()
    obs_hist = y[entrena].groupby(p.CIV[entrena]).sum()
    return empirical_bayes(mu_hist, obs_hist, alpha) / mu_hist, alpha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--vecindad", type=Path, default=VECINDAD)
    ap.add_argument("--seleccion", type=Path, default=SELECCION)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--tracking-uri", default=TRACKING)
    ap.add_argument("--experimento", default=EXPERIMENTO)
    a = ap.parse_args()

    eleccion = json.loads(a.seleccion.read_text(encoding="utf-8"))
    corrida_conteo = eleccion["conteo"]["seleccion"]["elegida"]
    corrida_clase = eleccion["clasificacion"]["seleccion"]["elegida"]

    mlflow.set_tracking_uri(a.tracking_uri)
    cliente = mlflow.MlflowClient()
    encontrado = [e for e in cliente.search_experiments() if e.name == a.experimento]
    if not encontrado:
        raise SystemExit(f"no hay experimento '{a.experimento}' en {a.tracking_uri}")
    exp_id = encontrado[0].experiment_id
    modelo_conteo = cargar_modelo(cliente, exp_id, corrida_conteo)
    modelo_clase = cargar_modelo(cliente, exp_id, corrida_clase)

    panel = pd.read_parquet(a.panel)
    aristas = pd.read_parquet(a.vecindad) if a.vecindad.exists() else None
    p = preparar(filas_de_pronostico(panel), aristas)
    X = matriz_diseno(p)
    offset = p.log_long_km.astype(float)
    y = p.graves.astype(float)

    esperado = pd.Series(predecir(modelo_conteo, X, offset), index=p.index)
    prob = pd.Series(
        modelo_clase.predict(np.ascontiguousarray(X.to_numpy(np.float64))), index=p.index
    )

    factor, alpha = factor_empirical_bayes(p, esperado, y)

    salidas = []
    for anio in (ANIO_VALIDA, ANIO_PRONOSTICO):
        sel = (p.anio == anio).to_numpy()
        fila = pd.DataFrame(
            {
                "CIV": p.CIV[sel].to_numpy(),
                "anio_ref": anio,
                "esperado": esperado[sel].to_numpy(),
                "prob_grave": prob[sel].to_numpy(),
            }
        )
        fila["esperado_eb"] = fila.esperado * fila.CIV.map(factor).to_numpy()
        fila["modelo"] = corrida_conteo
        salidas.append(fila)

    out = pd.concat(salidas, ignore_index=True)[
        ["CIV", "anio_ref", "esperado", "esperado_eb", "prob_grave", "modelo"]
    ]
    a.salida.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.salida / "predicciones.parquet", index=False)

    # Metricas de validacion sobre 2021, que es el unico anio con observado.
    real = pd.Series(y[(p.anio == ANIO_VALIDA).to_numpy()].to_numpy(), index=out[out.anio_ref == ANIO_VALIDA].CIV)
    universo = int(real.index.nunique())
    k = max(1, int(round(universo * CORTE)))
    v = out[out.anio_ref == ANIO_VALIDA].set_index("CIV")
    historico = panel[panel.anio <= ANIO_FIN_ENTRENA].groupby("CIV").graves.sum()
    metricas = {
        "modelo_conteo": corrida_conteo,
        "modelo_clasificacion": corrida_clase,
        "origen": "MLflow",
        "alpha_nb2": round(alpha, 4),
        f"captura_top5_{ANIO_VALIDA}": {
            "esperado": round(captura_top_alineada(v.esperado, real, k), 2),
            "esperado_eb": round(captura_top_alineada(v.esperado_eb, real, k), 2),
            "prob_grave": round(captura_top_alineada(v.prob_grave, real, k), 2),
            "linea_base_historico": round(captura_top_alineada(historico, real, k), 2),
        },
        f"mae_{ANIO_VALIDA}": round(float((v.esperado_eb - real).abs().mean()), 4),
        f"razon_suma_{ANIO_VALIDA}": round(float(v.esperado.sum() / real.sum()), 3),
        # La linea base son conteos enteros y empata en masa justo en el corte.
        # Se declara para que el reporte pueda decir por que la cifra se calcula
        # por la parte proporcional del grupo empatado y no ordenando y cortando.
        "empates_linea_base_en_el_corte": empates_en_el_corte(
            historico.reindex(real.index).fillna(0.0), k
        ),
        "universo_civ": universo,
        "corte_pct": round(100 * CORTE, 1),
    }
    (a.salida / "predicciones_metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
