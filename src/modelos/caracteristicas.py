"""Capa de caracteristicas compartida por todos los modelos.

Un solo sitio construye la matriz de diseno. Si el modelo de conteo y el
clasificador la construyeran cada uno por su lado, la comparacion entre ellos
dejaria de ser de forma funcional y pasaria a ser de que vio cada uno.

ENTRA : data/procesado/panel.parquet  (una fila por CIV x anio)
        data/procesado/vecindad.parquet  (pares CIV, vecino)

VENTANA
  Entrena 2016-2020, valida 2021. Tiene que empezar en 2016: todas las obras
  utilizables estan marcadas ese anio, asi que desde 2017 las columnas post_ y
  tratado_ son identicas y el efecto de la intervencion deja de estar
  identificado. 2015 queda fuera porque el rezago lo deja vacio.

HISTORIA DE VENTANA COMPARABLE
  No se usa el acumulado crudo: su ventana crece con el anio, un anio de
  historia en 2016 y cinco en 2021, asi que el coeficiente mezcla nivel de
  riesgo con tiempo transcurrido y se dispara. Se usa la media por anio
  previo, que significa lo mismo en toda fila.

NADA MIRA EL FUTURO
  Toda variable de historia, propia o de los vecinos, se calcula con anios
  ESTRICTAMENTE anteriores al que se predice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RAIZ = Path(__file__).resolve().parents[2]
PANEL = RAIZ / "data" / "procesado" / "panel.parquet"
VECINDAD = RAIZ / "data" / "procesado" / "vecindad.parquet"

ANIO_INICIO = 2016
ANIO_FIN_ENTRENA = 2020
ANIO_VALIDA = 2021
ANIO_CENTRO = 2019
ANIO_PANDEMIA = 2020
CLASE_REFERENCIA = 3


def _agregados_de_vecinos(p: pd.DataFrame, aristas: pd.DataFrame) -> pd.DataFrame:
    """Para cada (CIV, anio), el promedio de la historia de sus vecinos.

    Se toma la historia YA calculada del vecino, que solo usa anios previos,
    de modo que la variable no puede filtrar el anio que se predice.
    """
    del_vecino = p[["CIV", "anio", "hist_media", "rezago1"]].rename(
        columns={"CIV": "vecino", "hist_media": "v_hist", "rezago1": "v_rezago1"}
    )
    cruce = aristas.merge(del_vecino, on="vecino", how="inner")
    return cruce.groupby(["CIV", "anio"]).agg(
        vecinos_hist=("v_hist", "mean"),
        vecinos_rezago1=("v_rezago1", "mean"),
        vecinos_n=("vecino", "size"),
    )


def preparar(panel: pd.DataFrame, aristas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Anade las derivadas que necesitan los modelos. No toca nada en disco.

    Sin `aristas` las variables de vecindad quedan en cero, que es lo que hace
    falta para probar el resto sin construir el grafo.
    """
    p = panel.sort_values(["CIV", "anio"]).reset_index(drop=True)
    p["n_prev"] = (p.anio - 2015).astype(float)
    p["hist_media"] = p.graves_acum_previo / p.n_prev.clip(lower=1)
    p["hist_media_log"] = np.log1p(p.hist_media)
    p["rezago1"] = p.groupby("CIV").graves.shift(1).fillna(0.0)
    p["rezago1_log"] = np.log1p(p.rezago1)
    p["log_km"] = p.log_long_km

    if aristas is not None and len(aristas):
        agg = _agregados_de_vecinos(p, aristas)
        unido = p.set_index(["CIV", "anio"]).join(agg, how="left").reset_index()
        p = unido.sort_values(["CIV", "anio"]).reset_index(drop=True)
    for col in ("vecinos_hist", "vecinos_rezago1", "vecinos_n"):
        p[col] = p[col].fillna(0.0).astype(float) if col in p else 0.0
    p["vecinos_hist_log"] = np.log1p(p.vecinos_hist)
    p["vecinos_rezago1_log"] = np.log1p(p.vecinos_rezago1)
    return p


def matriz_diseno(p: pd.DataFrame) -> pd.DataFrame:
    """Matriz sobre el panel completo, para que las indicadoras de clase de via
    sean las mismas en entrenamiento y en validacion."""
    post = sorted(c for c in p.columns if c.startswith("post_"))
    tratado = sorted(c for c in p.columns if c.startswith("tratado_"))

    X = pd.DataFrame(index=p.index)
    X["hist_media_log"] = p.hist_media_log.astype(float)
    X["rezago1_log"] = p.rezago1_log.astype(float)
    X["vecinos_hist_log"] = p.vecinos_hist_log.astype(float)
    X["vecinos_rezago1_log"] = p.vecinos_rezago1_log.astype(float)
    X["vecinos_n"] = p.vecinos_n.astype(float)
    # n_prev NO entra: es anio-2015 y tendencia es anio-2019, asi que difieren
    # en una constante y la matriz queda deficiente de rango contra el intercepto.
    X["log_km"] = p.log_km.astype(float)
    X["carriles"] = p.carriles.astype(float)
    X["vel_reg"] = p.vel_reg.fillna(p.vel_reg.median()).astype(float)
    X["vel_faltante"] = p.vel_faltante.astype(float)
    X["tendencia"] = (p.anio - ANIO_CENTRO).astype(float)
    X["pandemia"] = (p.anio == ANIO_PANDEMIA).astype(float)
    for col in post + tratado:
        X[col] = p[col].astype(float)

    clase = pd.get_dummies(p.clase_via.astype(int), prefix="clase", dtype=float)
    clase = clase.drop(columns=[f"clase_{CLASE_REFERENCIA}"], errors="ignore")
    return sm.add_constant(pd.concat([X, clase], axis=1), has_constant="add")


def particion(p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mascaras de entrenamiento y validacion, iguales para todos los modelos."""
    entrena = p.anio.between(ANIO_INICIO, ANIO_FIN_ENTRENA).to_numpy()
    valida = (p.anio == ANIO_VALIDA).to_numpy()
    return entrena, valida


def cargar(panel: Path = PANEL, vecindad: Path | None = VECINDAD) -> pd.DataFrame:
    """Lee el panel y el grafo de disco y devuelve el panel ya preparado."""
    aristas = pd.read_parquet(vecindad) if vecindad and Path(vecindad).exists() else None
    return preparar(pd.read_parquet(panel), aristas)
