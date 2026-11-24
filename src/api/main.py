"""API que sirve las inferencias al tablero.

Las predicciones las calcula en cada peticion con el paquete `modelo-riesgo`,
que empaqueta `python -m src.modelos.empaquetar` sacando de MLflow las dos
corridas que elige reportes/modelos/seleccion.json. De las tablas que produce
`python -m src.tablero.datos` toma lo descriptivo, que no se predice, y de
`python -m src.modelos.factores` los factores del what-if. La carpeta de datos
se cambia con la variable de entorno DATOS_DIR.

    uvicorn src.api.main:app --port 8000
    documentacion interactiva: http://localhost:8000/docs

Endpoints del tablero (los de la maqueta, mas tres de apoyo):
    GET /salud                      estado, modelos cargados y modo de inferencia
    GET /localidades                lista para el filtro
    GET /ranking?anio&top&localidad top-N sin geometria (tabla B y optimizador D)
    GET /segmentos/top?anio&n&localidad&bbox   el mismo top-N como GeoJSON (mapa A)
    GET /segmento/{civ}?anio        detalle de un segmento
    GET /factores                   factores del what-if (panel C)
    GET /metricas                   desempeno de los modelos en validacion

Endpoints del modelo, para usarlo sin pasar por el tablero:
    GET  /api/v1/health             version de la API y del paquete del modelo
    POST /api/v1/predict            predice sobre filas de caracteristicas
"""

from __future__ import annotations

# inferencia trae lightgbm, y lightgbm tiene que cargar ANTES que pandas: si
# pyarrow registra primero su runtime de OpenMP, la DLL nativa revienta con una
# violacion de acceso al predecir.
from src.api import inferencia  # noqa: I001

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

RAIZ = Path(__file__).resolve().parents[2]
TODAS = "TODAS"


def version_del_paquete() -> str | None:
    """La version de modelo-riesgo instalada, o None si no esta."""
    try:
        from modelo_riesgo import __version__

        return __version__
    except Exception:
        return None

app = FastAPI(
    title="Seguridad vial Bogotá · API de priorización",
    description="Riesgo esperado de siniestros graves por segmento CIV y factores de intervención.",
    version="0.2.0",
)


def carpeta() -> Path:
    return Path(os.environ.get("DATOS_DIR", RAIZ / "data" / "procesado"))


@lru_cache(maxsize=1)
def tablas() -> dict:
    """Se cargan una vez por proceso: unos 30 MB en memoria."""
    d = carpeta()
    return {
        "segmentos": pd.read_parquet(d / "tablero_segmentos.parquet"),
        "geometria": pd.read_parquet(d / "tablero_geometria.parquet").set_index("CIV").trazos,
        "factores": json.loads((d / "factores.json").read_text(encoding="utf-8")),
        "metricas": json.loads((d / "predicciones_metricas.json").read_text(encoding="utf-8")),
    }


def registros(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista de dicts con NaN convertido en null."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


PREDICHAS = ["esperado", "esperado_eb", "prob_grave"]


def filtrar(anio: int, localidad: str) -> pd.DataFrame:
    """Atributos del segmento desde la tabla, predicciones desde el modelo.

    La tabla aporta lo descriptivo, que no se predice: via, localidad,
    longitud, carriles. Las tres columnas predichas se calculan en esta
    peticion con el modelo empaquetado, y el ranking se rehace con ellas.
    """
    seg = tablas()["segmentos"]
    sel = seg[seg.anio_ref == anio]
    if sel.empty:
        raise HTTPException(404, f"No hay predicciones para el año {anio}")
    if localidad.upper() != TODAS:
        sel = sel[sel.localidad == localidad.upper()]

    if inferencia.disponible():
        vivo = inferencia.predecir(anio, sel.CIV.tolist())
        if not vivo.empty:
            sel = sel.drop(columns=PREDICHAS).merge(vivo, on="CIV", how="inner")
            sel = sel.sort_values("esperado", ascending=False)
            sel["rango"] = range(1, len(sel) + 1)
            return sel
    return sel.sort_values("rango")


@app.get("/salud")
def salud() -> dict:
    seg = tablas()["segmentos"]
    return {
        "estado": "ok",
        "modelo": seg.modelo.iloc[0],
        "anios": sorted(int(a) for a in seg.anio_ref.unique()),
        "segmentos": int(seg.CIV.nunique()),
        **inferencia.ficha(),
    }


@app.get("/localidades")
def localidades() -> list[str]:
    return sorted(tablas()["segmentos"].localidad.unique().tolist())


@app.get("/ranking")
def ranking(
    anio: int = 2021,
    top: int = Query(2000, ge=1, le=20000),
    localidad: str = TODAS,
) -> list[dict]:
    return registros(filtrar(anio, localidad).head(top))


@app.get("/segmentos/top")
def segmentos_top(
    anio: int = 2021,
    n: int = Query(2000, ge=1, le=20000),
    localidad: str = TODAS,
    bbox: str | None = Query(None, description="lon_min,lat_min,lon_max,lat_max"),
) -> dict:
    sel = filtrar(anio, localidad).head(n)
    geom = tablas()["geometria"]
    caja = [float(v) for v in bbox.split(",")] if bbox else None
    features = []
    for fila in registros(sel[["CIV", "via", "localidad", "esperado", "rango"]]):
        trazos = json.loads(geom.get(fila["CIV"], "[]"))
        if caja and not any(
            caja[0] <= lon <= caja[2] and caja[1] <= lat <= caja[3]
            for t in trazos
            for lon, lat in t
        ):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": trazos},
                "properties": fila,
            }
        )
    return {"type": "FeatureCollection", "features": features}


@app.get("/segmento/{civ}")
def segmento(civ: int, anio: int = 2021) -> dict:
    seg = tablas()["segmentos"]
    fila = seg[(seg.CIV == civ) & (seg.anio_ref == anio)]
    if fila.empty:
        raise HTTPException(404, f"CIV {civ} no encontrado para {anio}")
    if inferencia.disponible():
        vivo = inferencia.predecir(anio, [civ])
        if not vivo.empty:
            fila = fila.drop(columns=PREDICHAS).merge(vivo, on="CIV", how="inner")
    detalle = registros(fila)[0]
    detalle["trazos"] = json.loads(tablas()["geometria"].get(civ, "[]"))
    return detalle


@app.get("/factores")
def factores() -> dict:
    return tablas()["factores"]


@app.get("/metricas")
def metricas() -> dict:
    return tablas()["metricas"]


# --------------------------------------------------------------------------
# El modelo servido por si mismo, para quien quiera usarlo sin el tablero.


class EntradaPrediccion(BaseModel):
    inputs: list[dict]


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "name": app.title,
        "api_version": app.version,
        "model_version": version_del_paquete(),
    }


@app.post("/api/v1/predict")
def predict(entrada: EntradaPrediccion) -> dict:
    """Predice sobre filas de caracteristicas, sin pasar por el tablero.

    Cada fila lleva las variables del modelo y el desplazamiento. `factor_eb`
    y `CIV` son opcionales: si vienen, se devuelven el esperado corregido y la
    llave del segmento.
    """
    if not inferencia.disponible():
        raise HTTPException(503, "el paquete del modelo no esta disponible")
    from modelo_riesgo.predict import make_prediction

    resultado = make_prediction(input_data=pd.DataFrame(entrada.inputs))
    if resultado["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(resultado["errors"]))
    return resultado
