"""API que sirve las inferencias al tablero.

Lee las tablas que produce `python -m src.tablero.datos` y los factores de
`python -m src.modelos.factores`. La carpeta se cambia con la variable de
entorno DATOS_DIR (en Docker apunta al volumen montado).

    uvicorn src.api.main:app --port 8000
    documentacion interactiva: http://localhost:8000/docs

Endpoints (los de la maqueta, mas tres de apoyo):
    GET /salud                      estado y modelo cargado
    GET /localidades                lista para el filtro
    GET /ranking?anio&top&localidad top-N sin geometria (tabla B y optimizador D)
    GET /segmentos/top?anio&n&localidad&bbox   el mismo top-N como GeoJSON (mapa A)
    GET /segmento/{civ}?anio        detalle de un segmento
    GET /factores                   factores del what-if (panel C)
    GET /metricas                   desempeno del modelo en validacion
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

RAIZ = Path(__file__).resolve().parents[2]
TODAS = "TODAS"

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


def filtrar(anio: int, localidad: str) -> pd.DataFrame:
    seg = tablas()["segmentos"]
    sel = seg[seg.anio_ref == anio]
    if sel.empty:
        raise HTTPException(404, f"No hay predicciones para el año {anio}")
    if localidad.upper() != TODAS:
        sel = sel[sel.localidad == localidad.upper()]
    return sel.sort_values("rango")


@app.get("/salud")
def salud() -> dict:
    seg = tablas()["segmentos"]
    return {
        "estado": "ok",
        "modelo": seg.modelo.iloc[0],
        "anios": sorted(int(a) for a in seg.anio_ref.unique()),
        "segmentos": int(seg.CIV.nunique()),
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
    for fila in registros(sel[["CIV", "via", "localidad", "esperado_eb", "rango"]]):
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
    detalle = registros(fila)[0]
    detalle["trazos"] = json.loads(tablas()["geometria"].get(civ, "[]"))
    return detalle


@app.get("/factores")
def factores() -> dict:
    return tablas()["factores"]


@app.get("/metricas")
def metricas() -> dict:
    return tablas()["metricas"]
