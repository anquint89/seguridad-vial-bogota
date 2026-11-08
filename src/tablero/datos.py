"""Prepara las tablas que sirve la API y consume el tablero.

ENTRA : data/raw (malla, siniestros), data/procesado/panel.parquet,
        data/procesado/predicciones.parquet (contrato en src/modelos/provisional.py)
SALE  : data/procesado/tablero_segmentos.parquet  una fila por CIV x anio_ref
        data/procesado/tablero_geometria.parquet  una fila por CIV, lon/lat

    python -m src.tablero.datos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

RAIZ = Path(__file__).resolve().parents[2]
TOLERANCIA_M = 15.0  # Douglas-Peucker, en metros (la malla viene en EPSG:3116)
VENTANAS = {2021: range(2015, 2021), 2022: range(2015, 2022)}

# Los dos primeros digitos del CIV son el codigo de la localidad (1 a 20).
# Prefijos 30, 40 y 50 son malla arterial: se usa la localidad de sus siniestros.
LOCALIDADES = {
    1: "USAQUEN", 2: "CHAPINERO", 3: "SANTA FE", 4: "SAN CRISTOBAL", 5: "USME",
    6: "TUNJUELITO", 7: "BOSA", 8: "KENNEDY", 9: "FONTIBON", 10: "ENGATIVA",
    11: "SUBA", 12: "BARRIOS UNIDOS", 13: "TEUSAQUILLO", 14: "LOS MARTIRES",
    15: "ANTONIO NARINO", 16: "PUENTE ARANDA", 17: "CANDELARIA",
    18: "RAFAEL URIBE URIBE", 19: "CIUDAD BOLIVAR", 20: "SUMAPAZ",
}
TIPOS_VIA = {"AK": "AK", "AC": "AC", "CL": "CL", "KR": "KR", "TV": "TV", "DG": "DG"}


def douglas_peucker(puntos: np.ndarray, tol: float) -> np.ndarray:
    """Simplifica una polilinea conservando los extremos."""
    if len(puntos) < 3:
        return puntos
    inicio, fin = puntos[0], puntos[-1]
    linea = fin - inicio
    norma = np.hypot(*linea)
    if norma == 0:
        dist = np.hypot(*(puntos - inicio).T)
    else:
        rel = puntos - inicio
        dist = np.abs(linea[0] * rel[:, 1] - linea[1] * rel[:, 0]) / norma
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return np.vstack([inicio, fin])
    izq = douglas_peucker(puntos[: i + 1], tol)
    der = douglas_peucker(puntos[i:], tol)
    return np.vstack([izq[:-1], der])


def geometria_lonlat(malla: pd.DataFrame) -> pd.DataFrame:
    """Une los trazos de cada CIV, simplifica y reproyecta a WGS84."""
    a_wgs84 = Transformer.from_crs(3116, 4326, always_xy=True)
    trazos: dict[int, list] = {}
    for civ, geom in zip(malla.MVICIV, malla.geometry):
        if not geom:
            continue
        for trazo in json.loads(geom).get("paths", []):
            xy = douglas_peucker(np.asarray(trazo, dtype=float), TOLERANCIA_M)
            lon, lat = a_wgs84.transform(xy[:, 0], xy[:, 1])
            trazos.setdefault(int(civ), []).append(
                [[round(x, 6), round(y, 6)] for x, y in zip(lon, lat)]
            )
    return pd.DataFrame(
        {"CIV": list(trazos), "trazos": [json.dumps(t) for t in trazos.values()]}
    )


def nombre_via(malla: pd.DataFrame) -> pd.Series:
    nombre = malla.MVINOMBRE.fillna("").str.strip()
    tipo = malla.MVITIPO.fillna("").str.strip()
    sin_nombre = nombre.isin(["", "SIN_NMG"])
    via = np.where(sin_nombre, tipo.map(TIPOS_VIA).fillna("Vía") + " sin nombre", nombre)
    return pd.Series(via, index=malla.index).groupby(malla.MVICIV).first()


def localidad_por_civ(civs: pd.Index, siniestros: pd.DataFrame) -> pd.Series:
    sin = siniestros[(siniestros.CIV > 0) & (siniestros.LOCALIDAD.str.strip() != "")]
    moda = sin.groupby("CIV").LOCALIDAD.agg(lambda s: s.value_counts().index[0])
    por_codigo = pd.Series(civs // 1_000_000, index=civs).map(LOCALIDADES)
    return moda.reindex(civs).fillna(por_codigo).fillna("MALLA ARTERIAL")


def construir(raw: Path, procesado: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    malla = pd.read_parquet(raw / "malla_vial_integral.parquet")
    malla = malla[malla.MVICIV != 0]
    siniestros = pd.read_parquet(raw / "historico_siniestros.parquet", columns=["CIV", "LOCALIDAD"])
    panel = pd.read_parquet(procesado / "panel.parquet")
    pred = pd.read_parquet(procesado / "predicciones.parquet")

    seg = panel.drop_duplicates("CIV").set_index("CIV")
    tratado = [c for c in seg.columns if c.startswith("tratado_")]
    atributos = pd.DataFrame(
        {
            "via": nombre_via(malla),
            "localidad": localidad_por_civ(seg.index, siniestros),
            "long_m": seg.long_m.round(1),
            "carriles": seg.carriles,
            "clase_via": seg.clase_via,
            "vel_reg": seg.vel_reg,
            "intervenido_2016": seg[tratado].sum(axis=1).gt(0),
        }
    ).rename_axis("CIV")

    por_anio = panel.pivot(index="CIV", columns="anio", values="graves")
    filas = []
    for anio_ref, ventana in VENTANAS.items():
        p = pred[pred.anio_ref == anio_ref].set_index("CIV")
        bloque = atributos.join(p[["esperado", "esperado_eb", "prob_grave", "modelo"]], how="inner")
        bloque["graves_previos"] = por_anio[list(ventana)].sum(axis=1)
        bloque["observado"] = por_anio[anio_ref] if anio_ref in por_anio else np.nan
        bloque["anio_ref"] = anio_ref
        bloque["rango"] = bloque.esperado_eb.rank(ascending=False, method="first").astype(int)
        filas.append(bloque.reset_index())
    segmentos = pd.concat(filas, ignore_index=True).sort_values(["anio_ref", "rango"])
    return segmentos, geometria_lonlat(malla)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAIZ / "data" / "raw")
    ap.add_argument("--procesado", type=Path, default=RAIZ / "data" / "procesado")
    a = ap.parse_args()

    segmentos, geometria = construir(a.raw, a.procesado)
    segmentos.to_parquet(a.procesado / "tablero_segmentos.parquet", index=False)
    geometria.to_parquet(a.procesado / "tablero_geometria.parquet", index=False)
    print(f"segmentos: {len(segmentos):,} filas  |  geometrias: {len(geometria):,} CIV")
    print(segmentos.head(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
