"""Grafo de vecindad de la malla vial: que segmentos se tocan.

ENTRA : data/raw/malla_vial_integral.parquet
SALE  : data/procesado/vecindad.parquet, una fila por par (CIV, vecino)

POR QUE
  El riesgo de un tramo depende de la interseccion en la que esta, no solo de
  su propia historia. Sin esto el modelo mira cada segmento aislado.

COMO
  Cada polilinea viene en EPSG:3116, o sea en metros. Se toman los dos
  extremos de cada trazo y se redondean al metro: ese par de enteros es el
  nodo. Dos CIV son vecinos si comparten un nodo.

  No hace falta geopandas: no es un cruce entre capas sino la topologia de una
  sola, y el redondeo al metro hace de tolerancia de union.

  El grafo es simetrico y sin lazos: si A es vecino de B, B lo es de A, y
  ningun CIV es vecino de si mismo.

    python -m src.modelos.vecindad
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PRECISION_NODO = 0  # decimales al redondear la coordenada: 0 = al metro


def nodos_por_civ(malla: pd.DataFrame) -> dict[tuple[int, int], set[int]]:
    """Nodo -> conjunto de CIV que lo tocan. El CIV 0 no es llave y se ignora."""
    nodos: dict[tuple[int, int], set[int]] = defaultdict(set)
    con_llave = malla[malla.MVICIV != 0]
    for civ, geometria in zip(con_llave.MVICIV.astype("int64"), con_llave.geometry):
        if not geometria:
            continue
        for trazo in json.loads(geometria).get("paths", []):
            if not trazo:
                continue
            for x, y in (trazo[0], trazo[-1]):
                nodos[(round(x, PRECISION_NODO), round(y, PRECISION_NODO))].add(civ)
    return nodos


def aristas(nodos: dict[tuple[int, int], set[int]]) -> pd.DataFrame:
    """Lista de pares (CIV, vecino). Un nodo con un solo CIV no genera arista."""
    vecinos: dict[int, set[int]] = defaultdict(set)
    for civs in nodos.values():
        if len(civs) < 2:
            continue
        for civ in civs:
            vecinos[civ] |= civs
    pares = [(civ, v) for civ, vs in vecinos.items() for v in vs if v != civ]
    tabla = pd.DataFrame(pares, columns=["CIV", "vecino"])
    # int32 basta para el rango de los CIV. No reduce el parquet, que ya
    # codifica los enteros de forma compacta, pero deja el tipo correcto.
    return tabla.astype("int32").sort_values(["CIV", "vecino"], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=RAIZ / "data" / "raw")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data" / "procesado")
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)

    malla = pd.read_parquet(a.entrada / "malla_vial_integral.parquet")
    tabla = aristas(nodos_por_civ(malla))
    tabla.to_parquet(a.salida / "vecindad.parquet", index=False)

    grado = tabla.groupby("CIV").size()
    universo = int(malla[malla.MVICIV != 0].MVICIV.nunique())
    print(f"CIV con al menos un vecino: {len(grado):,} de {universo:,}")
    print(f"aristas: {len(tabla):,}  |  grado medio {grado.mean():.2f}  "
          f"mediana {int(grado.median())}  maximo {int(grado.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
