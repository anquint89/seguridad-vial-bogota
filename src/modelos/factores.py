"""Factores de reduccion por categoria de intervencion (version provisional).

Diferencia en diferencias en razon, por categoria:

    factor = (post_T / pre_T) / (post_C / pre_C) - 1

  T  = segmentos con esa categoria (pueden tener otras: se reporta el traslape)
  C  = segmentos que nunca recibieron ninguna intervencion
  pre  = 2015, el unico ano completo antes de las obras de 2016
  post = promedio 2018-2021 (2016-2017 es la ventana de ejecucion y se excluye)

El intervalo del 90 % sale de un bootstrap por segmentos. NO es un efecto
causal cerrado (un solo periodo previo, tratamientos multiples): en el tablero
entra como parametro del what-if, con su intervalo.

Los costos unitarios son SUPUESTOS ilustrativos (millones COP por segmento;
kilometros demarcados por km). El tablero deja editarlos.

    python -m src.modelos.factores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PRE, POST = [2015], [2018, 2019, 2020, 2021]
REPETICIONES = 500

CATEGORIAS = {  # clave del panel -> (nombre visible, costo unitario, unidad)
    "senalizacion": ("Señalización", 2.1, "segmento"),
    "kilometros_demarcados": ("Kilómetros demarcados", 8.4, "km"),
    "paso_peatonal": ("Paso peatonal", 14.0, "segmento"),
    "trafico_calmado": ("Tráfico calmado", 31.5, "segmento"),
    "zona_escolar": ("Zona escolar", 22.0, "segmento"),
}


def razon_dd(pre_t, post_t, pre_c, post_c) -> float:
    return (post_t / pre_t) / (post_c / pre_c) - 1


def estimar(panel: pd.DataFrame, semilla: int = 2026) -> list[dict]:
    por_anio = panel.pivot(index="CIV", columns="anio", values="graves")
    pre = por_anio[PRE].sum(axis=1).to_numpy(float)
    post = por_anio[POST].mean(axis=1).to_numpy(float)
    marcas = panel.drop_duplicates("CIV").set_index("CIV").loc[por_anio.index]
    columnas = [f"tratado_{c}" for c in CATEGORIAS]
    control = (marcas[columnas].sum(axis=1) == 0).to_numpy()
    otras = marcas[columnas].sum(axis=1)

    rng = np.random.default_rng(semilla)
    idx_c = np.flatnonzero(control)
    resultado = []
    for clave, (nombre, costo, unidad) in CATEGORIAS.items():
        tratado = (marcas[f"tratado_{clave}"] == 1).to_numpy()
        idx_t = np.flatnonzero(tratado)
        punto = razon_dd(pre[idx_t].sum(), post[idx_t].sum(), pre[idx_c].sum(), post[idx_c].sum())
        boot = []
        for _ in range(REPETICIONES):
            bt = rng.choice(idx_t, len(idx_t))
            bc = rng.choice(idx_c, len(idx_c))
            boot.append(razon_dd(pre[bt].sum(), post[bt].sum(), pre[bc].sum(), post[bc].sum()))
        bajo, alto = np.percentile(boot, [5, 95])
        resultado.append(
            {
                "clave": clave,
                "categoria": nombre,
                "factor": round(float(punto), 3),
                "ic90": [round(float(bajo), 3), round(float(alto), 3)],
                "segmentos_tratados": int(tratado.sum()),
                "con_otra_categoria_pct": round(100 * float((otras[tratado] > 1).mean()), 1),
                "costo_unitario_millones": costo,
                "unidad_costo": unidad,
            }
        )
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=RAIZ / "data" / "procesado" / "panel.parquet")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data" / "procesado" / "factores.json")
    a = ap.parse_args()

    factores = estimar(pd.read_parquet(a.panel))
    ficha = {
        "metodo": "DiD en razon, pre 2015 vs post 2018-2021, control = nunca intervenidos, "
        f"IC 90 % por bootstrap de segmentos ({REPETICIONES} repeticiones)",
        "advertencia": "Provisional y no causal: un solo periodo previo y tratamientos multiples. "
        "Los costos unitarios son supuestos.",
        "factores": factores,
    }
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ficha, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
