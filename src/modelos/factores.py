"""Supuesto de reduccion por intervencion, y la evidencia de por que es un supuesto.

El tablero necesita un factor de reduccion para el panel what-if. Este modulo
NO lo estima: lo declara, y deja medida la razon.

POR QUE NO SE ESTIMA
  Se calculo la diferencia en diferencias de dos maneras sobre los mismos datos:

    sin emparejar   compara tratados contra nunca intervenidos. Los tratados
                    partian del doble de graves en 2015, asi que revierten a la
                    media y salen reducciones aparentes.
    emparejado      compara dentro de cada estrato de graves en 2015. Queda la
                    seleccion por exposicion: la ciudad intervino vias mas
                    largas, con mas carriles y de mayor trafico, asi que salen
                    aumentos aparentes.

  El mismo dato da reducciones del 18 al 33 % o aumentos del 44 al 69 % segun
  una decision de modelado. Coincide con el modelo de panel, que controla
  longitud, carriles, clase, tendencia y vecinos: ningun coeficiente post_ es
  significativo. Con obras marcadas todas en 2016 y sin un periodo previo largo,
  el efecto no esta identificado.

QUE HACE EL TABLERO
  Usa un supuesto uniforme, igual para las cinco categorias, que el usuario
  edita. Al no haber evidencia de que una categoria sea mas efectiva que otra,
  el optimizador reparte por costo, que es lo unico que si esta medido.

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
ESTRATO_TOPE = 4
MINIMO_POR_ESTRATO = 20

# Supuesto por defecto del tablero: reduccion del 10 %, con una banda de la
# mitad al doble. Es un supuesto declarado, no una estimacion.
SUPUESTO = 0.10
BANDA = (0.05, 0.20)

CATEGORIAS = {  # clave del panel -> (nombre visible, costo unitario, unidad)
    "senalizacion": ("Señalización", 2.1, "segmento"),
    "kilometros_demarcados": ("Kilómetros demarcados", 8.4, "km"),
    "paso_peatonal": ("Paso peatonal", 14.0, "segmento"),
    "trafico_calmado": ("Tráfico calmado", 31.5, "segmento"),
    "zona_escolar": ("Zona escolar", 22.0, "segmento"),
}


def razon_dd(pre, post, t, c) -> float:
    """DiD en razon, sin emparejar."""
    return (post[t].sum() / pre[t].sum()) / (post[c].sum() / pre[c].sum()) - 1


def razon_dd_emparejada(post, estrato, t, c) -> float:
    """Razon de medias dentro de cada estrato de graves previos, ponderada."""
    num = den = 0.0
    for e in np.unique(estrato[t]):
        it, ic = t & (estrato == e), c & (estrato == e)
        if it.sum() < MINIMO_POR_ESTRATO or ic.sum() < MINIMO_POR_ESTRATO or post[ic].mean() == 0:
            continue
        num += it.sum() * (post[it].mean() / post[ic].mean() - 1)
        den += it.sum()
    return float(num / den) if den else float("nan")


def evidencia(panel: pd.DataFrame) -> list[dict]:
    """Las dos estimaciones por categoria, para declarar la sensibilidad."""
    por_anio = panel.pivot(index="CIV", columns="anio", values="graves")
    pre = por_anio[PRE].sum(axis=1).to_numpy(float)
    post = por_anio[POST].mean(axis=1).to_numpy(float)
    estrato = np.minimum(pre, ESTRATO_TOPE).astype(int)
    marcas = panel.drop_duplicates("CIV").set_index("CIV").loc[por_anio.index]
    columnas = [f"tratado_{c}" for c in CATEGORIAS]
    control = (marcas[columnas].sum(axis=1) == 0).to_numpy()
    otras = marcas[columnas].sum(axis=1)

    salida = []
    for clave in CATEGORIAS:
        t = (marcas[f"tratado_{clave}"] == 1).to_numpy()
        salida.append(
            {
                "clave": clave,
                "did_sin_emparejar": round(razon_dd(pre, post, t, control), 3),
                "did_emparejado": round(razon_dd_emparejada(post, estrato, t, control), 3),
                "segmentos_tratados": int(t.sum()),
                "con_otra_categoria_pct": round(100 * float((otras[t] > 1).mean()), 1),
            }
        )
    return salida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=RAIZ / "data" / "procesado" / "panel.parquet")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data" / "procesado" / "factores.json")
    ap.add_argument("--supuesto", type=float, default=SUPUESTO,
                    help="reduccion supuesta por intervencion, entre 0 y 1")
    a = ap.parse_args()

    medido = {e["clave"]: e for e in evidencia(pd.read_parquet(a.panel))}
    factores = []
    for clave, (nombre, costo, unidad) in CATEGORIAS.items():
        factores.append(
            {
                "clave": clave,
                "categoria": nombre,
                "factor": -a.supuesto,
                "rango": [-BANDA[1], -BANDA[0]],
                "segmentos_tratados": medido[clave]["segmentos_tratados"],
                "con_otra_categoria_pct": medido[clave]["con_otra_categoria_pct"],
                "costo_unitario_millones": costo,
                "unidad_costo": unidad,
            }
        )

    ficha = {
        "origen_del_factor": "supuesto declarado, no estimado",
        "supuesto": a.supuesto,
        "banda": list(BANDA),
        "advertencia": "La reducción es un supuesto del usuario, no un efecto medido. Con estos "
        "datos el efecto de las intervenciones no está identificado: la misma diferencia en "
        "diferencias da reducciones o aumentos según se empareje o no por el nivel previo. Los "
        "costos unitarios también son supuestos.",
        "evidencia_de_no_identificacion": list(medido.values()),
        "factores": factores,
    }
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {a.salida} con un supuesto de {a.supuesto:.0%}")
    for e in medido.values():
        print(f"   {e['clave']:24s} DiD sin emparejar {e['did_sin_emparejar']:+.3f}"
              f"   emparejado {e['did_emparejado']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
