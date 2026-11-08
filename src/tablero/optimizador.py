"""What-if y paquete de inversion. Funciones puras: no tocan la API ni Dash.

1. opciones: para cada segmento y cada categoria activa calcula
       reduccion = esperado_eb x |factor|          (graves evitados por ano)
       costo     = costo unitario (x km si la unidad es km)
2. armar_paquete: mochila de eleccion multiple (una intervencion por segmento)
   resuelta con el voraz clasico:
     a. por segmento se dejan solo las opciones eficientes (frontera convexa
        de costo vs reduccion, partiendo de "no intervenir");
     b. cada paso de la frontera es un incremento (mas costo, mas reduccion);
     c. se ordenan todos los incrementos por reduccion/costo y se toman
        mientras quepan. Asi, si sobra plata, un segmento pasa de una obra
        barata a una mas efectiva en vez de quedarse con la primera.
3. resumen: las cifras del panel D.

Limitacion conocida: elegibilidad uniforme (no sabemos que segmentos estan
junto a un colegio, por ejemplo). El usuario puede apagar categorias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNAS = ["intervencion", "clave", "costo", "reduccion", "reduccion_bajo", "reduccion_alto"]


def opciones(
    ranking: pd.DataFrame, factores: list[dict], activas: set[str], costos: dict[str, float]
) -> pd.DataFrame:
    """Tabla larga: una fila por segmento x categoria activa que reduce."""
    eb = ranking.esperado_eb.to_numpy(float)
    km = np.maximum(ranking.long_m.to_numpy(float) / 1000, 0.05)  # piso de 50 m
    partes = []
    for f in factores:
        if f["clave"] not in activas or f["factor"] >= 0:
            continue
        unitario = float(costos.get(f["clave"]) or f["costo_unitario_millones"])
        costo = unitario * km if f["unidad_costo"] == "km" else np.full(len(eb), unitario)
        partes.append(
            pd.DataFrame(
                {
                    "fila": ranking.index,
                    "intervencion": f["categoria"],
                    "clave": f["clave"],
                    "costo": costo,
                    "reduccion": eb * -f["factor"],
                    "reduccion_bajo": eb * -min(f["rango"][1], 0.0),
                    "reduccion_alto": eb * -f["rango"][0],
                }
            )
        )
    if not partes:
        return pd.DataFrame(columns=["fila", *COLUMNAS])
    return pd.concat(partes, ignore_index=True)


def _frontera(candidatos: list[tuple[float, float, int]]) -> list[int]:
    """Indices de las opciones en la frontera convexa, de la mas barata a la mas cara.

    candidatos: (costo, reduccion, indice) de un mismo segmento.
    """
    pila: list[tuple[float, float, int]] = [(0.0, 0.0, -1)]  # parte de no intervenir
    for c, r, i in sorted(candidatos, key=lambda t: (t[0], -t[1])):
        if r <= pila[-1][1]:
            continue  # dominada: cuesta igual o mas y reduce menos
        while len(pila) >= 2:
            (c0, r0, _), (c1, r1, _) = pila[-2], pila[-1]
            if (r1 - r0) * (c - c1) <= (r - r1) * (c1 - c0):  # pendiente no decrece
                pila.pop()
            else:
                break
        pila.append((c, r, i))
    return [i for _, _, i in pila[1:]]


def armar_paquete(ranking: pd.DataFrame, op: pd.DataFrame, presupuesto: float) -> pd.DataFrame:
    """Devuelve el ranking con la intervencion asignada y la columna en_paquete.

    Fuera del paquete, la intervencion mostrada es la recomendada (la de mejor
    reduccion por peso) para que el usuario vea que se haria con mas plata.
    """
    out = ranking.copy()
    for c in COLUMNAS:
        out[c] = None if c in ("intervencion", "clave") else 0.0
    out["en_paquete"] = False
    if op.empty:
        return out

    # Listas simples: con pandas fila a fila el recalculo tardaba ~3 s.
    costo, reduccion = op.costo.tolist(), op.reduccion.tolist()
    por_fila: dict = {}
    for i, fila in enumerate(op.fila.tolist()):
        por_fila.setdefault(fila, []).append((costo[i], reduccion[i], i))

    pasos = []  # (razon, fila, nivel, indice_opcion, delta_costo)
    fronteras: dict = {}
    for fila, cands in por_fila.items():
        f = _frontera(cands)
        fronteras[fila] = f
        c_prev = r_prev = 0.0
        for nivel, i in enumerate(f):
            dc, dr = costo[i] - c_prev, reduccion[i] - r_prev
            pasos.append((dr / dc if dc > 0 else np.inf, fila, nivel, i, dc))
            c_prev, r_prev = costo[i], reduccion[i]
    pasos.sort(key=lambda p: -p[0])

    nivel_actual: dict = {}
    bloqueado: set = set()
    restante = float(presupuesto)
    for _, fila, nivel, i, dc in pasos:
        if fila in bloqueado or nivel != nivel_actual.get(fila, -1) + 1:
            continue
        if dc <= restante + 1e-9:
            restante -= dc
            nivel_actual[fila] = nivel
        else:
            bloqueado.add(fila)  # sus mejoras siguientes tampoco aplican

    filas = list(fronteras)
    elegida = [f[nivel_actual[fila]] if fila in nivel_actual else f[0] for fila, f in fronteras.items()]
    valores = op.iloc[elegida][COLUMNAS].to_numpy()
    out.loc[filas, COLUMNAS] = valores
    out.loc[filas, "en_paquete"] = [fila in nivel_actual for fila in filas]
    out[COLUMNAS[2:]] = out[COLUMNAS[2:]].astype(float)
    return out


def resumen(tabla: pd.DataFrame) -> dict:
    """Cifras del panel D a partir de la tabla que devuelve armar_paquete."""
    paq = tabla[tabla.en_paquete]
    costo = float(paq.costo.sum())
    evitados = float(paq.reduccion.sum())
    riesgo_loc = tabla.groupby("localidad").esperado_eb.sum().sort_values(ascending=False)
    top5 = set(riesgo_loc.head(5).index)
    return {
        "segmentos": int(len(paq)),
        "costo": costo,
        "evitados": evitados,
        "evitados_bajo": float(paq.reduccion_bajo.sum()),
        "evitados_alto": float(paq.reduccion_alto.sum()),
        "costo_por_evitado": costo / evitados if evitados > 0 else None,
        "localidades": int(paq.localidad.nunique()),
        "localidades_total": int(tabla.localidad.nunique()),
        "participacion_top5": 100 * float(paq[paq.localidad.isin(top5)].costo.sum()) / costo
        if costo > 0
        else None,
        "por_intervencion": paq.intervencion.value_counts().to_dict(),
    }
