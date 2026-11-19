"""Selecciona el modelo de cada tarea y mide si la diferencia es defendible.

ENTRA : data/procesado/puntajes_validacion.parquet  (lo escribe experimentos.py)
SALE  : reportes/modelos/seleccion.json

Las primeras corridas de la rejilla caben en tres decimas de punto, asi que
quedarse con la primera sin medir la incertidumbre es elegir al azar.

EL CRITERIO
  1. Captura en el top 5 % sobre 2021, la misma metrica de la linea base.
  2. Intervalo del 95 % de la diferencia entre la lider y cada rival. Las que
     contienen el cero son indistinguibles.
  3. Entre las indistinguibles, la de menos hojas: si la diferencia no es
     defendible, el modelo pequeno es mas estable y mas barato de servir.
  4. Una corrida por tarea, porque predicciones.parquet necesita un valor
     esperado, que da el conteo, y una probabilidad, que da la ocurrencia.

El bootstrap es PAREADO: cada replica evalua todas las corridas sobre el mismo
remuestreo. Comparar intervalos sueltos exagera las diferencias, porque ignora
que los modelos fallan en los mismos segmentos.

Se calcula tambien el AUC de las corridas de conteo, que no lo llevaban. Mide
si el orden separa los segmentos con grave, mientras la captura mide cuanto
dano se concentra arriba, asi que un modelo de conteo puede ganar en una y
perder en la otra.

    python -m src.modelos.seleccion
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
PUNTAJES = RAIZ / "data" / "procesado" / "puntajes_validacion.parquet"
SALIDA = RAIZ / "reportes" / "modelos" / "seleccion.json"

PERCENTILES = (0.01, 0.02, 0.05, 0.10)
CORTE_PRINCIPAL = 0.05
REPLICAS = 500
SEMILLA = 11
# Rivales que entran al bootstrap: las que quedan a menos de esto de la lider.
MARGEN_FINALISTA = 1.5

NO_SON_CORRIDAS = ("CIV", "graves_2021")
LINEA_BASE = "linea_base_conteo_historico"

TAREA_POR_PREFIJO = (
    ("linea_base", "conteo"),
    ("poisson_glm", "conteo"),
    ("binomial_negativa", "conteo"),
    ("lightgbm_poisson", "conteo"),
    ("regresion_logistica", "clasificacion"),
    ("lightgbm_binario", "clasificacion"),
    ("catboost", "clasificacion"),
)


def tarea_de(nombre: str) -> str:
    for prefijo, tarea in TAREA_POR_PREFIJO:
        if nombre.startswith(prefijo):
            return tarea
    return "desconocida"


def complejidad_de(nombre: str) -> int:
    """Numero de hojas o profundidad, leido del nombre de la corrida. Sirve
    para desempatar entre modelos indistinguibles. Los GLM no tienen, y se les
    da la complejidad mas baja porque son los mas simples de todos."""
    hojas = re.search(r"_h(\d+)", nombre)
    if hojas:
        return int(hojas.group(1))
    prof = re.search(r"_d(\d+)", nombre)
    if prof:
        return 2 ** int(prof.group(1))  # un arbol de profundidad d tiene hasta 2^d hojas
    return 0


def captura_en(puntaje: np.ndarray, graves: np.ndarray, k: int) -> float:
    """Porcentaje de los graves observados que cae en los k mejores del orden.

    Usa argpartition en vez de ordenar entero: con quinientas replicas por
    sesenta corridas la diferencia es de minutos. Los empates se rompen de
    forma arbitraria, igual que en la funcion del perfilamiento.
    """
    total = graves.sum()
    if total <= 0:
        return float("nan")
    top = np.argpartition(-puntaje, k)[:k]
    return 100.0 * graves[top].sum() / total


def metricas_puntuales(datos: pd.DataFrame, corridas: list[str]) -> dict:
    graves = datos.graves_2021.to_numpy()
    ocurre = (graves >= 1).astype(int)
    n = len(datos)
    ficha = {}
    for nombre in corridas:
        puntaje = datos[nombre].to_numpy(float)
        fila = {
            f"captura_top_{int(q * 100)}_pct": round(
                captura_en(puntaje, graves, int(round(n * q))), 2
            )
            for q in PERCENTILES
        }
        # El AUC se calcula tambien para las corridas de conteo, que no lo
        # traian: es lo que permite leer la brecha entre las dos metricas.
        fila["auc_roc"] = round(float(roc_auc_score(ocurre, puntaje)), 4)
        fila["tarea"] = tarea_de(nombre)
        fila["complejidad"] = complejidad_de(nombre)
        ficha[nombre] = fila
    return ficha


def bootstrap(datos: pd.DataFrame, corridas: list[str], replicas: int, semilla: int) -> dict:
    """Captura en el corte principal de cada corrida, sobre las mismas replicas."""
    rng = np.random.default_rng(semilla)
    graves = datos.graves_2021.to_numpy()
    puntajes = {c: datos[c].to_numpy(float) for c in corridas}
    n = len(datos)
    k = int(round(n * CORTE_PRINCIPAL))

    salida = {c: np.empty(replicas) for c in corridas}
    for b in range(replicas):
        idx = rng.integers(0, n, n)
        g = graves[idx]
        for c in corridas:
            salida[c][b] = captura_en(puntajes[c][idx], g, k)
    return salida


def _intervalo(v: np.ndarray) -> dict:
    return {
        "media": round(float(np.mean(v)), 3),
        "ic95_bajo": round(float(np.percentile(v, 2.5)), 3),
        "ic95_alto": round(float(np.percentile(v, 97.5)), 3),
    }


def comparar(muestras: dict, lider: str) -> dict:
    """Intervalo de la diferencia lider menos rival, replica a replica."""
    comparacion = {}
    for nombre, v in muestras.items():
        dif = muestras[lider] - v
        ic = _intervalo(dif)
        comparacion[nombre] = {
            "diferencia_con_lider": ic,
            "indistinguible": bool(ic["ic95_bajo"] <= 0.0 <= ic["ic95_alto"]),
            "replicas_en_que_gana_al_lider": round(float(np.mean(dif < 0)), 3),
        }
    return comparacion


def seleccionar(puntuales: dict, muestras: dict, comparacion: dict, lider: str) -> dict:
    """Entre las indistinguibles de la lider, la mas simple. Si varias empatan
    tambien en complejidad, la de mejor metrica: el orden alfabetico no es un
    criterio defendible."""
    clave = f"captura_top_{int(CORTE_PRINCIPAL * 100)}_pct"
    empatadas = [n for n, c in comparacion.items() if c["indistinguible"] and n != LINEA_BASE]
    if not empatadas:
        empatadas = [lider]
    elegida = min(empatadas, key=lambda n: (puntuales[n]["complejidad"], -puntuales[n][clave]))
    return {
        "lider_por_metrica": lider,
        "indistinguibles_de_la_lider": sorted(empatadas),
        "elegida": elegida,
        "por_que": (
            f"{len(empatadas)} corridas son indistinguibles de {lider} al 95 %. "
            f"Entre ellas se toma la de menor complejidad, que es {elegida}."
        )
        if len(empatadas) > 1
        else f"{lider} es la unica que no queda empatada dentro del intervalo.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puntajes", type=Path, default=PUNTAJES)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--replicas", type=int, default=REPLICAS)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    a = ap.parse_args()

    datos = pd.read_parquet(a.puntajes)
    corridas = [c for c in datos.columns if c not in NO_SON_CORRIDAS]
    puntuales = metricas_puntuales(datos, corridas)
    clave = f"captura_top_{int(CORTE_PRINCIPAL * 100)}_pct"

    ficha = {
        "criterio": {
            "metrica_principal": clave,
            "anio_de_validacion": 2021,
            "replicas_bootstrap": a.replicas,
            "semilla": a.semilla,
            "regla": (
                "Se toma la corrida lider por la metrica principal. Por bootstrap pareado se "
                "calcula el intervalo del 95 % de la diferencia con cada rival. Las que "
                "contienen el cero son indistinguibles de la lider, y entre esas se elige la "
                "de menor complejidad."
            ),
        },
        "universo_civ": int(len(datos)),
        "graves_2021": int(datos.graves_2021.sum()),
        "corridas_evaluadas": len(corridas),
        "metricas": puntuales,
    }

    for tarea in ("conteo", "clasificacion"):
        de_la_tarea = [c for c in corridas if puntuales[c]["tarea"] == tarea]
        if not de_la_tarea:
            continue
        lider = max(de_la_tarea, key=lambda n: puntuales[n][clave])
        tope = puntuales[lider][clave]
        finalistas = [c for c in de_la_tarea if puntuales[c][clave] >= tope - MARGEN_FINALISTA]
        if LINEA_BASE in corridas and LINEA_BASE not in finalistas:
            finalistas.append(LINEA_BASE)  # la referencia entra siempre

        muestras = bootstrap(datos, finalistas, a.replicas, a.semilla)
        comparacion = comparar(muestras, lider)
        ficha[tarea] = {
            "finalistas": {n: _intervalo(v) for n, v in muestras.items()},
            "comparacion_con_lider": comparacion,
            "seleccion": seleccionar(puntuales, muestras, comparacion, lider),
        }

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(corridas)} corridas, {a.replicas} replicas de bootstrap\n")
    for tarea in ("conteo", "clasificacion"):
        if tarea not in ficha:
            continue
        s = ficha[tarea]["seleccion"]
        emp = len(s["indistinguibles_de_la_lider"])
        print(f"{tarea}:")
        print(f"  lider      {s['lider_por_metrica']}  ({puntuales[s['lider_por_metrica']][clave]} %)")
        print(f"  empatadas  {emp}")
        print(f"  ELEGIDA    {s['elegida']}  ({puntuales[s['elegida']][clave]} %, "
              f"AUC {puntuales[s['elegida']]['auc_roc']})")
    print(f"\nescrito {a.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
