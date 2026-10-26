"""Perfilamiento de las cuatro capas y linea base del metodo vigente.

Produce reportes/perfilamiento.json con todas las cifras que se citan en el
reporte de la Entrega 1, y las dos figuras que lo acompanan. Todo numero del
documento sale de aqui: si el servicio cambia, se vuelve a correr y las cifras
se actualizan solas.

    python -m src.exploracion.perfilamiento --entrada data/raw
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PERCENTILES = (0.01, 0.02, 0.05, 0.10)


def longitud_proyectada(geometria_json: str | None) -> float:
    """Longitud en metros de una polilinea ya entregada en EPSG:3116."""
    if not geometria_json:
        return float("nan")
    geometria = json.loads(geometria_json)
    total = 0.0
    for trazo in geometria.get("paths", []):
        for inicio, fin in zip(trazo[:-1], trazo[1:]):
            total += math.hypot(fin[0] - inicio[0], fin[1] - inicio[1])
    return total


def captura(ranking: pd.Series, validacion: pd.Series, k: int) -> tuple[int, int, float]:
    """Siniestros graves del ano de validacion que caen en los k primeros."""
    seleccionados = set(ranking.sort_values(ascending=False).head(k).index)
    capturados = int(validacion[validacion.index.isin(seleccionados)].sum())
    total = int(validacion.sum())
    return capturados, total, 100 * capturados / total

def puente_formulario(victimas: pd.DataFrame, siniestros: pd.DataFrame) -> dict:
    """Cruza las victimas de 2021 con los siniestros por FORMULARIO"""
    victimas_2021 = victimas[victimas.ANO_OCURRENCIA_ACC == 2021]
    siniestros_2021 = siniestros[siniestros.ANO_OCURRENCIA_ACC == 2021]
    puente = siniestros_2021.drop_duplicates("FORMULARIO").set_index("FORMULARIO").CIV
    cruzan = victimas_2021.FORMULARIO.isin(puente.index)
    civ_asignado = victimas_2021.FORMULARIO.map(puente).fillna(0)
    return {"victimas_2021": int(len(victimas_2021)),
        "cruzan_con_siniestros": int(cruzan.sum()),
        "cruzan_pct": round(100 * cruzan.mean(), 1),
        "con_civ_valido": int((civ_asignado != 0).sum()),
        "con_civ_valido_pct": round(100 * (civ_asignado != 0).mean(), 1),
        "formularios_duplicados_en_siniestros": int(siniestros_2021.FORMULARIO.duplicated().sum())}


def main(argumentos: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--entrada", type=Path, default=RAIZ / "data" / "raw")
    analizador.add_argument("--salida", type=Path, default=RAIZ / "reportes")
    opciones = analizador.parse_args(argumentos)
    entrada, salida = opciones.entrada, opciones.salida
    (salida / "figuras").mkdir(parents=True, exist_ok=True)

    malla = pd.read_parquet(entrada / "malla_vial_integral.parquet")
    siniestros = pd.read_parquet(entrada / "historico_siniestros.parquet")
    victimas = pd.read_parquet(entrada / "victimas_2021_2025.parquet")
    intervenciones = pd.read_parquet(entrada / "intervenciones_seguridad_vial.parquet")

    ficha: dict = {}

    # --- Malla vial -------------------------------------------------------
    malla["long_m"] = malla["geometry"].map(longitud_proyectada)
    con_civ = malla[malla.MVICIV != 0]
    universo = sorted(con_civ.MVICIV.unique())
    velocidad_vacia = malla.MVIVELREG.fillna("").astype(str).str.strip().eq("")
    ficha["malla"] = {
        "registros": len(malla),
        "civ_cero": int((malla.MVICIV == 0).sum()),
        "civ_distintos": len(universo),
        "civ_con_mas_de_un_registro": int((con_civ.MVICIV.value_counts() > 1).sum()),
        "maximo_registros_por_civ": int(con_civ.MVICIV.value_counts().max()),
        "velocidad_reglamentada_vacia": int(velocidad_vacia.sum()),
        "velocidad_reglamentada_vacia_pct": round(100 * velocidad_vacia.mean(), 1),
        "clase_via_nulos": int(malla.MVITCLA.isna().sum()),
        "carriles_nulos": int(malla.MVINUMC.isna().sum()),
        "longitud_total_km": round(malla.long_m.sum() / 1000, 1),
        "longitud_mediana_m_por_civ": round(
            con_civ.groupby("MVICIV").long_m.sum().median(), 1
        ),
        "segmentos_menores_a_1_m": int((malla.long_m < 1).sum()),
    }
    universo_civ = set(universo)

    # --- Siniestros -------------------------------------------------------
    siniestros["grave"] = siniestros.GRAVEDAD.ne("SOLO DANOS")
    sin_llave = siniestros.CIV.eq(0)
    fuera_de_malla = ~siniestros.CIV.isin(universo_civ) & ~sin_llave
    ficha["siniestros"] = {
        "registros": len(siniestros),
        "graves": int(siniestros.grave.sum()),
        "con_muertos": int(siniestros.GRAVEDAD.eq("CON MUERTOS").sum()),
        "sin_civ": int(sin_llave.sum()),
        "sin_civ_pct": round(100 * sin_llave.mean(), 2),
        "civ_fuera_de_la_malla_vigente": int(fuera_de_malla.sum()),
        "civ_fuera_de_la_malla_vigente_pct": round(100 * fuera_de_malla.mean(), 2),
        "por_anio": siniestros.groupby("ANO_OCURRENCIA_ACC")
        .grave.agg(["size", "sum"])
        .rename(columns={"size": "total", "sum": "graves"})
        .astype(int)
        .to_dict("index"),
        "por_clase": siniestros.CLASE_ACC.value_counts().to_dict(),
    }

    graves = siniestros[siniestros.grave & siniestros.CIV.isin(universo_civ)]
    panel = graves.groupby(["CIV", "ANO_OCURRENCIA_ACC"]).size().unstack(fill_value=0)
    panel = panel.reindex(columns=range(2015, 2022), fill_value=0)

    conteo_total = graves.groupby("CIV").size().sort_values(ascending=False)
    vector = np.zeros(len(universo))
    vector[: len(conteo_total)] = conteo_total.to_numpy()
    media = vector.mean()
    ficha["distribucion_respuesta"] = {
        "graves_asignables_a_un_civ": int(len(graves)),
        "civ_sin_ningun_grave": int(len(universo) - len(conteo_total)),
        "civ_sin_ningun_grave_pct": round(
            100 * (len(universo) - len(conteo_total)) / len(universo), 1
        ),
        "media_por_civ": round(media, 3),
        "varianza_por_civ": round(float(vector.var()), 3),
        "indice_de_dispersion": round(float(vector.var()) / media, 2),
        "maximo_en_un_civ": int(vector.max()),
        "concentracion": {
            f"top_{int(p * 100)}_pct": round(
                100 * vector[: int(round(len(universo) * p))].sum() / vector.sum(), 1
            )
            for p in PERCENTILES
        },
    }

    # --- Linea base del metodo vigente -----------------------------------
    k = int(round(len(universo) * 0.05))
    historico = panel.loc[:, 2015:2020].sum(axis=1)
    validacion = panel[2021]
    capturados, total, porcentaje = captura(historico, validacion, k)
    historico_2019 = panel.loc[:, 2015:2019].sum(axis=1)
    cap_2020, tot_2020, pct_2020 = captura(historico_2019, panel[2020], k)
    ficha["linea_base"] = {
        "universo": len(universo),
        "tamano_top_5_pct": k,
        "entrena_2015_2020_valida_2021": {
            "capturados": capturados,
            "total": total,
            "pct": round(porcentaje, 1),
        },
        "entrena_2015_2019_valida_2020": {
            "capturados": cap_2020,
            "total": tot_2020,
            "pct": round(pct_2020, 1),
        },
    }

    # --- Intervenciones ---------------------------------------------------
    intervenciones["categoria_norm"] = (
        intervenciones.CATEGORIA.str.strip().str.lower().str.replace("  ", " ")
    )
    con_llave = intervenciones[intervenciones.CIV.ne(0)]
    en_malla = con_llave[con_llave.CIV.isin(universo_civ)]
    por_civ = en_malla.groupby("CIV").categoria_norm.nunique()
    ficha["intervenciones"] = {
        "registros": len(intervenciones),
        "sin_civ": int(intervenciones.CIV.eq(0).sum()),
        "categorias_crudas": int(intervenciones.CATEGORIA.nunique()),
        "categorias_normalizadas": int(intervenciones.categoria_norm.nunique()),
        "registros_por_categoria": intervenciones.categoria_norm.value_counts().to_dict(),
        "civ_tratados": int(en_malla.CIV.nunique()),
        "civ_con_mas_de_una_categoria": int((por_civ > 1).sum()),
        "fuera_de_la_malla_vigente": int(len(con_llave) - len(en_malla)),
        "fecha_fin_minima": str(intervenciones.FECHA_FIN_.min().date()),
        "fecha_fin_maxima": str(intervenciones.FECHA_FIN_.max().date()),
        "registros_por_anio": intervenciones.ANIO.value_counts().sort_index().to_dict(),
    }

    tratados = set(en_malla.CIV.unique())
    marca = panel.index.isin(tratados)
    series = panel.groupby(marca).sum()
    previo_t, previo_c = series.loc[True, 2015], series.loc[False, 2015]
    post_t = series.loc[True, [2018, 2019, 2020, 2021]].mean()
    post_c = series.loc[False, [2018, 2019, 2020, 2021]].mean()
    ficha["comparacion_preliminar"] = {
        "nota": (
            "Diferencia en diferencias cruda, sin exposicion, sin covariables y "
            "sin emparejamiento. Solo orienta el diseno causal."
        ),
        "tratados_2015": int(previo_t),
        "tratados_promedio_2018_2021": round(float(post_t), 1),
        "tratados_variacion_pct": round(100 * (post_t / previo_t - 1), 1),
        "control_2015": int(previo_c),
        "control_promedio_2018_2021": round(float(post_c), 1),
        "control_variacion_pct": round(100 * (post_c / previo_c - 1), 1),
        "diferencia_en_diferencias_pp": round(
            100 * (post_t / previo_t - 1) - 100 * (post_c / previo_c - 1), 1
        ),
    }

    # --- Victimas y puente FORMULARIO -------------------------------------

    edad_implausible = int(((victimas.EDAD > 100) | (victimas.EDAD == 0)).sum())

    edad_valid = victimas.EDAD.where(victimas.EDAD.between(1,100))
    victimas["grupo_edad"] = pd.cut(edad_valid, bins=[0,17,29,59,100], labels=["0-17", "18-29", "30-59", "60+"])
    ficha["victimas"] = {
        "registros": len(victimas),
        "fallecidos": int(victimas.TIPO_VICTIMA.eq("Fallecido").sum()),
        "por_condicion": victimas.CONDICION.value_counts().to_dict(),
        "por_localidad": victimas.LOCALIDAD.fillna("SIN DATO").value_counts().to_dict(),
        "por_grupo_edad": victimas.grupo_edad.value_counts(sort=False).to_dict(),
        "condicion_por_grupo_edad": pd.crosstab(victimas.CONDICION, victimas.grupo_edad).to_dict("index"),
        "por_anio": victimas.ANO_OCURRENCIA_ACC.value_counts().sort_index().to_dict(),
        "edad_nula": int(victimas.EDAD.isna().sum()),
        "edad_implausible": edad_implausible,
        "puente_formulario_2021": puente_formulario(victimas, siniestros)
    }

    (salida / "perfilamiento.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _figura_concentracion(vector, salida / "figuras" / "concentracion.png")
    _figura_tendencia(series, salida / "figuras" / "tendencia_tratados.png")
    print(json.dumps(ficha, indent=2, ensure_ascii=False))
    return 0


def _figura_concentracion(vector: np.ndarray, destino: Path) -> None:
    acumulado = np.cumsum(vector) / vector.sum() * 100
    eje_x = np.arange(1, len(vector) + 1) / len(vector) * 100
    figura, ejes = plt.subplots(figsize=(5.4, 3.2))
    ejes.plot(eje_x, acumulado, color="black", linewidth=1.2)
    ejes.plot([0, 100], [0, 100], color="grey", linewidth=0.8, linestyle="--")
    for porcentaje in (1, 5, 10):
        posicion = int(len(vector) * porcentaje / 100) - 1
        ejes.plot([porcentaje], [acumulado[posicion]], marker="o", color="black", markersize=3)
        ejes.annotate(
            f"{porcentaje} %: " + f"{acumulado[posicion]:.1f} %".replace(".", ","),
            (porcentaje, acumulado[posicion]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
        )
    ejes.set_xlabel("Segmentos ordenados por siniestros graves (%)", fontsize=8)
    ejes.set_ylabel("Siniestros graves acumulados (%)", fontsize=8)
    ejes.tick_params(labelsize=8)
    ejes.set_xlim(0, 100)
    ejes.set_ylim(0, 100)
    for lado in ("top", "right"):
        ejes.spines[lado].set_visible(False)
    figura.tight_layout()
    figura.savefig(destino, dpi=200)
    plt.close(figura)


def _figura_tendencia(series: pd.DataFrame, destino: Path) -> None:
    anios = list(range(2015, 2022))
    figura, ejes = plt.subplots(figsize=(5.4, 3.2))
    for marca, etiqueta, estilo in (
        (True, "Segmentos intervenidos (7.722)", "-"),
        (False, "Resto de la malla", "--"),
    ):
        valores = series.loc[marca, anios]
        ejes.plot(
            anios,
            valores / valores.loc[2015] * 100,
            estilo,
            color="black",
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=etiqueta,
        )
    ejes.axvspan(2016, 2017, color="0.9")
    ejes.annotate("ejecución", (2016.5, 72), ha="center", fontsize=8, color="0.35")
    ejes.set_ylabel("Siniestros graves, 2015 = 100", fontsize=8)
    ejes.tick_params(labelsize=8)
    ejes.legend(fontsize=8, frameon=False)
    for lado in ("top", "right"):
        ejes.spines[lado].set_visible(False)
    figura.tight_layout()
    figura.savefig(destino, dpi=200)
    plt.close(figura)


if __name__ == "__main__":
    raise SystemExit(main())
