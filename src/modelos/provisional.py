"""Predictor provisional: Empirical Bayes por grupos de referencia.

Sirve para que el tablero tenga predicciones reales mientras los modelos de
MLflow no esten listos. Cuando el modelo de conteo quede registrado, su etapa
escribe el MISMO archivo con las MISMAS columnas y el tablero no cambia.

CONTRATO de data/procesado/predicciones.parquet (una fila por CIV x anio_ref):
    CIV              int64    segmento
    anio_ref         int64    ano que se predice (2021 = validacion, 2022 = pronostico)
    esperado         float    graves/ano segun el modelo (sin corregir)
    esperado_eb      float    graves/ano corregidos por Empirical Bayes
    prob_grave       float    P(al menos un grave en anio_ref)
    modelo           str      nombre y version del modelo que produjo la fila

METODO (Hauer, grupos de referencia)
  1. Grupo = clase de via x decil de longitud x carriles (1, 2, 3+).
  2. Prior: mu = graves/ano promedio del grupo en los anos de entrenamiento.
  3. Sobredispersion phi por momentos sobre los totales del periodo.
  4. Posterior Gamma(a = phi + y, b = phi/mu + T): esperado_eb = a / b y
     P(Y = 0) = (b / (b + 1)) ** a, la predictiva binomial negativa.

    python -m src.modelos.provisional
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
NOMBRE = "eb_grupos_referencia_v1"
# anio_ref -> anos con los que se entrena
VENTANAS = {2021: range(2015, 2021), 2022: range(2015, 2022)}


def grupos_de_referencia(seg: pd.DataFrame) -> pd.Series:
    """Clave de grupo por segmento: clase x decil de longitud x carriles."""
    decil = pd.qcut(seg.long_m.rank(method="first"), 10, labels=False)
    carriles = seg.carriles.clip(upper=3)
    return seg.clase_via.astype(str) + "_" + decil.astype(str) + "_" + carriles.astype(str)


def empirical_bayes(y_total: pd.Series, grupo: pd.Series, anios: int) -> pd.DataFrame:
    """Prior por grupo, phi por momentos y posterior Gamma por segmento."""
    mu_anual = y_total.groupby(grupo).transform("mean") / anios
    mu_total = mu_anual * anios
    exceso = ((y_total - mu_total) ** 2 - y_total).sum()
    phi = float((mu_total**2).sum() / exceso) if exceso > 0 else 1e6
    a = phi + y_total
    b = phi / mu_anual.clip(lower=1e-6) + anios
    return pd.DataFrame(
        {
            "esperado": mu_anual,
            "esperado_eb": a / b,
            "prob_grave": 1 - (b / (b + 1)) ** a,
        }
    ).assign(phi=phi)


def captura_top(pred: pd.Series, real: pd.Series, fraccion: float = 0.05) -> float:
    """% de los graves reales que caen en el top del ranking predicho."""
    k = int(round(len(pred) * fraccion))
    top = pred.sort_values(ascending=False).head(k).index
    return round(100 * real.loc[top].sum() / real.sum(), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=RAIZ / "data" / "procesado" / "panel.parquet")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data" / "procesado")
    a = ap.parse_args()

    panel = pd.read_parquet(a.panel)
    seg = panel.drop_duplicates("CIV").set_index("CIV")[["long_m", "clase_via", "carriles"]]
    grupo = grupos_de_referencia(seg)
    por_anio = panel.pivot(index="CIV", columns="anio", values="graves")

    salidas, metricas = [], {"modelo": NOMBRE}
    for anio_ref, ventana in VENTANAS.items():
        y_total = por_anio[list(ventana)].sum(axis=1)
        pred = empirical_bayes(y_total, grupo, len(ventana))
        metricas[f"phi_{anio_ref}"] = round(float(pred.phi.iloc[0]), 3)
        if anio_ref in por_anio.columns:  # hay observado: se valida
            real = por_anio[anio_ref]
            metricas[f"captura_top5_{anio_ref}"] = {
                "esperado": captura_top(pred.esperado, real),
                "esperado_eb": captura_top(pred.esperado_eb, real),
                "linea_base_historico": captura_top(y_total.astype(float), real),
            }
            metricas[f"mae_{anio_ref}"] = round(float((pred.esperado_eb - real).abs().mean()), 4)
        salidas.append(
            pred.drop(columns="phi").reset_index().assign(anio_ref=anio_ref, modelo=NOMBRE)
        )

    out = pd.concat(salidas, ignore_index=True)[
        ["CIV", "anio_ref", "esperado", "esperado_eb", "prob_grave", "modelo"]
    ]
    a.salida.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.salida / "predicciones.parquet", index=False)
    (a.salida / "predicciones_metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
