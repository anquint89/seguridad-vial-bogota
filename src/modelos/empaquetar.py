"""Deja los modelos elegidos dentro del paquete instalable modelo-riesgo.

Saca de MLflow las dos corridas que elige reportes/modelos/seleccion.json y
las escribe en modelo-riesgo/modelo_riesgo/trained/, junto con el config.yml
que describe la entrada y una muestra de prueba. De ahi sale la rueda que
instala la imagen de la API, de modo que el contenedor no necesita mlflow ni
alcanzar el servidor de seguimiento, que se apaga al cerrar la entrega.

El vinculo con la corrida no se pierde: el nombre, el experimento y las
metricas de validacion de cada modelo quedan escritos en el config.yml.

ENTRA : reportes/modelos/seleccion.json y los modelos en MLflow
        data/procesado/panel.parquet y vecindad.parquet
SALE  : modelo-riesgo/modelo_riesgo/trained/<paquete>-conteo<version>.txt
        modelo-riesgo/modelo_riesgo/trained/<paquete>-clasificacion<version>.txt
        modelo-riesgo/modelo_riesgo/config.yml
        modelo-riesgo/modelo_riesgo/datasets/muestra_2021.csv
        data/procesado/caracteristicas_inferencia.parquet

    python -m src.modelos.empaquetar
"""

from __future__ import annotations

import lightgbm as lgb  # noqa: I001  debe ir antes que pandas/pyarrow

import argparse
import json
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.modelos.caracteristicas import ANIO_VALIDA, PANEL, VECINDAD, matriz_diseno, preparar
from src.modelos.predecir import (
    ANIO_PRONOSTICO,
    cargar_modelo,
    factor_empirical_bayes,
    filas_de_pronostico,
    predecir,
)

RAIZ = Path(__file__).resolve().parents[2]
SELECCION = RAIZ / "reportes" / "modelos" / "seleccion.json"
PAQUETE = RAIZ / "modelo-riesgo" / "modelo_riesgo"
CARACTERISTICAS = RAIZ / "data" / "procesado" / "caracteristicas_inferencia.parquet"
TRACKING = "sqlite:///mlflow.db"
EXPERIMENTO = "seguridad-vial-bogota-e3"
NOMBRE_PAQUETE = "modelo-riesgo"
MUESTRA = 200
SEMILLA_MUESTRA = 7

# Columnas que viajan con las caracteristicas pero no entran al modelo: la
# llave, el anio, el desplazamiento y el factor de Empirical Bayes, que es un
# parametro ajustado y no una variable.
NO_SON_VARIABLES = ("CIV", "anio_ref", "log_long_km", "factor_eb")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--vecindad", type=Path, default=VECINDAD)
    ap.add_argument("--seleccion", type=Path, default=SELECCION)
    ap.add_argument("--paquete", type=Path, default=PAQUETE)
    ap.add_argument("--caracteristicas", type=Path, default=CARACTERISTICAS)
    ap.add_argument("--tracking-uri", default=TRACKING)
    ap.add_argument("--experimento", default=EXPERIMENTO)
    a = ap.parse_args()

    version = (a.paquete / "VERSION").read_text(encoding="utf-8").strip()
    eleccion = json.loads(a.seleccion.read_text(encoding="utf-8"))
    corridas = {
        "conteo": eleccion["conteo"]["seleccion"]["elegida"],
        "clasificacion": eleccion["clasificacion"]["seleccion"]["elegida"],
    }

    mlflow.set_tracking_uri(a.tracking_uri)
    cliente = mlflow.MlflowClient()
    encontrado = [e for e in cliente.search_experiments() if e.name == a.experimento]
    if not encontrado:
        raise SystemExit(f"no hay experimento '{a.experimento}' en {a.tracking_uri}")
    exp_id = encontrado[0].experiment_id

    entrenados = a.paquete / "trained"
    entrenados.mkdir(parents=True, exist_ok=True)
    (a.paquete / "datasets").mkdir(parents=True, exist_ok=True)

    modelos: dict[str, dict] = {}
    for tarea, corrida in corridas.items():
        modelo = cargar_modelo(cliente, exp_id, corrida)
        archivo = entrenados / f"{NOMBRE_PAQUETE}-{tarea}{version}.txt"
        modelo.save_model(str(archivo))
        modelos[tarea] = {
            "archivo": archivo.name,
            "corrida": corrida,
            "arboles": modelo.num_trees(),
            "variables": modelo.num_feature(),
            "metricas_validacion": eleccion["metricas"][corrida],
        }
        print(f"  {tarea:14s} {corrida}  ->  {archivo.name}  ({modelo.num_trees()} arboles)")

    # La matriz que la API necesita para inferir, solo de los anios que el
    # tablero muestra. El desplazamiento viaja como una columna mas.
    panel = pd.read_parquet(a.panel)
    aristas = pd.read_parquet(a.vecindad) if a.vecindad.exists() else None
    p = preparar(filas_de_pronostico(panel), aristas)
    X = matriz_diseno(p)

    conteo = lgb.Booster(model_file=str(entrenados / modelos["conteo"]["archivo"]))
    esperado = pd.Series(predecir(conteo, X, p.log_long_km.astype(float)), index=p.index)
    factor, alpha = factor_empirical_bayes(p, esperado, p.graves.astype(float))

    sel = p.anio.isin((ANIO_VALIDA, ANIO_PRONOSTICO)).to_numpy()
    fuera = X[sel].astype("float32")
    fuera.insert(0, "anio_ref", p.anio[sel].to_numpy())
    fuera.insert(0, "CIV", p.CIV[sel].to_numpy())
    fuera["log_long_km"] = p.log_long_km[sel].to_numpy().astype("float32")
    fuera["factor_eb"] = p.CIV[sel].map(factor).to_numpy().astype("float32")
    fuera = fuera.reset_index(drop=True)
    a.caracteristicas.parent.mkdir(parents=True, exist_ok=True)
    fuera.to_parquet(a.caracteristicas, index=False)

    config = {
        "package_name": NOMBRE_PAQUETE,
        "version": version,
        "empaquetado": date.today().isoformat(),
        "experimento": a.experimento,
        "tracking_uri": a.tracking_uri,
        "criterio": eleccion["criterio"]["regla"],
        "modelos": modelos,
        "alpha_nb2": round(alpha, 4),
        "target": "graves",
        "offset": "log_long_km",
        "test_data_file": "muestra_2021.csv",
        "anio_entrenamiento": [2016, 2020],
        "anio_validacion": ANIO_VALIDA,
        "anio_pronostico": ANIO_PRONOSTICO,
        "no_son_variables": list(NO_SON_VARIABLES),
        "features": list(X.columns),
    }
    (a.paquete / "config.yml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    muestra = fuera[fuera.anio_ref == ANIO_VALIDA].sample(MUESTRA, random_state=SEMILLA_MUESTRA)
    muestra.to_csv(a.paquete / "datasets" / "muestra_2021.csv", index=False)

    tam = a.caracteristicas.stat().st_size / 1e6
    print(f"  caracteristicas  {len(fuera):,} filas x {len(fuera.columns)} columnas  ({tam:.1f} MB)")
    print(f"  paquete          {NOMBRE_PAQUETE} {version}  alpha_nb2 {alpha:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
