"""Empaqueta el modelo elegido para que la API pueda inferir.

Saca de MLflow los dos modelos seleccionados y los deja como archivos planos
de LightGBM, junto con la matriz de caracteristicas de los anios que el
tablero muestra. Con eso la API calcula la prediccion en cada peticion en vez
de leer una tabla ya calculada, y no necesita mlflow instalado en tiempo de
ejecucion, que es lo que mantiene la imagen pequena.

ENTRA : reportes/modelos/seleccion.json y los modelos en MLflow
        data/procesado/panel.parquet y vecindad.parquet
SALE  : modelos/conteo.txt, modelos/clasificacion.txt
        modelos/metadatos.json       de que corrida salio cada uno
        data/procesado/caracteristicas_inferencia.parquet

POR QUE ARCHIVOS PLANOS Y NO EL REGISTRO DE MLFLOW
  La API corre en un contenedor que no tiene por que alcanzar el servidor de
  seguimiento, que ademas se apaga al cerrar la entrega. El vinculo con la
  corrida no se pierde: queda escrito en metadatos.json.

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
DESTINO = RAIZ / "modelos"
CARACTERISTICAS = RAIZ / "data" / "procesado" / "caracteristicas_inferencia.parquet"
TRACKING = "sqlite:///mlflow.db"
EXPERIMENTO = "seguridad-vial-bogota-e3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--vecindad", type=Path, default=VECINDAD)
    ap.add_argument("--seleccion", type=Path, default=SELECCION)
    ap.add_argument("--destino", type=Path, default=DESTINO)
    ap.add_argument("--caracteristicas", type=Path, default=CARACTERISTICAS)
    ap.add_argument("--tracking-uri", default=TRACKING)
    ap.add_argument("--experimento", default=EXPERIMENTO)
    a = ap.parse_args()

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

    a.destino.mkdir(parents=True, exist_ok=True)
    metadatos = {
        "empaquetado": date.today().isoformat(),
        "experimento": a.experimento,
        "tracking_uri": a.tracking_uri,
        "modelos": {},
    }
    for tarea, corrida in corridas.items():
        modelo = cargar_modelo(cliente, exp_id, corrida)
        archivo = a.destino / f"{tarea}.txt"
        modelo.save_model(str(archivo))
        metadatos["modelos"][tarea] = {
            "corrida": corrida,
            "archivo": archivo.name,
            "arboles": modelo.num_trees(),
            "variables": modelo.num_feature(),
            "metricas_validacion": eleccion["metricas"][corrida],
        }
        print(f"  {tarea:14s} {corrida}  ->  {archivo.name}  ({modelo.num_trees()} arboles)")

    (a.destino / "metadatos.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # La matriz que la API necesita para inferir, solo de los anios que el
    # tablero muestra. El desplazamiento viaja como una columna mas.
    panel = pd.read_parquet(a.panel)
    aristas = pd.read_parquet(a.vecindad) if a.vecindad.exists() else None
    p = preparar(filas_de_pronostico(panel), aristas)
    X = matriz_diseno(p)
    # El factor de Empirical Bayes es un parametro ajustado, no una
    # prediccion, asi que viaja con el modelo y la API solo lo multiplica.
    modelo_conteo = lgb.Booster(model_file=str(a.destino / "conteo.txt"))
    esperado = pd.Series(
        predecir(modelo_conteo, X, p.log_long_km.astype(float)), index=p.index
    )
    factor, alpha = factor_empirical_bayes(p, esperado, p.graves.astype(float))
    metadatos["alpha_nb2"] = round(alpha, 4)
    (a.destino / "metadatos.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sel = p.anio.isin((ANIO_VALIDA, ANIO_PRONOSTICO)).to_numpy()
    fuera = X[sel].astype("float32")
    fuera.insert(0, "anio_ref", p.anio[sel].to_numpy())
    fuera.insert(0, "CIV", p.CIV[sel].to_numpy())
    fuera["log_long_km"] = p.log_long_km[sel].to_numpy().astype("float32")
    fuera["factor_eb"] = p.CIV[sel].map(factor).to_numpy().astype("float32")
    a.caracteristicas.parent.mkdir(parents=True, exist_ok=True)
    fuera.reset_index(drop=True).to_parquet(a.caracteristicas, index=False)

    tam = a.caracteristicas.stat().st_size / 1e6
    print(f"  caracteristicas  {len(fuera):,} filas x {len(fuera.columns)} columnas  ({tam:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
