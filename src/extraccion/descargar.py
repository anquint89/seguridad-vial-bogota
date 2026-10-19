"""Descarga las cuatro capas principales de la SDM a data/raw.

Uso:
    python -m src.extraccion.descargar                 # las cuatro capas
    python -m src.extraccion.descargar malla siniestros
    python -m src.extraccion.descargar --salida /ruta/alterna

Cada capa queda en dos archivos: el parquet con los atributos y un .meta.json
con el conteo devuelto por el servidor, la fecha de consulta y la lista de
campos. El .meta.json es el soporte de trazabilidad que pide la rubrica: deja
constancia de contra que version del servicio se corrio el pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import arcgis

log = logging.getLogger("descargar")

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_POR_DEFECTO = RAIZ / "data" / "raw"

# Campos pedidos explicitamente: traer "*" duplica el peso con columnas de
# etiquetado que no entran al modelo.
CAPAS = {
    "malla": {
        "servicio": "Malla_Vial_Integral",
        "archivo": "malla_vial_integral",
        "campos": (
            "OBJECTID,MVICIV,MVICODIGO,MVICCAT,MVITIPO,MVITCLA,MVINUMC,"
            "MVIVELREG,MVICCALZAD,MVINOMBRE,Shape__Length"
        ),
        "geometria": True,
        # Se pide la geometria reproyectada a MAGNA-SIRGAS / Bogota (EPSG:3116)
        # para que la longitud quede en metros y sirva como exposicion.
        "srs_salida": 3116,
        "descripcion": "Malla Vial Integral. Unidad de analisis del proyecto.",
    },
    "siniestros": {
        "servicio": "HistoricoSiniestros",
        "archivo": "historico_siniestros",
        "campos": (
            "OBJECTID,FORMULARIO,CODIGO_ACCIDENTE,FECHA_OCURRENCIA_ACC,"
            "ANO_OCURRENCIA_ACC,GRAVEDAD,CLASE_ACC,LOCALIDAD,CIV,PK_CALZADA,"
            "LATITUD,LONGITUD"
        ),
        "geometria": False,
        "srs_salida": None,
        "descripcion": "Historico de siniestros 2015-2021. Variable respuesta.",
    },
    "victimas": {
        "servicio": "Victima_2021_may2025_Loc",
        "archivo": "victimas_2021_2025",
        "campos": (
            "OBJECTID,CODIGO_ACCIDENTADO,FORMULARIO,FECHA_OCURRENCIA_ACC,"
            "ANO_OCURRENCIA_ACC,CLASE_ACC,LOCALIDAD,CONDICION,GENERO,EDAD,"
            "TIPO_VICTIMA,MUERTE_POSTERIOR,LOCNOMBRE,LOCCODIGO"
        ),
        "geometria": True,
        "srs_salida": 3116,
        "descripcion": "Victimas 2021 - mayo 2025. No trae CIV: se cruza por FORMULARIO.",
    },
    "intervenciones": {
        "servicio": "Intervenciones_Seguridad_Vial",
        "archivo": "intervenciones_seguridad_vial",
        "campos": (
            "OBJECTID,CIV,TIPO,CATEGORIA,DESCRIPCIO,DIRECCION,FECHA_FIN_,"
            "ANIO,LATITUD_TR,LONGITUD_T"
        ),
        "geometria": False,
        "srs_salida": None,
        "descripcion": "Intervenciones ejecutadas. Tratamiento del diseno causal.",
    },
}


def descargar_capa(clave: str, salida: Path) -> Path:
    config = CAPAS[clave]
    servicio = config["servicio"]
    log.info("Capa %s (%s): consultando metadatos.", clave, servicio)

    metadatos = arcgis.describir(servicio)
    esperados = arcgis.contar(servicio)
    log.info("El servicio reporta %s registros.", f"{esperados:,}".replace(",", "."))

    registros = list(
        arcgis.paginar(
            servicio,
            campos=config["campos"],
            con_geometria=config["geometria"],
            srs_salida=config["srs_salida"],
        )
    )
    if len(registros) != esperados:
        raise RuntimeError(
            f"Se descargaron {len(registros)} registros y el servicio reporta "
            f"{esperados}. No se escribe el archivo."
        )

    tabla = pd.DataFrame.from_records(registros)
    for columna in ("FECHA_OCURRENCIA_ACC", "FECHA_FIN_"):
        if columna in tabla.columns:
            tabla[columna] = pd.to_datetime(tabla[columna], unit="ms", errors="coerce")
    if "geometry" in tabla.columns:
        tabla["geometry"] = tabla["geometry"].apply(
            lambda valor: json.dumps(valor) if valor is not None else None
        )

    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / f"{config['archivo']}.parquet"
    tabla.to_parquet(destino, index=False)

    ficha = {
        "capa": clave,
        "servicio": servicio,
        "url": f"{arcgis.BASE}/{servicio}/FeatureServer/0",
        "descripcion": config["descripcion"],
        "consultado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registros": len(tabla),
        "registros_reportados_por_el_servicio": esperados,
        "srs_origen": metadatos.get("extent", {}).get("spatialReference"),
        "srs_salida": config["srs_salida"],
        "campos": list(tabla.columns),
    }
    (salida / f"{config['archivo']}.meta.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Escrito %s (%s filas).", destino.name, len(tabla))
    return destino


def main(argumentos: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument(
        "capas",
        nargs="*",
        choices=list(CAPAS),
        metavar="capa",
        help="Capas a descargar. Sin argumentos se descargan las cuatro.",
    )
    analizador.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    opciones = analizador.parse_args(argumentos)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for clave in opciones.capas or list(CAPAS):
        descargar_capa(clave, opciones.salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
