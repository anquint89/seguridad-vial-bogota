"""Cliente minimo para los servicios ArcGIS REST de la Secretaria Distrital de Movilidad.

Las cuatro capas del proyecto estan publicadas como Feature Services en la
organizacion de ArcGIS Online de la SDM. El servidor limita cada respuesta a
2.000 registros (maxRecordCount) y soporta paginacion por resultOffset, de modo
que toda descarga se hace por bloques.

No requiere token: las capas son publicas.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

BASE = "https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services"
PAGINA = 2000
REINTENTOS = 4
ESPERA_BASE = 3  # segundos; se duplica en cada reintento

log = logging.getLogger(__name__)


def _pedir(url: str, parametros: dict[str, Any]) -> dict:
    """Hace una peticion GET y devuelve el JSON, con reintentos exponenciales."""
    parametros = {**parametros, "f": "json"}
    consulta = f"{url}?{urllib.parse.urlencode(parametros)}"
    ultimo_error: Exception | None = None
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(consulta, timeout=120) as respuesta:
                cuerpo = json.load(respuesta)
            if "error" in cuerpo:
                raise RuntimeError(f"El servicio devolvio un error: {cuerpo['error']}")
            return cuerpo
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            ultimo_error = error
            espera = ESPERA_BASE * (2**intento)
            log.warning("Fallo la peticion (%s). Reintento en %ss.", error, espera)
            time.sleep(espera)
    raise RuntimeError(f"No se pudo consultar {url}") from ultimo_error


def describir(servicio: str, capa: int = 0) -> dict:
    """Metadatos de la capa: nombre, campos, geometria, sistema de referencia."""
    return _pedir(f"{BASE}/{servicio}/FeatureServer/{capa}", {})


def contar(servicio: str, capa: int = 0, where: str = "1=1") -> int:
    """Numero de registros que cumplen la condicion."""
    respuesta = _pedir(
        f"{BASE}/{servicio}/FeatureServer/{capa}/query",
        {"where": where, "returnCountOnly": "true"},
    )
    return int(respuesta["count"])


def agrupar(servicio: str, campo: str, capa: int = 0, where: str = "1=1") -> list[dict]:
    """Conteo agrupado por un campo, resuelto en el servidor.

    Se usa para el perfilamiento: evita descargar la capa completa cuando solo
    se necesita una distribucion de frecuencias.
    """
    estadistica = [
        {
            "statisticType": "count",
            "onStatisticField": "OBJECTID",
            "outStatisticFieldName": "n",
        }
    ]
    respuesta = _pedir(
        f"{BASE}/{servicio}/FeatureServer/{capa}/query",
        {
            "where": where,
            "groupByFieldsForStatistics": campo,
            "outStatistics": json.dumps(estadistica),
            "orderByFields": campo,
        },
    )
    return [rasgo["attributes"] for rasgo in respuesta.get("features", [])]


def paginar(
    servicio: str,
    campos: str = "*",
    capa: int = 0,
    where: str = "1=1",
    con_geometria: bool = False,
    srs_salida: int | None = None,
) -> Iterator[dict]:
    """Recorre la capa completa y entrega los registros uno a uno.

    con_geometria=True trae la geometria; para los cruces por CIV no hace falta
    y la descarga pesa un orden de magnitud menos.
    """
    url = f"{BASE}/{servicio}/FeatureServer/{capa}/query"
    desplazamiento = 0
    while True:
        parametros = {
            "where": where,
            "outFields": campos,
            "returnGeometry": "true" if con_geometria else "false",
            "resultOffset": desplazamiento,
            "resultRecordCount": PAGINA,
            "orderByFields": "OBJECTID",
        }
        if srs_salida:
            parametros["outSR"] = srs_salida
        bloque = _pedir(url, parametros)
        rasgos = bloque.get("features", [])
        if not rasgos:
            return
        for rasgo in rasgos:
            registro = dict(rasgo["attributes"])
            if con_geometria:
                registro["geometry"] = rasgo.get("geometry")
            yield registro
        desplazamiento += len(rasgos)
        if not bloque.get("exceededTransferLimit") and len(rasgos) < PAGINA:
            return
