"""Tablero de priorizacion (Dash). Consume la API; no lee archivos de datos.

    1) API:     uvicorn src.api.main:app --port 8000
    2) Tablero: python -m src.tablero.app          -> http://localhost:8050

La URL de la API se cambia con la variable de entorno API_URL.

Reparto de la maqueta:
  A. Mapa       <- GET /segmentos/top  (GeoJSON del top-N)
  B. Ranking    <- GET /ranking        (se baja una vez por ano/localidad)
  C. What-if    <- GET /factores       (una vez); recalculo local
  D. Paquete    <- calculo local: mochila voraz sobre el ranking ya descargado
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from src.tablero.optimizador import armar_paquete, opciones, resumen

# 127.0.0.1 y no localhost: en Windows localhost prueba IPv6 primero y pierde ~2 s por llamada.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
TOP_N = 2000
PRESUPUESTO_MAX = 50_000  # millones COP

# Cortes del color del mapa (graves esperados por ano). El top-N arranca cerca
# de 0,8, por eso no se usan los cortes ilustrativos de la maqueta.
CORTES = [0, 1.0, 1.5, 2.5, 5.0, float("inf")]
ETIQUETAS = ["< 1,0", "1,0 - 1,5", "1,5 - 2,5", "2,5 - 5,0", "> 5,0"]
COLORES = ["#fed976", "#fd8d3c", "#e8590c", "#c92a2a", "#67000d"]
SELECCION = "#1c7ed6"
ANIOS = {2021: "2021 · validación (hay observado)", 2022: "2022 · pronóstico"}


# ----------------------------------------------------------------- API
def api(ruta: str, **params):
    r = requests.get(f"{API_URL}{ruta}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def num(valor, dec=2):
    """Formato colombiano: 1.234,56"""
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def millones(valor, dec=1):
    return f"$ {num(valor, dec)} M"


# -------------------------------------------------------------- layout
def tarjeta(titulo, subtitulo, *hijos, clase=""):
    return html.Section(
        [html.H2(titulo), html.P(subtitulo, className="sub"), *hijos], className=f"tarjeta {clase}"
    )


def panel_whatif(factores: list[dict]):
    filas = [
        html.Tr([html.Th("Categoría"), html.Th("Factor (IC 90 %)"), html.Th("Costo unitario"), html.Th("Aplicar")])
    ]
    for f in factores:
        unidad = "M/km" if f["unidad_costo"] == "km" else "M/seg."
        filas.append(
            html.Tr(
                [
                    html.Td(f["categoria"]),
                    html.Td(f"{num(f['factor'], 2)}  ({num(f['ic90'][0], 2)}; {num(f['ic90'][1], 2)})"),
                    html.Td(
                        [
                            dcc.Input(
                                id={"tipo": "costo", "clave": f["clave"]},
                                type="number", min=0, step=0.1,
                                value=f["costo_unitario_millones"], debounce=True,
                                className="costo",
                            ),
                            html.Span(unidad, className="unidad"),
                        ]
                    ),
                    html.Td(
                        dcc.Checklist(
                            id={"tipo": "aplicar", "clave": f["clave"]},
                            options=[{"label": "", "value": "si"}], value=["si"],
                        )
                    ),
                ]
            )
        )
    return html.Table(filas, className="whatif")


def construir_layout():
    try:
        salud = api("/salud")
        factores = api("/factores")
        metricas = api("/metricas")
        localidades = api("/localidades")
    except requests.RequestException as e:
        return html.Div(
            [html.H1("No se pudo conectar con la API"), html.P(f"{API_URL}: {e}"),
             html.P("Levántela con: uvicorn src.api.main:app --port 8000")],
            className="error",
        )

    captura = metricas.get("captura_top5_2021", {})
    modelo_txt = (
        f"Modelo: {salud['modelo']} · captura del top 5 % en 2021: "
        f"{num(captura.get('esperado_eb'), 1)} % (línea base {num(captura.get('linea_base_historico'), 1)} %)"
    )

    barra = html.Header(
        [
            html.Div(
                [
                    html.H1("Priorización de intervenciones de seguridad vial · Bogotá"),
                    html.P("Secretaría Distrital de Movilidad · unidad de decisión: el segmento CIV · " + modelo_txt),
                ]
            ),
            html.Div(
                [
                    html.Label("Año de referencia"),
                    dcc.Dropdown(
                        id="anio", clearable=False, value=2021,
                        options=[{"label": ANIOS.get(a, str(a)), "value": a} for a in salud["anios"]],
                    ),
                ],
                className="filtro",
            ),
            html.Div(
                [
                    html.Label("Localidad"),
                    dcc.Dropdown(
                        id="localidad", clearable=False, value="TODAS",
                        options=[{"label": "Todas", "value": "TODAS"}]
                        + [{"label": l.title(), "value": l} for l in localidades],
                    ),
                ],
                className="filtro",
            ),
        ],
        className="barra",
    )

    mapa = tarjeta(
        "A. Mapa de la malla coloreado por riesgo esperado",
        f"Siniestros graves esperados por segmento y por año, corregidos por Empirical Bayes. "
        f"Se dibujan los {num(TOP_N, 0)} segmentos de mayor riesgo; haga clic en uno para ver su detalle.",
        html.Div(
            [
                dcc.Loading(dcc.Graph(id="mapa", config={"displaylogo": False, "scrollZoom": True})),
                html.Div(id="detalle", className="detalle"),
            ],
            className="mapa-cuerpo",
        ),
        clase="a",
    )

    whatif = tarjeta(
        "C. What-if por categoría de intervención",
        "Factores por diferencias en diferencias sobre las intervenciones de 2016. Edite los costos "
        "o apague categorías y el paquete se recalcula.",
        panel_whatif(factores["factores"]),
        html.P(factores.get("advertencia", ""), className="nota"),
        clase="c",
    )

    presupuesto = tarjeta(
        "D. Presupuesto disponible y paquete resultante",
        "Algoritmo voraz de mochila sobre el ranking descargado: prioriza la mayor reducción por peso invertido.",
        html.Div(id="presupuesto-valor", className="presupuesto-valor"),
        dcc.Slider(
            id="presupuesto", min=0, max=PRESUPUESTO_MAX, step=500, value=10_000,
            marks={0: "0", 25_000: "25.000", PRESUPUESTO_MAX: "50.000"},
            updatemode="mouseup",
        ),
        html.Div(id="kpis", className="kpis"),
        clase="d",
    )

    columnas = [
        ("rango", "#", "numeric"), ("CIV", "CIV", "numeric"), ("via", "Vía", "text"),
        ("localidad", "Localidad", "text"), ("observado", "Graves año ref.", "numeric"),
        ("esperado_eb", "Esperado EB", "numeric"), ("prob_grave", "P(≥1 grave)", "numeric"),
        ("long_m", "Longitud (m)", "numeric"), ("intervencion", "Intervención recomendada", "text"),
        ("reduccion", "Reducción", "numeric"), ("costo", "Costo (M)", "numeric"),
        ("en_paquete", "En paquete", "text"),
    ]
    tabla = tarjeta(
        "B. Tabla del ranking priorizado",
        "Ordenable y filtrable por cualquier columna, exportable a CSV. Al hacer clic en una fila "
        "se ilumina el segmento en el mapa.",
        dash_table.DataTable(
            id="tabla",
            columns=[{"id": c, "name": n, "type": t} for c, n, t in columnas],
            page_size=15, sort_action="native", filter_action="native",
            export_format="csv", export_headers="display",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "'Segoe UI', system-ui, sans-serif", "fontSize": 13,
                        "padding": "6px 8px", "textAlign": "left"},
            style_header={"fontWeight": 600, "backgroundColor": "#f1f3f5"},
            style_data_conditional=[
                {"if": {"filter_query": '{en_paquete} = "sí"'}, "backgroundColor": "#fff4e6"},
                {"if": {"state": "active"}, "backgroundColor": "#d0ebff", "border": "1px solid #1c7ed6"},
            ],
        ),
        clase="b",
    )

    return html.Div(
        [
            barra,
            html.Main([mapa, html.Div([whatif, presupuesto], className="lateral"), tabla], className="rejilla"),
            dcc.Store(id="datos-ranking"),
            dcc.Store(id="datos-geo"),
            dcc.Store(id="datos-tabla"),
            dcc.Store(id="civ-sel"),
            dcc.Store(id="factores", data=factores["factores"]),
        ]
    )


app = Dash(__name__, title="Seguridad vial Bogotá")
app.layout = construir_layout  # funcion: se evalua en cada carga de la pagina
server = app.server  # para gunicorn en Docker


# ----------------------------------------------------------- callbacks
@app.callback(
    Output("datos-ranking", "data"),
    Output("datos-geo", "data"),
    Output("civ-sel", "data", allow_duplicate=True),
    Input("anio", "value"),
    Input("localidad", "value"),
    prevent_initial_call="initial_duplicate",
)
def cargar(anio, localidad):
    """Unica llamada a la API por cambio de filtros."""
    ranking = api("/ranking", anio=anio, top=TOP_N, localidad=localidad)
    geo = api("/segmentos/top", anio=anio, n=TOP_N, localidad=localidad)
    return ranking, geo, None


@app.callback(
    Output("datos-tabla", "data"),
    Output("kpis", "children"),
    Output("presupuesto-valor", "children"),
    Input("datos-ranking", "data"),
    Input({"tipo": "costo", "clave": ALL}, "value"),
    Input({"tipo": "aplicar", "clave": ALL}, "value"),
    Input("presupuesto", "value"),
    State("factores", "data"),
)
def recalcular(ranking, costos, aplicar, presupuesto, factores):
    """What-if + mochila. No llama a la API."""
    if not ranking:
        return [], "", ""
    claves = [f["clave"] for f in factores]
    costos_d = {k: (v if v is not None else 0) for k, v in zip(claves, costos)}
    activas = {k for k, v in zip(claves, aplicar) if v}
    base = pd.DataFrame(ranking)
    tabla = armar_paquete(base, opciones(base, factores, activas, costos_d), presupuesto)
    r = resumen(tabla)
    tabla["en_paquete"] = tabla.en_paquete.map({True: "sí", False: "no"})
    mezcla = ", ".join(f"{k}: {num(v, 0)}" for k, v in r["por_intervencion"].items()) or "—"

    filas = [
        ("Segmentos en el paquete", num(r["segmentos"], 0)),
        ("Inversión del paquete", millones(r["costo"], 0)),
        ("Siniestros graves evitados por año",
         f"{num(r['evitados'], 1)}  (IC 90 %: {num(r['evitados_bajo'], 1)} – {num(r['evitados_alto'], 1)})"),
        ("Costo por siniestro grave evitado", millones(r["costo_por_evitado"])),
        ("Localidades alcanzadas", f"{r['localidades']} de {r['localidades_total']}"),
        ("Inversión en las 5 localidades de mayor riesgo", f"{num(r['participacion_top5'], 1)} %"),
        ("Mezcla de intervenciones", mezcla),
    ]
    kpis = html.Table([html.Tr([html.Td(a), html.Td(b, className="valor")]) for a, b in filas])
    for col, dec in (("esperado_eb", 2), ("prob_grave", 3), ("reduccion", 2), ("costo", 1)):
        tabla[col] = tabla[col].round(dec)
    tabla["id"] = tabla.CIV
    return tabla.to_dict("records"), kpis, f"$ {num(presupuesto, 0)} millones COP"


@app.callback(Output("tabla", "data"), Input("datos-tabla", "data"))
def llenar_tabla(datos):
    return datos


@app.callback(
    Output("civ-sel", "data"),
    Input("mapa", "clickData"),
    Input("tabla", "active_cell"),
    prevent_initial_call=True,
)
def seleccionar(click, celda):
    if ctx.triggered_id == "mapa" and click:
        civ = click["points"][0].get("customdata")
        return civ if civ is not None else no_update  # clic sobre el trazo de seleccion
    if ctx.triggered_id == "tabla" and celda:
        return celda.get("row_id")
    return no_update


@app.callback(Output("mapa", "figure"), Input("datos-geo", "data"), Input("civ-sel", "data"))
def dibujar_mapa(geo, civ_sel):
    fig = go.Figure()
    grupos = {i: ([], [], []) for i in range(len(ETIQUETAS))}
    sel = ([], [])
    for feat in (geo or {}).get("features", []):
        p = feat["properties"]
        i = next(j for j in range(len(ETIQUETAS)) if p["esperado_eb"] < CORTES[j + 1])
        lon, lat, cd = grupos[i]
        for trazo in feat["geometry"]["coordinates"]:
            for x, y in trazo:
                lon.append(x)
                lat.append(y)
                cd.append(p["CIV"])
            lon.append(None)
            lat.append(None)
            cd.append(None)
            if p["CIV"] == civ_sel:
                sel[0].extend([x for x, _ in trazo] + [None])
                sel[1].extend([y for _, y in trazo] + [None])
    for i, (lon, lat, cd) in grupos.items():
        fig.add_trace(
            go.Scattermap(
                lon=lon, lat=lat, customdata=cd, mode="lines",
                line={"width": 2 + i * 0.8, "color": COLORES[i]},
                name=ETIQUETAS[i], hovertemplate="CIV %{customdata}<extra>" + ETIQUETAS[i] + "</extra>",
            )
        )
    if sel[0]:
        fig.add_trace(
            go.Scattermap(lon=sel[0], lat=sel[1], mode="lines", line={"width": 8, "color": SELECCION},
                          name="Seleccionado", hoverinfo="skip")
        )
        centro, zoom = encuadre(sel[0], sel[1], zoom_max=15.5)
    else:
        todos_lon = [x for lon, _, _ in grupos.values() for x in lon]
        todos_lat = [y for _, lat, _ in grupos.values() for y in lat]
        centro, zoom = encuadre(todos_lon, todos_lat)
    fig.update_layout(
        map={"style": "carto-positron", "center": centro, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=480,
        legend={"title": {"text": "Graves esperados/año"}, "x": 0.01, "y": 0.99,
                "bgcolor": "rgba(255,255,255,0.85)"},
        # El encuadre se recalcula al cambiar filtros o seleccion; entre tanto se
        # conserva el zoom que haya puesto el usuario.
        uirevision=f"{centro}-{civ_sel}",
    )
    return fig


def encuadre(lon, lat, zoom_max=13.0, ancho_px=520, alto_px=480):
    """Centro y zoom (Web Mercator) para que quepan las coordenadas dadas.

    Usa los percentiles 2-98 para que un segmento aislado no aleje la vista.
    A la latitud de Bogota (4,6 grados) la distorsion de Mercator es despreciable.
    """
    lon = [x for x in lon if x is not None]
    lat = [y for y in lat if y is not None]
    if not lon:
        return {"lat": 4.65, "lon": -74.1}, 10.3
    corte = (2, 98) if len(lon) > 50 else (0, 100)
    lon0, lon1 = np.percentile(lon, corte)
    lat0, lat1 = np.percentile(lat, corte)
    # MapLibre: en zoom 0 el mundo (360 grados) mide 512 px.
    z_lon = np.log2(360 * ancho_px / (512 * max(lon1 - lon0, 1e-4)))
    z_lat = np.log2(360 * alto_px / (512 * max(lat1 - lat0, 1e-4)))
    zoom = float(np.clip(min(z_lon, z_lat) - 0.15, 9.5, zoom_max))
    return {"lat": float((lat0 + lat1) / 2), "lon": float((lon0 + lon1) / 2)}, round(zoom, 2)


@app.callback(Output("detalle", "children"), Input("civ-sel", "data"), Input("datos-tabla", "data"))
def detalle(civ, tabla):
    if not tabla:
        return ""
    if civ is None:
        return html.P("Seleccione un segmento en el mapa o en la tabla.", className="nota")
    fila = next((f for f in tabla if f["CIV"] == civ), None)
    if fila is None:
        return html.P(f"CIV {civ} fuera del top cargado.", className="nota")
    filas = [
        ("Vía", fila["via"]),
        ("Localidad", fila["localidad"].title()),
        ("Ranking", f"#{num(fila['rango'], 0)}"),
        ("Observado en el año", num(fila["observado"], 0)),
        ("Esperado (EB)", f"{num(fila['esperado_eb'])} graves/año"),
        ("P(≥1 grave)", f"{num(100 * fila['prob_grave'], 1)} %"),
        ("Graves años previos", num(fila["graves_previos"], 0)),
        ("Exposición", f"{num(fila['long_m'], 0)} m, clase {fila['clase_via']}, {fila['carriles']} carriles"),
        ("Intervenido en 2016", "sí" if fila["intervenido_2016"] else "no"),
        ("Recomendado", fila["intervencion"] or "—"),
        ("Reducción esperada", f"{num(fila['reduccion'])} graves/año"),
        ("Costo", millones(fila["costo"])),
        ("En el paquete", fila["en_paquete"]),
    ]
    return [html.H3(f"CIV {civ}"), html.Table([html.Tr([html.Td(a), html.Td(b)]) for a, b in filas])]


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "0") == "1", host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
