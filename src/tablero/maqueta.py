"""Dibuja la maqueta del tablero que se entrega en la Entrega 1.

Es un alambrado, no un diseno: fija la disposicion de los cuatro elementos, de
donde sale cada dato y que se calcula en el navegador. Se genera por codigo
para que quede versionado y se pueda iterar en las semanas siguientes.

    python -m src.tablero.maqueta
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RAIZ = Path(__file__).resolve().parents[2]
DESTINO = RAIZ / "reportes" / "figuras" / "maqueta_tablero.png"

ANCHO, ALTO = 1440, 900
BORDE = "0.4"
RELLENO = "0.98"


def caja(ejes, x, y, ancho, alto, relleno=RELLENO, borde=BORDE, grosor=1.0):
    ejes.add_patch(
        Rectangle(
            (x, y), ancho, alto, facecolor=relleno, edgecolor=borde, linewidth=grosor
        )
    )


def texto(ejes, x, y, contenido, tam=9, peso="normal", color="black", ha="left"):
    ejes.text(
        x, y, contenido, fontsize=tam, fontweight=peso, color=color, ha=ha, va="top"
    )


def etiqueta(ejes, x, y, contenido):
    """Marca de donde sale el dato: de la API o del navegador."""
    ancho = len(contenido) * 6.45 + 14
    caja(ejes, x, y, ancho, 17, relleno="black", borde="black", grosor=0)
    ejes.text(
        x + 6,
        y + 8.5,
        contenido,
        fontsize=7.5,
        color="white",
        va="center",
        family="monospace",
    )


def main() -> int:
    figura = plt.figure(figsize=(14.4, 9.0))
    ejes = figura.add_axes([0, 0, 1, 1])
    ejes.set_xlim(0, ANCHO)
    ejes.set_ylim(ALTO, 0)
    ejes.set_autoscale_on(False)
    ejes.axis("off")

    # --------------------------------------------------------------- barra
    caja(ejes, 20, 20, 1400, 56, relleno="0.92")
    texto(ejes, 36, 32, "Priorización de intervenciones de seguridad vial · Bogotá", 13, "bold")
    texto(ejes, 36, 54, "Secretaría Distrital de Movilidad · unidad de decisión: el segmento CIV", 8, color="0.3")
    for x, ancho, titulo in (
        (950, 170, "Año de referencia: 2021"),
        (1130, 150, "Localidad: todas"),
        (1290, 100, "Actualizar"),
    ):
        caja(ejes, x, 34, ancho, 28, relleno="white")
        texto(ejes, x + 10, 42, titulo, 8.5)

    # ------------------------------------------------------------- A. mapa
    caja(ejes, 20, 92, 830, 470, relleno="white")
    texto(ejes, 36, 102, "A. Mapa de la malla coloreado por riesgo esperado", 11.5, "bold")
    texto(ejes, 36, 122, "Siniestros graves esperados por segmento y por año, corregidos por Empirical Bayes.", 8.5, color="0.3")
    caja(ejes, 36, 142, 798, 330, relleno="0.985", borde="0.6")
    texto(ejes, 290, 290, "[ mapa base + capa de segmentos priorizados ]", 10, color="0.55", ha="center")
    texto(ejes, 290, 310, "zoom, desplazamiento y selección de segmento", 8, color="0.6", ha="center")

    texto(ejes, 598, 166, "Riesgo esperado (graves/año)", 8, "bold")
    for indice, (rango, tono) in enumerate(
        (
            ("0,00 - 0,05", "0.90"),
            ("0,05 - 0,20", "0.72"),
            ("0,20 - 0,60", "0.54"),
            ("0,60 - 1,50", "0.33"),
            ("más de 1,50", "0.08"),
        )
    ):
        y = 184 + indice * 18
        caja(ejes, 598, y, 22, 11, relleno=tono, borde="0.5", grosor=0.6)
        texto(ejes, 628, y - 1, rango, 8)

    texto(ejes, 570, 292, "Seleccionado: CIV 10003679", 8, "bold")
    for indice, linea in enumerate(
        (
            "Observado 2021: 9 graves",
            "Esperado (EB): 6,41 graves/año",
            "Exposición: 412 m, arterial, 3 carriles",
            "Recomendado: tráfico calmado",
            "Reducción esperada: 1,47 graves/año",
        )
    ):
        texto(ejes, 570, 310 + indice * 15, linea, 8)

    etiqueta(ejes, 36, 486, "API  GET /segmentos/top?n=2000&bbox=...  ->  GeoJSON")
    texto(
        ejes,
        36,
        512,
        "Rendimiento: no se dibujan los 139.107 registros de la malla. La API devuelve únicamente el top-N del\n"
        "ranking, con la geometría simplificada a 15 m por Douglas-Peucker y recortada al encuadre visible. El\n"
        "resto de la malla entra como teselas vectoriales pre-generadas: es fondo y no se consulta.",
        8,
        color="0.25",
    )

    # ---------------------------------------------------------- C. what-if
    caja(ejes, 866, 92, 554, 240, relleno="white")
    texto(ejes, 882, 102, "C. What-if por categoría de intervención", 11.5, "bold")
    texto(ejes, 882, 121, "Factores estimados con diferencias en diferencias sobre las intervenciones de 2016.", 8, color="0.3")
    for x, encabezado in ((886, "Categoría"), (1066, "Factor (IC 90 %)"), (1236, "Costo unitario"), (1360, "Aplicar")):
        texto(ejes, x, 146, encabezado, 8, "bold")
    ejes.plot([882, 1404], [160, 160], color="0.6", linewidth=0.8)
    for indice, (categoria, factor, costo) in enumerate(
        (
            ("Señalización", "-0,08  (-0,14; -0,02)", "$ 2,1 M"),
            ("Kilometros demarcados", "-0,11  (-0,19; -0,03)", "$ 8,4 M"),
            ("Paso peatonal", "-0,17  (-0,26; -0,08)", "$ 14,0 M"),
            ("Tráfico calmado", "-0,23  (-0,35; -0,11)", "$ 31,5 M"),
            ("Zona escolar", "-0,19  (-0,33; -0,05)", "$ 22,0 M"),
        )
    ):
        y = 170 + indice * 22
        texto(ejes, 886, y, categoria, 8.5)
        texto(ejes, 1066, y, factor, 8.5)
        caja(ejes, 1236, y - 3, 66, 16, relleno="0.96", borde="0.7", grosor=0.6)
        texto(ejes, 1242, y, costo, 8)
        caja(ejes, 1368, y - 2, 12, 12, relleno="0.2", borde="0.4", grosor=0.6)
    texto(ejes, 882, 286, "Valores ilustrativos: los definitivos los produce el módulo causal.", 7.5, color="0.45")
    etiqueta(ejes, 882, 306, "API GET /factores  |  recálculo en el navegador")

    # ------------------------------------------------------ D. presupuesto
    caja(ejes, 866, 348, 554, 214, relleno="white")
    texto(ejes, 882, 358, "D. Presupuesto disponible y paquete resultante", 11.5, "bold")
    ejes.text(1182, 392, "$ 28.500 millones COP", fontsize=9.5, fontweight="bold", ha="center", va="bottom")
    ejes.plot([890, 1396], [402, 402], color="0.5", linewidth=1.4)
    ejes.plot([890, 1182], [402, 402], color="black", linewidth=3.0)
    ejes.plot([1182], [402], marker="o", color="black", markersize=8)
    texto(ejes, 890, 410, "0", 8, color="0.4")
    texto(ejes, 1396, 410, "$ 50.000 M", 8, color="0.4", ha="right")
    for indice, (concepto, valor) in enumerate(
        (
            ("Segmentos en el paquete", "1.284"),
            ("Siniestros graves evitados por año", "212  (IC 90 %: 141 - 288)"),
            ("Costo por siniestro grave evitado", "$ 134,4 M COP"),
            ("Localidades alcanzadas", "19 de 20"),
            ("Participación de las 5 localidades con mayor tasa", "41,2 %"),
        )
    ):
        y = 436 + indice * 19
        texto(ejes, 890, y, concepto, 8.5)
        texto(ejes, 1396, y, valor, 8.5, "bold", ha="right")
    etiqueta(ejes, 882, 536, "TABLERO  greedy de mochila sobre el top-N ya descargado")

    # ---------------------------------------------------------- B. ranking
    caja(ejes, 20, 578, 1400, 268, relleno="white")
    texto(ejes, 36, 588, "B. Tabla del ranking priorizado", 11.5, "bold")
    texto(ejes, 36, 608, "Ordenable por cualquier columna y exportable a CSV. La fila seleccionada ilumina el segmento en el mapa.", 8, color="0.3")
    columnas = (
        ("#", 40),
        ("CIV", 70),
        ("Vía", 150),
        ("Localidad", 420),
        ("Graves 2021", 545),
        ("Esperado EB", 650),
        ("Exposición", 755),
        ("Intervención recomendada", 860),
        ("Reducción", 1095),
        ("Costo", 1190),
        ("En paquete", 1300),
    )
    for encabezado, x in columnas:
        texto(ejes, x, 632, encabezado, 8, "bold")
    ejes.plot([36, 1404], [648, 648], color="0.6", linewidth=0.8)
    filas = (
        ("1", "10003679", "AK 68 con CL 13", "Puente Aranda", "9", "6,41", "412 m", "Tráfico calmado", "1,47", "$ 31,5 M", "sí"),
        ("2", "11002415", "AC 80 con KR 69", "Engativá", "7", "5,88", "388 m", "Paso peatonal", "1,00", "$ 14,0 M", "sí"),
        ("3", "8001277", "AV Caracas con CL 45", "Chapinero", "8", "5,52", "341 m", "Tráfico calmado", "1,27", "$ 31,5 M", "sí"),
        ("4", "4000931", "AV 1 de Mayo con KR 24", "Puente Aranda", "6", "4,97", "295 m", "Señalización", "0,40", "$ 2,1 M", "sí"),
        ("5", "19004277", "AV Boyacá con CL 42 S", "Ciudad Bolívar", "5", "4,61", "503 m", "Km demarcados", "0,51", "$ 8,4 M", "sí"),
        ("6", "9002188", "CL 26 con KR 50", "Teusaquillo", "6", "4,30", "376 m", "Paso peatonal", "0,73", "$ 14,0 M", "no"),
    )
    for indice, fila in enumerate(filas):
        y = 660 + indice * 23
        if indice % 2:
            caja(ejes, 36, y - 5, 1368, 21, relleno="0.97", borde="0.97", grosor=0)
        for valor, (_, x) in zip(fila, columnas):
            texto(ejes, x, y, valor, 8.2)
    texto(ejes, 36, 806, "1 - 25 de 2.000      <    >", 8, color="0.4")
    etiqueta(ejes, 1120, 802, "API GET /ranking?top=2000")
    texto(ejes, 36, 824, "La columna 'En paquete' no viene de la API: la pone el optimizador que corre en el tablero.", 7.5, color="0.45")

    texto(ejes, 20, 862, "Maqueta de la Entrega 1. Los valores son ilustrativos y se reemplazan con la salida del modelo.", 8, color="0.45")

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(DESTINO, dpi=150, facecolor="white")
    plt.close(figura)
    print(f"Escrito {DESTINO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
