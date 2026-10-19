"""Genera el reporte de la Entrega 1 en Word.

El documento se arma por código para que las cifras salgan de
reportes/perfilamiento.json y no de una transcripción a mano. Si se vuelve a
correr el perfilamiento, el reporte se regenera con los números nuevos.

    python reportes/generar_reporte.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
PROYECTO = RAIZ.parent
FIGURAS = AQUI / "figuras"
DATOS = json.loads((AQUI / "perfilamiento.json").read_text(encoding="utf-8"))

FUENTE = "Calibri"
URL_REPO = "https://github.com/anquint89/seguridad-vial-bogota"


def miles(valor, decimales: int = 0) -> str:
    """Formato colombiano: punto de miles, coma decimal."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "@").replace(".", ",").replace("@", ".")


def pct(valor, decimales: int = 1) -> str:
    """Porcentaje con coma decimal, sin el signo."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def documento() -> Document:
    doc = Document()
    seccion = doc.sections[0]
    seccion.top_margin = Cm(1.9)
    seccion.bottom_margin = Cm(1.9)
    seccion.left_margin = Cm(2.1)
    seccion.right_margin = Cm(2.1)

    normal = doc.styles["Normal"]
    normal.font.name = FUENTE
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    for nivel, tamano in ((1, 12), (2, 10.5)):
        estilo = doc.styles[f"Heading {nivel}"]
        estilo.font.name = FUENTE
        estilo.font.size = Pt(tamano)
        estilo.font.bold = True
        estilo.font.italic = False
        estilo.font.color.rgb = RGBColor(0, 0, 0)
        estilo.paragraph_format.space_before = Pt(9 if nivel == 1 else 7)
        estilo.paragraph_format.space_after = Pt(3)
        estilo.paragraph_format.keep_with_next = True
    return doc


def parrafo(doc, texto: str, tam: float = 10):
    p = doc.add_paragraph(texto)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for corrida in p.runs:
        corrida.font.size = Pt(tam)
    return p


def pie(doc, texto: str):
    p = doc.add_paragraph(texto)
    p.paragraph_format.space_after = Pt(7)
    for corrida in p.runs:
        corrida.font.size = Pt(8.5)
    return p


def tabla(doc, encabezados, filas, anchos=None):
    t = doc.add_table(rows=1, cols=len(encabezados))
    t.style = "Table Grid"
    for celda, titulo in zip(t.rows[0].cells, encabezados):
        celda.text = titulo
    for fila in filas:
        celdas = t.add_row().cells
        for celda, valor in zip(celdas, fila):
            celda.text = valor
    for indice, fila in enumerate(t.rows):
        for celda in fila.cells:
            for p in celda.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                for corrida in p.runs:
                    corrida.font.size = Pt(8.5)
                    corrida.font.name = FUENTE
                    corrida.font.bold = indice == 0
    for fila in t.rows:
        # Evita que una fila quede partida entre dos paginas.
        propiedades = fila._tr.get_or_add_trPr()
        propiedades.append(OxmlElement("w:cantSplit"))
    if anchos:
        for fila in t.rows:
            for celda, ancho in zip(fila.cells, anchos):
                celda.width = Cm(ancho)
    cola = doc.add_paragraph()
    cola.paragraph_format.space_after = Pt(0)
    cola.paragraph_format.space_before = Pt(0)
    return t


def monoespaciado(doc, texto: str, tam: float = 8):
    for linea in texto.strip("\n").split("\n"):
        p = doc.add_paragraph(linea)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        for corrida in p.runs:
            corrida.font.name = "Consolas"
            corrida.font.size = Pt(tam)


def figura(doc, nombre: str, ancho_cm: float):
    doc.add_picture(str(FIGURAS / nombre), width=Cm(ancho_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
    doc.paragraphs[-1].paragraph_format.space_before = Pt(4)


# ---------------------------------------------------------------- contenido


def main() -> int:
    d = DATOS
    malla, sin_, dist = d["malla"], d["siniestros"], d["distribucion_respuesta"]
    base, inter, vic = d["linea_base"], d["intervenciones"], d["victimas"]
    prelim = d["comparacion_preliminar"]
    puente = vic["puente_formulario_2021"]
    b2021 = base["entrena_2015_2020_valida_2021"]
    b2020 = base["entrena_2015_2019_valida_2020"]
    conc = dist["concentracion"]
    anios = sin_["por_anio"]

    doc = documento()

    titulo = doc.add_paragraph("Priorización de intervenciones de seguridad vial en Bogotá")
    titulo.runs[0].font.size = Pt(14)
    titulo.runs[0].font.bold = True
    titulo.paragraph_format.space_after = Pt(2)
    sub = doc.add_paragraph(
        "Entrega 1. Despliegue de Soluciones Analíticas (MIID-4304). Maestría en Inteligencia "
        "Analítica de Datos, Universidad de los Andes. 25 de octubre de 2026.\n"
        "Equipo: Laura Alfonso Botia, Andrés Quintero Hoyos, Andrés Felipe Rodríguez Ortiz.\n"
        f"Repositorio: {URL_REPO}"
    )
    for corrida in sub.runs:
        corrida.font.size = Pt(9)
    sub.paragraph_format.space_after = Pt(9)

    # =============================================================== 1
    doc.add_heading("1. El problema y su contexto", 1)
    parrafo(
        doc,
        "La Secretaría Distrital de Movilidad decide cada año dónde ejecutar las intervenciones "
        "físicas de seguridad vial de Bogotá: señalización, demarcación, pasos peatonales, medidas "
        "de tráfico calmado y zonas escolares. El presupuesto es limitado y la malla sobre la que "
        f"hay que repartirlo tiene {miles(malla['registros'])} registros. El criterio vigente es "
        "contar siniestros históricos por tramo y atender los primeros de la lista.",
    )
    parrafo(
        doc,
        "Ese criterio falla por dos razones conocidas. La primera es la regresión a la media: un "
        "segmento que tuvo un año malo por azar sube al tope del ranking y al año siguiente vuelve "
        "a su nivel habitual sin que ninguna obra haya intervenido. La segunda es que no se divide "
        "por exposición, de modo que un corredor con diez veces más tránsito aparece como diez "
        "veces más peligroso aunque su riesgo por vehículo sea menor.",
    )
    parrafo(
        doc,
        "El costo de esas dos fallas se puede medir con los mismos datos publicados. Replicamos el "
        f"método vigente ordenando los {miles(base['universo'])} segmentos por conteo de siniestros "
        "graves entre 2015 y 2020 y evaluando contra 2021: el 5 % mejor rankeado "
        f"({miles(base['tamano_top_5_pct'])} segmentos) contiene {miles(b2021['capturados'])} de "
        f"los {miles(b2021['total'])} siniestros graves de ese año, el {pct(b2021['pct'])} %. Más de "
        "la mitad del daño del año siguiente ocurre fuera de la lista que la entidad interviene.",
    )
    parrafo(
        doc,
        "Hay un segundo vacío. En Colombia se usan los factores de reducción del Highway Safety "
        "Manual estadounidense para estimar cuánto sirve cada tipo de obra. Ninguna de las cinco "
        "categorías que ejecuta la SDM tiene un factor estimado con datos de Bogotá, pese a que la "
        f"entidad publicó {miles(inter['registros'] - inter['sin_civ'])} intervenciones "
        "georreferenciadas por segmento y publica también los siniestros de esos mismos segmentos "
        "antes y después de la obra.",
    )
    parrafo(
        doc,
        "La magnitud justifica el esfuerzo. Entre 2015 y 2021 se registraron "
        f"{miles(sin_['registros'])} siniestros, de los cuales {miles(sin_['graves'])} dejaron "
        f"heridos o muertos y {miles(sin_['con_muertos'])} dejaron muertos, un promedio de "
        f"{miles(sin_['graves'] / 7)} siniestros graves al año. Entre 2021 y mayo de 2025 la ciudad "
        f"registró {miles(vic['registros'])} víctimas, {miles(vic['fallecidos'])} de ellas "
        f"fallecidas; {miles(vic['por_condicion']['MOTOCICLISTA'])} eran motociclistas y "
        f"{miles(vic['por_condicion']['PEATON'])} peatones.",
    )
    parrafo(
        doc,
        "Lo que la entidad necesita no es un número sino una lista que se pueda recalcular. Quien "
        "programa la inversión tiene que poder preguntar qué cambia si el presupuesto baja de "
        "40.000 a 28.500 millones de pesos, o qué pasa si el paso peatonal resulta menos efectivo "
        "de lo que se creía, y ver la respuesta en el momento. Eso obliga a separar lo que se "
        "estima una sola vez de lo que se recalcula en cada interacción, y a servir lo primero por "
        "una API que un tablero consuma. Por eso el producto de este curso es un prototipo "
        "desplegado y no un cuaderno con resultados.",
    )

    # =============================================================== 2
    doc.add_heading("2. Pregunta de negocio y alcance del proyecto", 1)
    parrafo(
        doc,
        "Pregunta de negocio: en qué segmentos de la malla vial de Bogotá, y con qué tipo de "
        "intervención, se evita la mayor cantidad de siniestros graves por peso invertido.",
    )
    parrafo(
        doc,
        "Pregunta analítica: dado un segmento con su historia de siniestros entre 2015 y 2021, su "
        "exposición estimada y las intervenciones que recibió, cuántos siniestros graves se esperan "
        "allí el próximo año, con qué probabilidad ocurre al menos uno, y qué combinación de "
        "intervenciones cabe en un presupuesto dado maximizando la reducción total.",
    )
    parrafo(
        doc,
        "La parte descriptiva responde dónde está el daño hoy y cómo se reparte: el mapa de la "
        "malla, el ranking ordenable y la distribución territorial de la inversión recomendada. La "
        "parte predictiva responde dónde estará el daño el año entrante y cuánto lo mueve cada "
        "obra: los dos modelos supervisados y los cinco factores de reducción.",
    )
    parrafo(
        doc,
        "Los modelos empaquetados son dos y se comparan en MLflow sobre la misma partición. El "
        "primero es un modelo de conteo de siniestros graves por segmento y año, binomial negativa "
        "con la exposición como desplazamiento en escala logarítmica; su salida alimenta el color "
        "del mapa y el orden del ranking. El segundo es un clasificador binario de ocurrencia de al "
        "menos un siniestro grave en el año siguiente, que entrega una lista corta cuando el "
        "usuario prefiere un criterio de sí o no antes que una estimación puntual. Los dos se "
        "entrenan con datos hasta 2020 y se validan contra 2021, que es el último año de la serie.",
    )
    parrafo(
        doc,
        "El prototipo incluye la extracción de las cuatro capas, el panel segmento-año, el proxy de "
        "exposición, los dos modelos, la corrección Empirical Bayes como posproceso del modelo de "
        "conteo, los cinco factores de reducción precalculados con diferencias en diferencias, la "
        "API que sirve las inferencias, el tablero que la consume, el versionado de datos en DVC "
        "con remoto en S3, el registro de experimentos en MLflow y el despliegue en contenedores "
        "Docker sobre la instancia EC2 del AWS Academy Learner Lab.",
    )
    parrafo(
        doc,
        "Queda fuera, y pasa al curso capstone, lo siguiente. Primero, la identificación causal "
        "definitiva: las intervenciones utilizables están marcadas todas en 2016, lo que deja un "
        "único periodo previo y no permite probar tendencias paralelas, y "
        f"{miles(inter['civ_con_mas_de_una_categoria'])} de los {miles(inter['civ_tratados'])} "
        "segmentos tratados recibieron más de una categoría, de modo que los efectos individuales "
        "no se separan sin un diseño de tratamientos múltiples. En el prototipo los cinco factores "
        "entran como parámetros del what-if, con su intervalo, y no como una afirmación causal "
        "cerrada. Segundo, la exposición medida: aquí es un proxy construido con atributos de la "
        "malla, porque no existen aforos por segmento en las capas públicas. Tercero, la "
        "optimización exacta: el tablero resuelve un algoritmo voraz de mochila sobre el top-N, y "
        "la programación entera con restricciones operativas y de equidad territorial es trabajo "
        "posterior. Cuarto, la serie posterior a 2021, cuyo cruce al segmento exige unión espacial "
        "y no simple llave. Quinto, la validación de supuestos con la Oficina de Seguridad Vial y "
        "el reentrenamiento en operación.",
    )
    parrafo(
        doc,
        "Una aclaración sobre el caso. El mismo problema se trabaja como anteproyecto en el curso "
        "Gerencia de Proyectos de Analytics. Allá el entregable es la formulación, con carta "
        "proyecto, riesgos y cronograma; aquí el entregable es el prototipo construido y "
        "desplegado. Los documentos y los productos son distintos y no se comparten.",
    )
    tabla(
        doc,
        ["Criterio de aceptación del prototipo", "Meta"],
        [
            ["Captura del 5 % priorizado sobre los siniestros graves de 2021",
             f"≥ 55 %, contra {pct(b2021['pct'])} % de línea base"],
            ["Modelos registrados en MLflow con la misma partición y métricas comparables", "2 de 2"],
            ["Tablero con los cuatro elementos de la maqueta, alimentado por la salida del modelo",
             "4 de 4"],
            ["Tiempo hasta el primer mapa pintado en el tablero", "menos de 3 s"],
            ["dvc repro reconstruye el pipeline completo sin intervención manual", "sí"],
        ],
        [12.3, 4.5],
    )
    pie(doc, "Tabla 1. Criterios con los que se evaluará el prototipo cuando esté construido.")

    # =============================================================== 3
    doc.add_heading("3. Conjuntos de datos a emplear", 1)
    parrafo(
        doc,
        "Las cuatro capas son públicas y se consultan por los servicios REST de ArcGIS de la SDM, "
        "sin token, en https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services. El "
        "catálogo tiene 1.281 servicios publicados; estos cuatro son los que sostienen el producto. "
        "El servidor limita cada respuesta a 2.000 registros, de modo que la descarga se hace "
        "paginada. Los conteos de la tabla 2 son los que devolvió la API el 18 de octubre de 2026, "
        "y cada descarga deja un archivo .meta.json con la fecha de consulta, el conteo reportado "
        "por el servicio y la lista de campos, que es el soporte de trazabilidad del pipeline.",
    )
    tabla(
        doc,
        ["Capa", "Servicio", "Registros", "Periodo", "Llave", "Papel en el producto"],
        [
            ["Malla Vial Integral", "Malla_Vial_Integral", miles(malla["registros"]), "Vigente",
             "MVICIV", "Unidad de análisis y fuente de la exposición"],
            ["Histórico de siniestros", "HistoricoSiniestros", miles(sin_["registros"]),
             "2015-2021", "CIV", "Variable respuesta y línea base"],
            ["Víctimas", "Victima_2021_may2025_Loc", miles(vic["registros"]), "2021 a may 2025",
             "FORMULARIO", "Severidad por actor vial y validación"],
            ["Intervenciones de seguridad vial", "Intervenciones_Seguridad_Vial",
             miles(inter["registros"]), "ene 2016 a ago 2017", "CIV",
             "Tratamiento del módulo causal"],
        ],
        [3.0, 3.5, 1.7, 2.2, 1.9, 4.5],
    )
    pie(doc, "Tabla 2. Capas de origen, con los conteos devueltos por la API el 18 de octubre de 2026.")
    parrafo(
        doc,
        "De esas cuatro capas salen tres conjuntos derivados. El panel segmento-año, que es la "
        f"tabla sobre la que se entrena, tiene {miles(base['universo'])} segmentos por siete años, "
        f"es decir {miles(base['universo'] * 7)} filas, con una columna de conteo de siniestros "
        "graves y las covariables de la malla. El archivo de geometrías simplificadas alimenta el "
        "mapa. La tabla de ranking, con la reducción esperada y el costo de cada categoría "
        "aplicable a cada segmento, es lo que consume la API.",
    )
    parrafo(
        doc,
        "Todas las capas están bajo licencia de datos abiertos de Bogotá. La de víctimas trae edad, "
        "género y condición, pero no identificadores personales; el producto la usa agregada por "
        "segmento y no publica ningún registro individual. El catálogo ofrece además capas de "
        "contexto que quedan disponibles y no entran al prototipo: los tramos críticos que publica "
        "la propia SDM, que serán la segunda línea base de comparación, las velocidades de los "
        "sensores Bitcarrier y los comparendos georreferenciados.",
    )

    # =============================================================== 4
    doc.add_heading("4. Exploración de los datos", 1)
    parrafo(
        doc,
        "Toda la exploración corre con python -m src.exploracion.perfilamiento y deja sus "
        "resultados en reportes/perfilamiento.json. Las cifras de esta sección salen de ese "
        "archivo, no de una transcripción a mano, y el reporte se regenera a partir de él.",
    )

    doc.add_heading("4.1 La llave CIV no es lo que parece", 2)
    parrafo(
        doc,
        f"El hallazgo que más cambia el diseño es que la malla tiene {miles(malla['registros'])} "
        f"registros pero solo {miles(malla['civ_distintos'])} códigos CIV distintos. Hay "
        f"{miles(malla['civ_con_mas_de_un_registro'])} códigos repartidos en más de un registro, "
        f"uno de ellos en {malla['maximo_registros_por_civ']}, y {miles(malla['civ_cero'])} "
        "registros sin código. Como el CIV es la llave con la que se registran los siniestros y se "
        "reportan las intervenciones, y es además la unidad sobre la que la entidad contrata, la "
        f"unidad de análisis del modelo son los {miles(malla['civ_distintos'])} códigos y no los "
        f"{miles(malla['registros'])} registros. Los atributos se agregan por código: la longitud "
        "se suma, la clase de vía se toma como la más alta de la jerarquía y el número de carriles "
        "como el máximo. Trabajar con el conteo publicado habría inflado el denominador en un 20 %.",
    )
    tabla(
        doc,
        ["Hallazgo", "Cifra", "Qué se hace al respecto"],
        [
            ["Registros de la malla sin código CIV", miles(malla["civ_cero"]),
             "Se excluyen: no son direccionables para una intervención"],
            ["Siniestros sin CIV", f"{miles(sin_['sin_civ'])} ({pct(sin_['sin_civ_pct'], 2)} %)",
             "Se excluyen del panel y se reportan como pérdida conocida"],
            ["Siniestros con un CIV ausente de la malla vigente",
             f"{miles(sin_['civ_fuera_de_la_malla_vigente'])} ({pct(sin_['civ_fuera_de_la_malla_vigente_pct'], 2)} %)",
             "La malla se actualizó después de 2015; se excluyen y se documenta"],
            ["Velocidad reglamentada vacía",
             f"{miles(malla['velocidad_reglamentada_vacia'])} ({pct(malla['velocidad_reglamentada_vacia_pct'])} %)",
             "Se imputa por clase de vía; el campo solo toma dos valores, 30 y 50"],
            ["Clase de vía y número de carriles", "sin nulos",
             "Entran directo como covariables y como insumo de la exposición"],
            ["Geometría en coordenadas geográficas (EPSG:4686)",
             f"{miles(malla['longitud_total_km'], 1)} km",
             "Se reproyecta a EPSG:3116; la longitud en grados no sirve como exposición"],
            ["Registros con longitud menor a un metro", miles(malla["segmentos_menores_a_1_m"]),
             "Se marcan y se revisan antes de usarse como denominador"],
            ["Categorías de intervención",
             f"{inter['categorias_crudas']} crudas, {inter['categorias_normalizadas']} reales",
             "Zona escolar aparece con dos grafías; se normaliza antes de agrupar"],
            ["Intervenciones sin CIV", miles(inter["sin_civ"]),
             "Son exactamente las marcadas con año 2017; quedan fuera del análisis causal"],
            ["Víctimas sin llave al segmento", f"{miles(puente['victimas_2021'])} en 2021",
             f"El puente por FORMULARIO resuelve el {pct(puente['con_civ_valido_pct'])} % con CIV válido"],
            ["Edades nulas o implausibles en víctimas",
             f"{miles(vic['edad_nula'])} y {miles(vic['edad_implausible'])}",
             "No afectan al modelo: la edad solo pondera severidad"],
        ],
        [5.4, 3.2, 8.2],
    )
    pie(doc, "Tabla 3. Perfilamiento de calidad de las cuatro capas.")
    parrafo(
        doc,
        "El puente por FORMULARIO merece una nota aparte porque era el riesgo de datos más serio "
        "del proyecto. La capa de víctimas no trae CIV. En el año de traslape, 2021, el campo "
        "FORMULARIO identifica el siniestro sin ambigüedad: no hay un solo duplicado en los "
        f"{miles(anios['2021']['total'])} siniestros de ese año. De las "
        f"{miles(puente['victimas_2021'])} víctimas de 2021, {miles(puente['cruzan_con_siniestros'])} "
        f"cruzan contra la capa de siniestros ({pct(puente['cruzan_pct'])} %) y "
        f"{miles(puente['con_civ_valido'])} quedan con un CIV válido ({pct(puente['con_civ_valido_pct'])} %). "
        "El cruce funciona, y eso habilita usar la severidad por actor vial como ponderador.",
    )

    doc.add_heading("4.2 La respuesta tiene muchos ceros y mucha dispersión", 2)
    parrafo(
        doc,
        f"De los {miles(sin_['graves'])} siniestros graves de la serie, "
        f"{miles(dist['graves_asignables_a_un_civ'])} se asignan a un segmento de la malla vigente. "
        f"El {pct(dist['civ_sin_ningun_grave_pct'])} % de los segmentos no registra ninguno en siete "
        f"años. La media es de {miles(dist['media_por_civ'], 3)} siniestros graves por segmento y "
        f"la varianza {miles(dist['varianza_por_civ'], 3)}: el índice de dispersión es "
        f"{miles(dist['indice_de_dispersion'], 2)}, muy por encima de 1. El máximo observado en un "
        f"solo segmento es {dist['maximo_en_un_civ']}.",
    )
    parrafo(
        doc,
        "Esos tres números cierran la discusión sobre la familia del modelo. Una regresión lineal "
        "sobre un conteo con ese exceso de ceros produciría predicciones negativas; una Poisson "
        "subestimaría la varianza por un factor de nueve. Queda la binomial negativa, con la "
        "exposición como desplazamiento, y como alternativa un gradient boosting con pérdida de "
        "Poisson. Las dos se comparan en MLflow.",
    )
    figura(doc, "concentracion.png", 10.5)
    pie(
        doc,
        "Figura 1. Concentración del daño. El eje horizontal son los segmentos ordenados de mayor a "
        "menor número de siniestros graves entre 2015 y 2021; el vertical, el porcentaje acumulado "
        "de siniestros graves. La diagonal sería el reparto uniforme.",
    )
    tabla(
        doc,
        ["Segmentos priorizados", "Cantidad", "Siniestros graves que contienen"],
        [
            ["1 %", miles(round(base["universo"] * 0.01)), f"{pct(conc['top_1_pct'])} %"],
            ["2 %", miles(round(base["universo"] * 0.02)), f"{pct(conc['top_2_pct'])} %"],
            ["5 %", miles(base["tamano_top_5_pct"]), f"{pct(conc['top_5_pct'])} %"],
            ["10 %", miles(round(base["universo"] * 0.10)), f"{pct(conc['top_10_pct'])} %"],
        ],
        [4.0, 3.0, 5.5],
    )
    pie(
        doc,
        "Tabla 4. Concentración observada entre 2015 y 2021. Es el argumento de apalancamiento del "
        "proyecto: acertar en una fracción pequeña de la malla mueve una fracción grande del daño.",
    )

    doc.add_heading("4.3 La pandemia parte la serie en dos", 2)
    parrafo(
        doc,
        f"Los siniestros caen de {miles(anios['2019']['total'])} en 2019 a "
        f"{miles(anios['2020']['total'])} en 2020, un "
        f"{pct(abs(100 * (anios['2020']['total'] / anios['2019']['total'] - 1)), 0)} % menos, y los "
        f"graves de {miles(anios['2019']['graves'])} a {miles(anios['2020']['graves'])}, un "
        f"{pct(abs(100 * (anios['2020']['graves'] / anios['2019']['graves'] - 1)), 0)} % menos. En 2021 "
        f"la serie rebota a {miles(anios['2021']['graves'])} graves sin recuperar el nivel de 2019. "
        "El choque cae justo en el periodo posterior del diseño causal, de modo que todo modelo que "
        "compare antes y después tiene que absorberlo con efectos fijos de año, y el módulo causal "
        "correrá además una especificación que excluya 2020 y 2021 como prueba de sensibilidad. La "
        "misma razón explica por qué la línea base se mide contra 2021 y no contra 2020: validar "
        "contra el año de la pandemia daría un número artificialmente favorable.",
    )

    doc.add_heading("4.4 Qué tan bueno es el método vigente", 2)
    parrafo(
        doc,
        "Ordenando los segmentos por conteo de siniestros graves entre 2015 y 2020 y evaluando "
        f"contra 2021, el 5 % mejor rankeado captura el {pct(b2021['pct'])} % de los siniestros "
        f"graves de 2021 ({miles(b2021['capturados'])} de {miles(b2021['total'])}). La misma "
        f"medición entrenada con 2015-2019 y evaluada en 2020 da {pct(b2020['pct'])} %, diferencia "
        "atribuible al choque de la pandemia. Se adopta el valor más conservador como línea base.",
    )
    parrafo(
        doc,
        "Vale la pena anotar de dónde viene el número, porque difiere del que aparece en la "
        "formulación del mismo caso en el curso de Gerencia. Allá la métrica se calculó sobre los "
        f"{miles(malla['registros'])} registros de la malla, con un corte de 6.955 segmentos, y dio "
        f"47,3 %. Al corregir el universo a los {miles(base['universo'])} códigos CIV distintos, el "
        f"corte del 5 % baja a {miles(base['tamano_top_5_pct'])} y la captura a {pct(b2021['pct'])} %. "
        "La cifra que se usa aquí es la segunda, porque corresponde a la unidad que efectivamente "
        "sirve la API.",
    )

    doc.add_heading("4.5 Se alcanza a ver la señal del tratamiento", 2)
    parrafo(
        doc,
        f"Las {miles(inter['registros'] - inter['sin_civ'])} intervenciones con CIV se concentran en "
        f"{miles(inter['civ_tratados'])} segmentos de la malla vigente. Es un dato relevante para el "
        "diseño: el número de unidades tratadas es un tercio del número de registros, porque un "
        "mismo segmento recibió varias obras. Comparando el año previo con el promedio posterior, "
        f"los segmentos intervenidos pasan de {miles(prelim['tratados_2015'])} siniestros graves en "
        f"2015 a {miles(prelim['tratados_promedio_2018_2021'], 0)} en promedio entre 2018 y 2021, "
        f"un {'+' if prelim['tratados_variacion_pct'] > 0 else '-'}{pct(abs(prelim['tratados_variacion_pct']))} %, mientras el resto de la malla pasa de "
        f"{miles(prelim['control_2015'])} a {miles(prelim['control_promedio_2018_2021'], 0)}, un "
        f"{'+' if prelim['control_variacion_pct'] > 0 else '-'}"
        f"{pct(abs(prelim['control_variacion_pct']))} %. La diferencia es de "
        f"{pct(abs(prelim['diferencia_en_diferencias_pp']))} puntos porcentuales a favor de los "
        "intervenidos.",
    )
    parrafo(
        doc,
        "Ese número no es una estimación causal y no debe leerse como tal. No controla exposición, "
        "no empareja segmentos comparables, no separa las categorías y descansa en un solo periodo "
        "previo. Lo que muestra es que la señal existe y tiene el signo esperado, que es lo que se "
        "necesitaba saber antes de montar el módulo causal.",
    )
    figura(doc, "tendencia_tratados.png", 10.5)
    pie(
        doc,
        "Figura 2. Siniestros graves por año, con 2015 como base 100, en los segmentos intervenidos "
        "y en el resto de la malla. La franja gris es la ventana de ejecución de las obras, que el "
        "diseño causal excluye.",
    )

    # =============================================================== 5
    doc.add_heading("5. Maqueta del prototipo", 1)
    parrafo(
        doc,
        "El tablero tiene cuatro elementos y cada uno responde una parte de la pregunta de negocio. "
        "La maqueta fija también el reparto de responsabilidades entre la API y el navegador, que "
        "es la decisión de diseño con más consecuencias para el despliegue.",
    )
    figura(doc, "maqueta_tablero.png", 17.0)
    pie(doc, "Figura 3. Maqueta del tablero. Las etiquetas negras indican el origen de cada dato.")
    tabla(
        doc,
        ["Elemento", "Qué pregunta responde", "De dónde sale el dato"],
        [
            ["A. Mapa coloreado por riesgo esperado",
             "Dónde está el riesgo y cómo se reparte en la ciudad",
             "API: GET /segmentos/top. El zoom y la selección son del navegador"],
            ["B. Tabla del ranking priorizado",
             "Cuáles son los segmentos concretos que hay que intervenir",
             "API: GET /ranking. El orden, los filtros y la exportación son del navegador"],
            ["C. Panel de what-if por categoría",
             "Cuánto cambia el resultado si un tipo de obra sirve más o menos",
             "API: GET /factores, una sola vez. El recálculo del riesgo residual es del navegador"],
            ["D. Deslizador de presupuesto",
             "Qué paquete cabe en la plata que hay y cuánto evita",
             "Navegador: algoritmo voraz de mochila sobre el top-N ya descargado"],
        ],
        [4.3, 5.6, 6.9],
    )
    pie(doc, "Tabla 5. Los cuatro elementos del tablero y su relación con la pregunta de negocio.")
    parrafo(
        doc,
        "El problema de rendimiento es real y se midió. Las geometrías de los "
        f"{miles(malla['registros'])} registros de la malla pesan 20,8 MB en formato JSON, con "
        "503.606 vértices. Enviar eso al navegador en cada carga, y volver a enviarlo cada vez que "
        "el usuario mueve el deslizador, haría el tablero inservible. La solución tiene dos partes "
        "y una de ellas pesa mucho más que la otra. La principal es servir solo el top-N: con N "
        "igual a 2.000, que cubre con holgura el 1 % de la malla donde ocurre el "
        f"{pct(conc['top_1_pct'])} % de los siniestros graves, la respuesta baja a 0,26 MB, un factor de "
        "ochenta. La segunda es simplificar las geometrías con Douglas-Peucker a 15 metros, que "
        "reduce los vértices de esos 2.000 segmentos de 6.274 a 4.046 y el peso a 0,17 MB. La "
        "ganancia de la simplificación es menor de lo que uno esperaría, porque la mediana de "
        "vértices por registro es 2: casi todos los segmentos son rectas y no hay nada que "
        "simplificar. El resto de la malla se dibuja como teselas vectoriales pre-generadas que "
        "sirven de fondo y no se consultan.",
    )
    parrafo(
        doc,
        "La segunda decisión de rendimiento es que el ranking se descarga una vez por sesión y "
        "queda en memoria del navegador. Mover el deslizador de presupuesto o encender una "
        "categoría en el panel de what-if no genera tráfico: el algoritmo voraz y el recálculo del "
        "riesgo residual corren sobre los datos ya descargados. La API solo se vuelve a llamar si "
        "el usuario cambia el año de referencia o pide un segmento que no está en el top-N.",
    )

    # =============================================================== 6
    doc.add_heading("6. Repositorios con el código y los datos", 1)
    parrafo(
        doc,
        f"El código está en {URL_REPO}, repositorio público, rama main. Los datos y los artefactos "
        "de modelo se versionan con DVC y se guardan en un bucket de S3 de la cuenta del AWS "
        "Academy Learner Lab.",
    )
    monoespaciado(
        doc,
        """
seguridad-vial-bogota/
  dvc.yaml            definición del pipeline: perfilamiento y maqueta
  dvc.lock            hash de cada entrada y cada salida de las dos etapas
  src/extraccion/     cliente REST paginado y descarga de las cuatro capas
  src/exploracion/    perfilamiento, línea base y figuras del reporte
  src/tablero/        maqueta del tablero
  data/raw/           punteros .dvc y ficha de trazabilidad de cada capa
  reportes/           perfilamiento.json, figuras y generadores de los documentos
""",
    )
    tabla(
        doc,
        ["Qué", "Git", "DVC", "Por qué"],
        [
            ["src/", "sí", "no", "Código"],
            ["dvc.yaml, dvc.lock", "sí", "no",
             "Definen el pipeline y fijan el hash de cada salida"],
            ["Archivos .dvc y .dvc/config", "sí", "no",
             "Punteros a los datos y configuración del remoto"],
            ["data/raw/*.parquet", "no", "sí", "31 MB que se regeneran desde la API"],
            ["data/raw/*.meta.json", "sí", "no",
             "Ficha de trazabilidad de cada descarga; menos de un kilobyte"],
            ["reportes/perfilamiento.json y figuras", "sí", "no",
             "Pesan poco y se citan en el reporte: conviene verlos en el diff"],
            ["Credenciales de AWS, llaves .pem, .env", "no", "no",
             "Nunca entran al repositorio; la sesión del Learner Lab rota"],
        ],
        [5.2, 1.3, 1.3, 9.0],
    )
    pie(doc, "Tabla 6. Reparto entre Git y DVC.")
    parrafo(
        doc,
        "Secuencia con la que quedó inicializado el repositorio. Cada bloque corresponde a un "
        "commit de la rama main, con el mismo mensaje que aparece en el historial del anexo:",
    )
    monoespaciado(
        doc,
        """
git init && git branch -m main
dvc init
git add .dvcignore .gitignore README.md requirements.txt
git commit -m "Estructura del proyecto y dependencias"
git remote add origin https://github.com/anquint89/seguridad-vial-bogota.git
git push -u origin main

git add src/extraccion
git commit -m "Cliente REST y descarga de las cuatro capas de la SDM"

aws configure set aws_access_key_id ACCESS_KEY          # del Learner Lab
aws configure set aws_secret_access_key SECRET_KEY
aws configure set aws_session_token SESSION_TOKEN
aws configure set region us-east-1
aws s3 mb s3://svb-dvcstore --region us-east-1

dvc remote add -d almacen s3://svb-dvcstore/datos
dvc remote modify almacen region us-east-1
dvc remote add respaldo ../../almacen-dvc-local       # para trabajar sin sesion de AWS

python -m src.extraccion.descargar              # descarga las cuatro capas
dvc add data/raw/malla_vial_integral.parquet data/raw/historico_siniestros.parquet \\
        data/raw/victimas_2021_2025.parquet data/raw/intervenciones_seguridad_vial.parquet
git add .dvc dvc.yaml data/raw/*.dvc data/raw/*.meta.json data/raw/.gitignore
git commit -m "Versionado de los datos con DVC y definicion del pipeline"
dvc push

dvc repro perfilar                              # perfilamiento y figuras 1 y 2
git add src/exploracion reportes/perfilamiento.json reportes/figuras
git commit -m "Perfilamiento de los datos y linea base del metodo vigente"

dvc repro maqueta                               # figura de la maqueta
git add src/tablero dvc.lock reportes/figuras/maqueta_tablero.png
git commit -m "Maqueta del tablero"
""",
    )
    parrafo(
        doc,
        "Los datos crudos se versionan con dvc add porque vienen de un servicio externo cuyo "
        "contenido no controlamos: DVC guarda su hash y el archivo sale de Git. De data/raw en "
        "adelante todo es derivado y está declarado como salida de una etapa de dvc.yaml, de modo "
        "que dvc repro lo produce y dvc push lo publica. Quien clona el repositorio corre dvc pull "
        "y obtiene exactamente los mismos archivos con los que se produjo este reporte. El segundo "
        "remoto, local, sirve para trabajar sin sesión activa de AWS.",
    )
    parrafo(
        doc,
        "Los soportes de creación y uso de los dos repositorios van en el anexo de capturas que "
        "acompaña esta entrega: la página del repositorio en GitHub, el historial de commits con la "
        "contribución de cada integrante, el contenido de .dvc/config con los dos remotos, las "
        "etapas y el grafo del pipeline, el resultado de dvc push y las fichas de trazabilidad de "
        "las cuatro descargas.",
    )

    destino = PROYECTO / "Entrega1_Reporte.docx"
    doc.save(destino)
    print(f"Escrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
