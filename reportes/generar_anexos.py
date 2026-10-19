"""Genera el reporte de trabajo en equipo y el plan de cierre de la Entrega 1.

    python reportes/generar_anexos.py
"""

from __future__ import annotations

from pathlib import Path

from generar_reporte import PROYECTO, documento, monoespaciado, parrafo, pie, tabla

from docx.shared import Pt

URL_REPO = "https://github.com/anquint89/seguridad-vial-bogota"


def encabezado(doc, titulo: str, subtitulo: str) -> None:
    p = doc.add_paragraph(titulo)
    p.runs[0].font.size = Pt(13)
    p.runs[0].font.bold = True
    p.paragraph_format.space_after = Pt(2)
    s = doc.add_paragraph(subtitulo)
    for corrida in s.runs:
        corrida.font.size = Pt(9)
    s.paragraph_format.space_after = Pt(8)


# ------------------------------------------------------- trabajo en equipo


def trabajo_en_equipo() -> Path:
    doc = documento()
    encabezado(
        doc,
        "Reporte de trabajo en equipo. Entrega 1",
        "Despliegue de Soluciones Analíticas (MIID-4304). Equipo: Laura Alfonso Botia, "
        "Andrés Quintero Hoyos, Andrés Felipe Rodríguez Ortiz. 25 de octubre de 2026.\n"
        f"Repositorio: {URL_REPO}",
    )
    parrafo(
        doc,
        "La nota del curso es individual y se sustenta en los commits, así que el trabajo se "
        "repartió por módulos con dueño único y archivos que no se solapan. Cada quien es "
        "responsable de su carpeta en el repositorio y revisa la de otro antes de integrar. Nos "
        "reunimos los martes a las 19:00 por videoconferencia para revisar avances y los viernes "
        "para cerrar lo de la semana.",
    )
    tabla(
        doc,
        ["Ítem de trabajo", "Responsable", "Carpeta o archivo", "Revisa", "Fecha"],
        [
            ["Cliente REST paginado de los servicios de la SDM y descarga de las cuatro capas, "
             "con ficha de trazabilidad por capa",
             "Andrés Felipe Rodríguez", "src/extraccion/", "Laura Alfonso", "19 oct"],
            ["Perfilamiento de calidad: completitud de CIV, nulos, duplicados, categorías con "
             "doble grafía, reproyección y longitudes",
             "Laura Alfonso", "src/exploracion/", "Andrés Quintero", "21 oct"],
            ["Réplica de la línea base del método vigente y análisis de concentración, "
             "dispersión y señal preliminar del tratamiento",
             "Andrés Quintero", "src/exploracion/, figuras 1 y 2", "Andrés F. Rodríguez", "22 oct"],
            ["Prueba del puente FORMULARIO entre víctimas y siniestros en el año de traslape",
             "Laura Alfonso", "src/exploracion/perfilamiento.py", "Andrés F. Rodríguez", "21 oct"],
            ["Maqueta del tablero y decisión de reparto entre la API y el navegador, incluida la "
             "medición del peso de las geometrías",
             "Andrés Quintero", "src/tablero/maqueta.py", "Laura Alfonso", "22 oct"],
            ["Pruebas del cliente REST: paginación, reintentos y reproyección de la geometría",
             "Andrés Felipe Rodríguez", "src/extraccion/", "Andrés Quintero", "23 oct"],
            ["Inicialización de Git y DVC, remoto en S3 del Learner Lab y primera versión de los "
             "datos versionada",
             "Andrés Quintero", ".dvc/, dvc.yaml, dvc.lock", "Andrés F. Rodríguez", "23 oct"],
            ["Redacción e integración del reporte y revisión cruzada de cifras contra el "
             "perfilamiento",
             "Los tres", "reportes/", "Los tres", "24 oct"],
        ],
        [6.6, 2.8, 3.6, 2.4, 1.4],
    )
    pie(doc, "Tabla 1. Reparto de ítems de trabajo de la Entrega 1, con revisor y fecha de corte.")
    parrafo(
        doc,
        "Cómo se verifica la contribución. Cada ítem entra al repositorio en commits del "
        "responsable, con el nombre del módulo en el mensaje. El historial de la rama main permite "
        "atribuir cada archivo a quien lo escribió, y el comando git shortlog -sn resume los "
        "commits por autor. No se usan commits colectivos ni se sube trabajo a nombre de otro: si "
        "alguien ayuda en el módulo de otro, el aporte queda como commit propio sobre esa rama.",
    )
    parrafo(
        doc,
        "Fronteras de trabajo acordadas. Laura Alfonso responde por el perfilamiento y la calidad "
        "de los datos; Andrés Felipe Rodríguez, por la extracción y sus pruebas; Andrés Quintero, "
        "por la línea base, la maqueta y la integración de los reportes. Cada quien "
        "trabaja sobre sus carpetas y nadie edita las de otro sin avisar, de modo que el historial "
        "del repositorio muestre con claridad quién hizo qué.",
    )
    parrafo(
        doc,
        "Riesgo del reparto y cómo lo manejamos. La exploración depende de que las cuatro capas "
        "estén descargadas y versionadas, que es la tarea con más dependencias hacia adelante; por "
        "eso quedó de primera en la semana y con un dueño único. Si alguna capa hubiera fallado, el "
        "plan era trabajar con la última copia versionada en DVC y documentar la diferencia, para "
        "que ningún retraso de la extracción detuviera a los otros dos frentes.",
    )
    destino = PROYECTO / "Entrega1_Trabajo_en_equipo.docx"
    doc.save(destino)
    return destino


# -------------------------------------------------------------- plan semanal


def plan_semanal() -> Path:
    doc = documento()
    encabezado(
        doc,
        "Plan de cierre de la Entrega 1",
        "Despliegue de Soluciones Analíticas (MIID-4304). Equipo: Laura Alfonso Botia, "
        "Andrés Quintero Hoyos, Andrés Felipe Rodríguez Ortiz.",
    )
    parrafo(
        doc,
        "La semana 3 del cronograma del curso pide tres cosas: repositorios de Git y de DVC en uso, "
        "exploración de los datos versionando código y datos, e iterar la maqueta con lo que vaya "
        "saliendo de los datos. El plan de abajo reparte esos tres frentes día por día hasta la "
        "fecha de entrega, el domingo 25 de octubre a las 23:59.",
    )
    tabla(
        doc,
        ["Día", "Qué debe quedar hecho", "Dueño"],
        [
            ["Lun 19 - mar 20",
             "Repositorio en GitHub con la estructura del proyecto, DVC inicializado y las cuatro "
             "capas de la SDM descargadas y versionadas. Perfilamiento reproducible corriendo.",
             "Andrés Quintero"],
            ["Mar 20 - mié 21",
             "Ampliación del perfilamiento con la caracterización de las víctimas. Pruebas del "
             "cliente REST de extracción.",
             "Laura Alfonso, Andrés Felipe Rodríguez"],
            ["Mié 21 - jue 22",
             "Línea base del método vigente, análisis de concentración y señal preliminar del "
             "tratamiento. Maqueta del tablero cerrada con el reparto entre la API y el navegador.",
             "Andrés Quintero"],
            ["Jue 22 - vie 23",
             "Cruce del puente FORMULARIO verificado sobre el año de traslape. Pruebas del cliente "
             "REST pasando sin tocar la red. Pipeline reproducido desde un clon limpio.",
             "Laura Alfonso, Andrés Felipe Rodríguez"],
            ["Vie 23",
             "Corte técnico: nadie sube código nuevo después de este punto. Revisión cruzada del "
             "historial de commits y del pipeline desde un clon limpio.",
             "Los tres"],
            ["Sáb 24",
             "Integración del reporte, el anexo de soportes y el reporte de trabajo en equipo. "
             "Cuaderno de revisión visual de calidad.",
             "Andrés Quintero, Laura Alfonso"],
            ["Dom 25",
             "Lectura final de los documentos y envío. El domingo queda libre para la Guía 2 de "
             "Gerencia de Proyectos, que vence el mismo día.",
             "Los tres"],
        ],
        [2.6, 10.2, 3.8],
    )
    pie(doc, "Tabla 1. Reparto día por día hasta la fecha de entrega.")
    parrafo(
        doc,
        "La restricción que manda esta semana es que la Entrega 1 de este curso y la Guía 2 de "
        "Gerencia de Proyectos vencen el mismo domingo. Por eso el corte técnico es el viernes 23 y "
        "no el domingo: deja el sábado para escribir y el domingo para la guía del otro curso. El "
        "caso de negocio es el mismo en ambos, pero los productos son disjuntos y ningún entregable "
        "se reutiliza entre ellos.",
    )
    parrafo(doc, "Criterios con los que damos por cerrada la entrega:")
    monoespaciado(
        doc,
        """
dvc repro corre de punta a punta sin intervención manual
los commits de la semana están en main y cada integrante tiene los suyos
lo que se promete en el plan está en el repositorio, no en el computador de alguien
cada cifra que se cita en un reporte sale de un archivo generado, no de una transcripción
""",
        8.5,
    )
    destino = PROYECTO / "Entrega1_Plan_semanal.docx"
    doc.save(destino)
    return destino


if __name__ == "__main__":
    for ruta in (trabajo_en_equipo(), plan_semanal()):
        print(f"Escrito {ruta}")
