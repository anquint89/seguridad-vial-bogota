"""Genera el documento de tareas por integrante para la Entrega 1.

    python reportes/generar_tareas.py
"""

from __future__ import annotations

from pathlib import Path

from generar_reporte import PROYECTO, documento, monoespaciado, parrafo, pie, tabla

from docx.shared import Pt

URL_REPO = "https://github.com/anquint89/seguridad-vial-bogota"
ANCHOS = [4.6, 4.4, 5.6, 2.2]
ENCABEZADOS = ["Tarea", "Archivos que toca", "Cuándo se da por terminada", "Fecha"]


def titulo(doc, texto: str, tam: float = 13) -> None:
    p = doc.add_paragraph(texto)
    p.runs[0].font.size = Pt(tam)
    p.runs[0].font.bold = True
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(6)


def main() -> int:
    doc = documento()

    titulo(doc, "Tareas por integrante para la Entrega 1")
    sub = doc.add_paragraph(
        "Despliegue de Soluciones Analíticas (MIID-4304). Equipo: Laura Alfonso Botia, "
        "Andrés Quintero Hoyos, Andrés Felipe Rodríguez Ortiz.\n"
        f"Repositorio: {URL_REPO}"
    )
    for corrida in sub.runs:
        corrida.font.size = Pt(9)
    sub.paragraph_format.space_after = Pt(9)

    parrafo(
        doc,
        "La nota 2 del enunciado dice que la nota es individual y que cada miembro debe demostrar "
        "su contribución por dos vías: los commits del repositorio y el reporte de trabajo en "
        "equipo. Este documento reparte lo que falta de modo que cada quien tenga archivos propios, "
        "no se pisen entre sí y su aporte quede en el historial con su nombre.",
    )

    # ------------------------------------------------------------------
    titulo(doc, "1. Cómo queda registrada la contribución", 11.5)
    parrafo(
        doc,
        "Antes del primer commit, cada uno configura su identidad de Git dentro de la carpeta del "
        "repositorio. El correo tiene que ser uno de los registrados en su cuenta de GitHub: si no "
        "coincide, GitHub muestra el commit sin enlace al perfil y la contribución no aparece en la "
        "lista de colaboradores ni en la gráfica de aportes, aunque el trabajo sea suyo. Es el error "
        "que más veces deja a alguien sin rastro visible.",
    )
    monoespaciado(
        doc,
        """
git clone https://github.com/anquint89/seguridad-vial-bogota.git
cd seguridad-vial-bogota
git config user.name  "Nombre Apellido"
git config user.email "el-correo-de-su-cuenta-de-github@ejemplo.com"

python -m venv .venv && .venv\\Scripts\\activate     # en macOS o Linux: source .venv/bin/activate
pip install -r requirements.txt
dvc pull                                           # trae data/ desde el remoto
""",
    )
    parrafo(
        doc,
        "Cada tarea se trabaja en una rama propia y se integra a main con un pull request. Así "
        "queda el autor, la fecha, el diff y la revisión del otro:",
    )
    monoespaciado(
        doc,
        """
git checkout -b laura/prueba-puente-formulario
# ... trabajar ...
git add tests/test_puente_formulario.py
git commit -m "Probar el puente FORMULARIO entre las victimas y los siniestros de 2021"
git push -u origin laura/prueba-puente-formulario
# abrir el pull request en GitHub y pedir revision al companero asignado
""",
    )
    parrafo(
        doc,
        "Tres reglas que nos evitan problemas. Nadie sube trabajo a nombre de otro: cada commit lo "
        "hace quien escribió el código, desde su propia cuenta. Los commits son pequeños y con un "
        "mensaje que diga qué cambió, no «avance» ni «cambios»; un commit por idea. Y nadie edita "
        "archivos que no le corresponden sin avisar, porque el reparto está hecho justamente para "
        "que no haya conflictos de fusión.",
    )

    # ------------------------------------------------------------------
    titulo(doc, "2. Qué falta por hacer, y de quién es cada cosa", 11.5)
    parrafo(
        doc,
        "El reporte de trabajo en equipo de la Entrega 1 asigna a cada integrante unos ítems. Para "
        "que esa asignación corresponda al historial del repositorio, estas son las tareas que cada "
        "uno debe tener commiteadas antes del viernes 24, que es cuando se congela la entrega.",
    )

    titulo(doc, "Laura Alfonso Botia", 10.5)
    tabla(
        doc,
        ENCABEZADOS,
        [
            ["Ampliar el perfilamiento con la caracterización de las víctimas: distribución por "
             "condición, por localidad y por grupo de edad",
             "src/exploracion/perfilamiento.py",
             "Las tres distribuciones aparecen en reportes/perfilamiento.json y "
             "dvc repro perfilar corre sin errores",
             "21 oct"],
            ["Probar el puente FORMULARIO que asigna CIV a las víctimas de 2021",
             "tests/test_puente_formulario.py",
             "La prueba verifica que cruza al menos el 99 % de las víctimas de 2021 y que no hay "
             "FORMULARIO duplicado; el mismo commit añade pytest a requirements.txt",
             "23 oct"],
            ["Cuaderno de revisión visual de calidad: nulos, duplicados, rangos imposibles de edad "
             "y segmentos de longitud cero",
             "notebooks/calidad_datos.ipynb",
             "El cuaderno corre de principio a fin con los datos de dvc pull y queda en el "
             "repositorio con sus salidas",
             "24 oct"],
        ],
        ANCHOS,
    )

    titulo(doc, "Andrés Felipe Rodríguez Ortiz", 10.5)
    tabla(
        doc,
        ENCABEZADOS,
        [
            ["Pruebas del cliente REST: paginación, reintentos y el caso en que el servicio "
             "devuelva menos registros de los que reporta",
             "tests/test_arcgis.py",
             "Las pruebas pasan con pytest sin tocar la red, usando respuestas simuladas",
             "21 oct"],
            ["Probar la puesta en marcha desde un clon limpio y corregir el README con lo que "
             "falte: entorno, requirements, dvc pull y dvc repro",
             "README.md",
             "Un clon nuevo reproduce las dos etapas del pipeline siguiendo solo el README, sin "
             "ningún paso que no esté escrito",
             "23 oct"],
            ["Revisión cruzada del perfilamiento: recalcular por su cuenta tres cifras del "
             "reporte y cotejarlas contra el archivo generado",
             "tests/test_perfilamiento.py",
             "Las tres cifras coinciden con reportes/perfilamiento.json y la prueba pasa con "
             "pytest",
             "24 oct"],
        ],
        ANCHOS,
    )

    titulo(doc, "Andrés Quintero Hoyos", 10.5)
    tabla(
        doc,
        ENCABEZADOS,
        [
            ["Línea base del método vigente, análisis de concentración y señal preliminar del "
             "tratamiento",
             "src/exploracion/perfilamiento.py, figuras 1 y 2",
             "Hecho. Las cifras salen de reportes/perfilamiento.json",
             "22 oct"],
            ["Maqueta del tablero y reparto entre la API y el navegador",
             "src/tablero/maqueta.py",
             "Hecho. La figura se regenera con dvc repro maqueta",
             "22 oct"],
            ["Integración del reporte, el anexo de soportes y el reporte de trabajo en equipo",
             "reportes/",
             "Los cuatro documentos salen de sus generadores y cada cifra viene del perfilamiento",
             "24 oct"],
        ],
        ANCHOS,
    )

    # ------------------------------------------------------------------
    titulo(doc, "3. Cómo verificamos antes de entregar", 11.5)
    parrafo(
        doc,
        "El viernes 24, antes de congelar la entrega, revisamos tres cosas en este orden. Que el historial "
        "muestre commits de los tres, con git shortlog -sn HEAD y con la pestaña de colaboradores "
        "de GitHub. Que el pipeline corra de punta a punta con dvc repro desde un clon limpio. Y "
        "que lo que dice el reporte de trabajo en equipo coincida con quién aparece como autor de "
        "cada archivo, que se consulta así:",
    )
    monoespaciado(
        doc,
        """
git shortlog -sn HEAD                      # commits por autor
git log --format="%an | %s" --no-merges    # autor y mensaje de cada commit
git log --format="%an" -- src/exploracion/ # quien ha tocado una carpeta
""",
    )
    parrafo(
        doc,
        "Si alguna de las tres revisiones no cuadra, se corrige antes de entregar, no después. Una "
        "asignación del reporte que no tenga commits detrás es un problema para quien aparece en "
        "ella, porque la nota es individual.",
    )
    pie(doc, "Fecha de la entrega: domingo 25 de octubre de 2026, 23:59.")

    destino = PROYECTO / "Tareas_por_integrante.docx"
    doc.save(destino)
    print(f"Escrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
