"""Genera el anexo de soportes de los repositorios Git y DVC.

Recoge la salida real de los comandos del repositorio y las capturas de la
página de GitHub, y las deja en un documento aparte para no gastar páginas del
reporte, que tiene tope de diez.

    python reportes/generar_soportes.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from generar_reporte import PROYECTO, RAIZ, documento, monoespaciado, parrafo, pie

from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image

from docx.shared import Cm, Pt

CAPTURAS = PROYECTO / "capturas_soporte"


def corre(comando: str) -> str:
    """Ejecuta un comando en la raíz del repositorio y devuelve su salida.

    Se decodifica primero como UTF-8 y, si falla, con la página de códigos de
    la consola de Windows: dvc recorta las líneas largas con un puntos
    suspensivos que escribe en cp1252.
    """
    resultado = subprocess.run(comando, shell=True, cwd=RAIZ, capture_output=True)
    crudo = resultado.stdout + resultado.stderr
    try:
        salida = crudo.decode("utf-8")
    except UnicodeDecodeError:
        salida = crudo.decode("cp1252")
    return salida.replace("\r\n", "\n").strip()


def bloque(doc, titulo: str, comando: str, salida: str, limite: int = 40) -> None:
    p = doc.add_paragraph(titulo)
    p.runs[0].font.bold = True
    p.runs[0].font.size = Pt(10)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    lineas = salida.split("\n")
    recorte = "\n".join(lineas[:limite])
    if len(lineas) > limite:
        recorte += f"\n... ({len(lineas) - limite} líneas más)"
    monoespaciado(doc, f"$ {comando}\n{recorte}", 7.5)


def imagen(doc, archivo: Path, leyenda: str, densidad: int = 135) -> None:
    """Inserta la captura al ancho que le da la densidad de puntos indicada."""
    if not archivo.exists():
        parrafo(doc, f"[Falta la captura {archivo.name}]")
        return
    with Image.open(archivo) as captura:
        ancho = min(16.0, captura.width / densidad * 2.54)
    doc.add_picture(str(archivo), width=Cm(ancho))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
    pie(doc, leyenda)


def main() -> int:
    doc = documento()
    titulo = doc.add_paragraph("Anexo de soportes: repositorios Git y DVC")
    titulo.runs[0].font.size = Pt(13)
    titulo.runs[0].font.bold = True
    titulo.paragraph_format.space_after = Pt(2)
    sub = doc.add_paragraph(
        "Entrega 1. Despliegue de Soluciones Analíticas (MIID-4304). 25 de octubre de 2026.\n"
        "Repositorio: https://github.com/anquint89/seguridad-vial-bogota"
    )
    for corrida in sub.runs:
        corrida.font.size = Pt(9)
    sub.paragraph_format.space_after = Pt(8)
    parrafo(
        doc,
        "Las salidas de consola de este anexo se generaron ejecutando los comandos sobre el "
        "repositorio con reportes/generar_soportes.py, de modo que no son transcripciones: "
        "corresponden al estado real del proyecto en la fecha de la entrega.",
    )

    imagen(doc, CAPTURAS / "github_repositorio.png",
           "Figura 1. Repositorio público en GitHub, rama main.", 230)
    imagen(doc, CAPTURAS / "github_commits.png",
           "Figura 2. Historial de commits con la contribución de cada integrante.", 215)

    bloque(doc, "Historial de commits", "git log --oneline --decorate",
           corre("git log --oneline --decorate"), 25)
    bloque(doc, "Commits por autor", "git shortlog -sn", corre("git shortlog -sn HEAD"), 10)
    bloque(doc, "Remoto de Git", "git remote -v", corre("git remote -v"), 6)
    bloque(doc, "Configuración de DVC", "cat .dvc/config", corre("cat .dvc/config"), 20)
    bloque(doc, "Remotos de DVC", "dvc remote list", corre("python -m dvc remote list"), 8)
    bloque(doc, "Etapas del pipeline", "dvc stage list", corre("python -m dvc stage list"), 12)
    bloque(doc, "Grafo de dependencias", "dvc dag --mermaid",
           corre("python -m dvc dag --mermaid"), 20)
    bloque(doc, "Datos versionados: resumen de dvc.lock", "cat dvc.lock",
           corre("cat dvc.lock"), 34)
    bloque(doc, "Estado del área de trabajo frente al cache",
           "dvc status", corre("python -m dvc status"), 14)
    parrafo(
        doc,
        "Las dos etapas del pipeline, perfilar y maqueta, aparecen al día: dvc.lock guarda el "
        "hash de cada entrada y de cada salida, y dvc status lo compara con el área de trabajo. "
        "El repositorio contiene solo el código que ya corre, de modo que no hay etapas "
        "declaradas sin implementación.",
    )
    bloque(doc, "Publicación de los datos en el remoto", "dvc push -r respaldo",
           corre("python -m dvc push -r respaldo"), 12)
    parrafo(
        doc,
        "El push responde que todo está al día porque los cuatro archivos ya se habían subido: la "
        "primera ejecución reportó «4 files pushed» y el contenido del remoto se lista a "
        "continuación.",
    )
    bloque(doc, "Contenido del remoto de respaldo",
           "find ../almacen-dvc-local -type f | head",
           corre("find ../almacen-dvc-local -type f | head -12"), 14)
    bloque(doc, "Archivos que Git ignora dentro de data/raw", "cat data/raw/.gitignore",
           corre("cat data/raw/.gitignore"), 8)
    parrafo(
        doc,
        "Conjuntos de datos a emplear. Los cuatro parquet están versionados en DVC y se obtienen "
        "con dvc pull después de clonar. Cada uno va acompañado de su ficha de trazabilidad, que "
        "sí está en Git porque pesa menos de un kilobyte y deja constancia de contra qué versión "
        "del servicio se corrió el pipeline.",
    )
    for capa in (
        "malla_vial_integral",
        "historico_siniestros",
        "victimas_2021_2025",
        "intervenciones_seguridad_vial",
    ):
        bloque(doc, f"Ficha de trazabilidad: {capa}", f"cat data/raw/{capa}.meta.json",
               corre(f"cat data/raw/{capa}.meta.json"), 24)

    parrafo(
        doc,
        "Sobre el remoto en S3. El bucket s3://svb-dvcstore queda declarado como remoto por "
        "defecto en .dvc/config y el push contra él se ejecuta dentro de una sesión activa del AWS "
        "Academy Learner Lab, cuyas credenciales rotan cada cuatro horas y no se guardan en el "
        "repositorio. El remoto de respaldo que aparece arriba es local y sirve para trabajar sin "
        "sesión abierta; los dos apuntan al mismo contenido, porque DVC guarda los objetos por "
        "hash.",
    )

    destino = PROYECTO / "Entrega1_Anexo_soportes.docx"
    doc.save(destino)
    print(f"Escrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
