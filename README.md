# Priorizacion de intervenciones de seguridad vial en Bogota

Prototipo del curso Despliegue de Soluciones Analiticas (MIID-4304), Universidad
de los Andes, periodo 2026-15.

El producto estima, para cada segmento de la malla vial de Bogota identificado
por su CIV, cuantos siniestros graves se esperan al ano, cuanto los reduciria
cada una de las cinco categorias de intervencion que ejecuta la Secretaria
Distrital de Movilidad, y que paquete de intervenciones cabe en un presupuesto
dado. Son dos modelos supervisados empaquetados, una API que sirve las
inferencias y un tablero que la consume, todo en contenedores Docker.

Equipo: Laura Alfonso Botia, Andres Quintero Hoyos, Andres Felipe Rodriguez
Ortiz.

## Fuentes

Las cuatro capas son publicas y se consultan por los servicios REST de ArcGIS
de la SDM, sin token:

    https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services

| Capa | Servicio | Registros |
|---|---|---|
| Malla Vial Integral | `Malla_Vial_Integral` | 139.107 |
| Historico de siniestros 2015-2021 | `HistoricoSiniestros` | 209.861 |
| Victimas 2021 - mayo 2025 | `Victima_2021_may2025_Loc` | 94.902 |
| Intervenciones de seguridad vial | `Intervenciones_Seguridad_Vial` | 27.950 |

Conteos verificados contra la API el 18 de octubre de 2026. Cada descarga deja
un `.meta.json` al lado del parquet con la fecha de consulta y el conteo que
reporto el servicio.

## Que versiona Git y que versiona DVC

Git lleva el codigo, la configuracion y los archivos de control. DVC lleva los
datos, que pesan y se regeneran desde la API.

| Ruta | Git | DVC | Por que |
|---|---|---|---|
| `src/` | si | no | Codigo. |
| `dvc.yaml`, `dvc.lock` | si | no | Definen el pipeline y fijan los hashes de cada salida. |
| `*.dvc`, `.dvc/config` | si | no | Punteros a los datos y configuracion del remoto. |
| `data/raw/*.parquet` | no | si | 31 MB y crecen; se regeneran desde la API. |
| `data/raw/*.meta.json` | si | no | Ficha de trazabilidad de cada descarga; pesa menos de 1 kB. |
| `reportes/perfilamiento.json`, `reportes/figuras/` | si | no | Pesan poco y se citan en los reportes: conviene verlos en el diff. |
| `.env`, `*.pem`, credenciales de AWS | no | no | Nunca entran al repositorio. El Learner Lab rota la sesion cada cuatro horas. |

## Puesta en marcha

Requiere Python 3.11, Git y una sesion activa del AWS Academy Learner Lab.

```bash
git clone https://github.com/anquint89/seguridad-vial-bogota.git
cd seguridad-vial-bogota
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Credenciales de AWS. En el Learner Lab, `AWS Details > AWS CLI > Show` entrega
las tres variables. Caducan con la sesion, asi que se vuelven a poner cada vez:

```bash
aws configure set aws_access_key_id ACCESS_KEY
aws configure set aws_secret_access_key SECRET_KEY
aws configure set aws_session_token SESSION_TOKEN
aws configure set region us-east-1
aws s3 ls                                   # verificacion
```

Traer los datos y reproducir el pipeline:

```bash
dvc pull            # baja data/raw desde el remoto
dvc repro           # rehace solo las etapas desactualizadas
dvc dag             # muestra el grafo de dependencias
```

## Como quedo inicializado el repositorio

Secuencia que se ejecuto una vez, al comienzo de la semana 3. Cada bloque
corresponde a un commit de la rama `main`:

```bash
mkdir seguridad-vial-bogota && cd seguridad-vial-bogota
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
dvc add data/raw/malla_vial_integral.parquet data/raw/historico_siniestros.parquet \
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
```

Los datos crudos se versionan con `dvc add`, porque vienen de un servicio
externo cuyo contenido no controlamos. De `data/raw` en adelante todo es
derivado y esta declarado como salida de una etapa de `dvc.yaml`: `dvc repro`
lo produce y `dvc push` lo publica.

Un segundo remoto local sirve para trabajar sin sesion de AWS activa:

```bash
dvc push -r respaldo
```

## Estructura

```
seguridad-vial-bogota/
  dvc.yaml                  definicion del pipeline
  dvc.lock                  hash de cada entrada y cada salida
  requirements.txt
  src/
    extraccion/             cliente REST y descarga de las cuatro capas
    exploracion/            perfilamiento y figuras del reporte
    tablero/                maqueta del tablero
  data/raw/                 punteros .dvc y fichas de trazabilidad
  reportes/                 perfilamiento.json, figuras y generadores de los documentos
```

## Estado

Entrega 1 (25 de octubre de 2026): las cuatro capas extraidas, versionadas en
DVC y perfiladas; pipeline y remoto en funcionamiento; linea base del metodo
vigente replicada; maqueta del tablero cerrada.

El repositorio contiene solo el codigo que ya corre. Los modulos de preparacion,
modelacion, analisis causal y optimizacion, la API y el tablero quedan fuera del
alcance de esta entrega y se desarrollan mas adelante.
