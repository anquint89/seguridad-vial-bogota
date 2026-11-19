# Imagen de la API de priorizacion.
# Se construye desde la raiz del repositorio:
#   docker build -f docker/api/Dockerfile -t svb-api:0.2.0 .
FROM python:3.12-slim

# Usuario sin privilegios que ejecuta la API.
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/api

# Las dependencias primero: cambian menos que el codigo y asi la capa se
# reutiliza entre construcciones. Todas traen rueda para Python 3.12, por eso
# no hace falta build-essential.
COPY docker/api/requirements.txt /opt/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /opt/api/requirements.txt

# Codigo de la API.
COPY src/__init__.py /opt/api/src/__init__.py
COPY src/api /opt/api/src/api

# Modelo empaquetado. Es lo que la API ejecuta en cada peticion.
COPY modelos /opt/api/modelos

# Entradas del modelo y lo descriptivo del segmento, que no se predice. Van
# dentro de la imagen para que el contenedor sea autosuficiente en ECS, donde
# no hay volumen que montar. Son unos 12 MB.
COPY data/procesado/caracteristicas_inferencia.parquet \
     data/procesado/tablero_segmentos.parquet \
     data/procesado/tablero_geometria.parquet \
     data/procesado/factores.json \
     data/procesado/predicciones_metricas.json \
     /opt/api/datos/

COPY docker/api/run.sh /opt/api/run.sh
RUN chmod +x /opt/api/run.sh && chown -R api-user:api-user /opt/api

USER api-user

# DATOS_DIR se puede apuntar a un volumen montado para servir otras salidas
# sin reconstruir la imagen.
ENV DATOS_DIR=/opt/api/datos \
    MODELOS_DIR=/opt/api/modelos \
    PORT=8000

EXPOSE 8000

CMD ["bash", "./run.sh"]
