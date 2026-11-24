FROM python:3.12-slim

RUN adduser --disabled-password --gecos '' api-user

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/svb-api

COPY docker/api/requirements.txt ./requirements.txt
COPY model-pkg/ ./model-pkg/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py ./src/__init__.py
COPY src/api ./src/api

# Entradas de la API. Van dentro de la imagen para que el contenedor sea
# autosuficiente: en ECS no hay volumen que montar. Son unos 17 MB. Los
# modelos no estan aqui: los trae la rueda de modelo-riesgo.
COPY data/procesado/caracteristicas_inferencia.parquet \
     data/procesado/tablero_segmentos.parquet \
     data/procesado/tablero_geometria.parquet \
     data/procesado/factores.json \
     data/procesado/predicciones_metricas.json \
     ./datos/

COPY docker/api/run.sh ./run.sh
RUN chmod +x /opt/svb-api/run.sh
RUN chown -R api-user:api-user ./

USER api-user

# DATOS_DIR se puede apuntar a un volumen montado para servir otras salidas sin
# reconstruir la imagen.
ENV DATOS_DIR=/opt/svb-api/datos \
    PORT=8000

EXPOSE 8000

CMD ["bash", "./run.sh"]
