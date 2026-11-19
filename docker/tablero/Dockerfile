# Imagen del tablero de priorizacion.
# Se construye desde la raiz del repositorio:
#   docker build -f docker/tablero/Dockerfile -t svb-tablero:0.2.0 .
FROM python:3.12-slim

# Usuario sin privilegios que ejecuta el tablero.
RUN adduser --disabled-password --gecos '' dash-user

WORKDIR /opt/tablero

COPY docker/tablero/requirements.txt /opt/tablero/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /opt/tablero/requirements.txt

# Solo los modulos que el tablero ejecuta. datos.py y maqueta.py se quedan
# fuera a proposito: pertenecen al pipeline, no a la aplicacion servida.
COPY src/__init__.py /opt/tablero/src/__init__.py
COPY src/tablero/__init__.py /opt/tablero/src/tablero/__init__.py
COPY src/tablero/app.py /opt/tablero/src/tablero/app.py
COPY src/tablero/optimizador.py /opt/tablero/src/tablero/optimizador.py
COPY src/tablero/assets /opt/tablero/src/tablero/assets

COPY docker/tablero/run.sh /opt/tablero/run.sh
RUN chmod +x /opt/tablero/run.sh && chown -R dash-user:dash-user /opt/tablero

USER dash-user

# En compose el nombre del servicio resuelve dentro de la red; fuera de compose
# hay que pasar API_URL al arrancar el contenedor.
ENV API_URL=http://api:8000 \
    PORT=8050

EXPOSE 8050

CMD ["bash", "./run.sh"]
