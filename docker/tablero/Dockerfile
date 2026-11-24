FROM python:3.12-slim

RUN adduser --disabled-password --gecos '' dash-user

WORKDIR /opt/svb-tablero

COPY docker/tablero/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py ./src/__init__.py
COPY src/tablero/__init__.py ./src/tablero/__init__.py
COPY src/tablero/app.py ./src/tablero/app.py
COPY src/tablero/optimizador.py ./src/tablero/optimizador.py
COPY src/tablero/assets ./src/tablero/assets

COPY docker/tablero/run.sh ./run.sh
RUN chmod +x /opt/svb-tablero/run.sh
RUN chown -R dash-user:dash-user ./

USER dash-user

# En compose el nombre del servicio resuelve dentro de la red interna. Fuera de
# compose hay que pasar API_URL al arrancar el contenedor.
ENV API_URL=http://api:8000 \
    PORT=8050

EXPOSE 8050

CMD ["bash", "./run.sh"]
