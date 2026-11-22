FROM python:3.12-slim

RUN adduser --disabled-password --gecos '' api-user

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/svb-api

COPY docker/api/requirements.txt ./requirements.txt
COPY model-pkg/ ./model-pkg/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/api ./src/api
COPY docker/api/run.sh ./run.sh
RUN chmod +x /opt/svb-api/run.sh
RUN chown -R api-user:api-user ./

USER api-user
ENV DATOS_DIR=/opt/svb-api/datos
EXPOSE 8001

CMD ["bash", "./run.sh"]