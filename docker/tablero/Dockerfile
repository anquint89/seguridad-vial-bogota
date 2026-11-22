FROM python:3.12-slim

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/svb-tablero

COPY docker/tablero/requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/tablero ./src/tablero
COPY docker/tablero/run.sh ./run.sh
RUN chmod +x /opt/svb-tablero/run.sh
RUN chown -R api-user:api-user ./

USER api-user
EXPOSE 8050

CMD ["bash", "./run.sh"]