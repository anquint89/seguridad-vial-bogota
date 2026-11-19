#!/usr/bin/env bash
exec gunicorn src.tablero.app:server --bind "0.0.0.0:$PORT" --workers 2 --timeout 120
