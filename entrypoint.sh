#!/bin/bash
set -e

VENV=/app/.venv

if [ ! -f "$VENV/bin/python" ]; then
    echo "[entrypoint] Initializing virtual environment..."
    python -m venv "$VENV"
fi

if [ ! -f "$VENV/bin/uvicorn" ]; then
    echo "[entrypoint] Installing dependencies..."
    poetry install
fi

exec "$VENV/bin/uvicorn" src.app.main:app --host 0.0.0.0 --port 8000 --reload
