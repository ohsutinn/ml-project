#!/bin/sh
set -e

echo "[boot] importing model from MLflow..."
python -m app.import_from_mlflow

echo "[boot] starting Bento service..."
exec bentoml serve app.service:svc --host 0.0.0.0 --port 3000