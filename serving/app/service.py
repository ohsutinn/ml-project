import os
import bentoml

BENTO_MODEL_NAME = os.environ["BENTO_MODEL_NAME"]

svc = bentoml.mlflow.get_service(f"{BENTO_MODEL_NAME}:latest")
