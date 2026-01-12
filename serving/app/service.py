import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import bentoml
from bentoml.exceptions import NotFound
from pydantic import BaseModel

BENTO_MODEL_NAME = os.environ["BENTO_MODEL_NAME"]
MODEL_SIGNATURE_METHOD = os.getenv("BENTO_SIGNATURE_METHOD", "predict")
PREPROCESS_MODEL_NAME = f"{BENTO_MODEL_NAME}_preprocess"

class InferencePayload(BaseModel):
    rows: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[Any]


@bentoml.service(name="mlflow_preprocess_service")
class MLflowPreprocessService:
    def __init__(self) -> None:
        self.model = bentoml.mlflow.load_model(f"{BENTO_MODEL_NAME}:latest")
        try:
            self.preprocessor = bentoml.picklable_model.load_model(
                f"{PREPROCESS_MODEL_NAME}:latest"
            )
        except NotFound:
            self.preprocessor = None

    @bentoml.api
    def predict(self, payload: InferencePayload) -> PredictionResponse:
        df = pd.DataFrame(payload.rows)

        if self.preprocessor:
            features = self.preprocessor.transform(df)
        else:
            features = df.to_numpy(dtype="float32")

        predict_fn = getattr(self.model, MODEL_SIGNATURE_METHOD)
        preds = predict_fn(features)
        return PredictionResponse(predictions=np.asarray(preds).tolist())


svc = MLflowPreprocessService
