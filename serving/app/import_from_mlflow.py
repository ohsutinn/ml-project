import os
import bentoml
import mlflow


def main():
    # MLflow 설정
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.environ["MLFLOW_MODEL_URI"]
    bento_name = os.environ["BENTO_MODEL_NAME"]
    signature_method = os.getenv("BENTO_SIGNATURE_METHOD", "predict")

    bento_model = bentoml.mlflow.import_model(
        bento_name,
        model_uri=model_uri,
        signatures={signature_method: {"batchable": True, "batchdim": 0}},
    )
    print("Imported Bento model:", bento_model)


if __name__ == "__main__":
    main()
