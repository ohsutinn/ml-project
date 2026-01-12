import os
from pathlib import Path
import bentoml
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from app.preprocessor import Preprocessor

PREPROCESS_TAG_KEY = "preprocess_artifact_path"
DEFAULT_PREPROCESS_ARTIFACT = "preprocess/preprocess_state.json"


def _parse_model_uri(model_uri: str) -> tuple[str | None, str | None]:
    """
    models:/name@alias -> (name, alias)
    models:/name/3     -> (name, version)
    """
    if not model_uri.startswith("models:/"):
        return None, None

    name_alias = model_uri[len("models:/") :]
    if "@" in name_alias:
        name, alias = name_alias.split("@", 1)
        return name, alias
    if "/" in name_alias:
        name, version = name_alias.split("/", 1)
        return name, version
    return None, None


def main():
    # MLflow 설정
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.environ["MLFLOW_MODEL_URI"]
    bento_name = os.environ["BENTO_MODEL_NAME"]
    signature_method = os.getenv("BENTO_SIGNATURE_METHOD", "predict")

    client = MlflowClient()

    bento_model = bentoml.mlflow.import_model(
        bento_name,
        model_uri=model_uri,
        signatures={signature_method: {"batchable": True, "batchdim": 0}},
    )
    print("Imported Bento model:", bento_model)

    # -------------------------
    # 전처리 아티팩트 다운로드 + Bento 모델 저장
    # -------------------------
    preprocess_local_path: str | None = None
    try:
        if model_uri.startswith("models:/"):
            name, alias_or_version = _parse_model_uri(model_uri)
            if name and alias_or_version:
                if alias_or_version.isdigit():
                    mv = client.get_model_version(name=name, version=alias_or_version)
                else:
                    mv = client.get_model_version_by_alias(name=name, alias=alias_or_version)

                artifact_relpath = mv.tags.get(PREPROCESS_TAG_KEY, DEFAULT_PREPROCESS_ARTIFACT)
                preprocess_local_path = download_artifacts(
                    artifact_uri=f"runs:/{mv.run_id}/{artifact_relpath}"
                )
        elif model_uri.startswith("runs:/"):
            parts = model_uri.split("/")
            run_id = parts[1] if len(parts) > 1 else None
            if run_id:
                preprocess_local_path = download_artifacts(
                    artifact_uri=f"runs:/{run_id}/{DEFAULT_PREPROCESS_ARTIFACT}"
                )
    except Exception as exc:  # pragma: no cover - 보호 로그
        print(f"[warn] failed to download preprocess artifact: {exc}")

    if preprocess_local_path and Path(preprocess_local_path).exists():
        preprocessor = Preprocessor.from_state_file(preprocess_local_path)
        preprocess_model_name = f"{bento_name}_preprocess"
        bento_preprocess_model = bentoml.picklable_model.save_model(
            preprocess_model_name,
            preprocessor,
            signatures={"transform": {"batchable": True, "batchdim": 0}},
        )
        print("Imported Bento preprocess model:", bento_preprocess_model)
    else:
        print("[warn] preprocess artifact not found; preprocessing will be skipped.")


if __name__ == "__main__":
    main()
