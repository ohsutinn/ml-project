import hashlib
import json
import logging
import os
from pathlib import Path
import bentoml
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from app.preprocessor import Preprocessor

PREPROCESS_URI_TAG_KEY = "preprocess_state_uri"
PREPROCESS_HASH_TAG_KEY = "feature_schema_hash"
PREPROCESS_SCHEMA_VERSION_TAG_KEY = "preprocess_schema_version"

DEFAULT_SCHEMA_VERSION = 1


def _parse_model_uri(model_uri: str) -> tuple[str | None, str | None, str | None]:
    """
    models:/name@alias -> (name, alias, None)
    models:/name/3     -> (name, None, version)
    """
    if not model_uri.startswith("models:/"):
        return None, None, None

    name_alias = model_uri[len("models:/") :]
    if "@" in name_alias:
        name, alias = name_alias.split("@", 1)
        return name, alias, None
    if "/" in name_alias:
        name, version = name_alias.split("/", 1)
        return name, None, version
    return None, None, None


def _compute_feature_schema_hash(state: dict) -> str:
    schema_payload = {
        "schema_version": state.get("schema_version", DEFAULT_SCHEMA_VERSION),
        "label_column": state.get("label_column"),
        "drop_columns": state.get("drop_columns", []),
        "categorical_columns": state.get("categorical_columns", []),
        "feature_columns": state.get("feature_columns", []),
        "config_name": state.get("config_name"),
        "problem_type": state.get("problem_type"),
    }
    canonical = json.dumps(
        schema_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

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
    logger.info("Imported Bento model: %s", bento_model)

    # -------------------------
    # 전처리 아티팩트 다운로드 + Bento 모델 저장
    # -------------------------
    preprocess_local_path: str | None = None
    preprocess_state: dict | None = None
    try:
        preprocess_state_uri = None
        expected_hash = None
        expected_schema_version = None

        if model_uri.startswith("models:/"):
            name, alias, version = _parse_model_uri(model_uri)
            if not name or (not alias and not version):
                raise RuntimeError("MLFLOW_MODEL_URI 에서 모델 이름/버전을 찾을 수 없습니다.")

            if version:
                mv = client.get_model_version(name=name, version=version)
            else:
                mv = client.get_model_version_by_alias(name=name, alias=alias)

            tags = mv.tags or {}
            preprocess_state_uri = tags.get(PREPROCESS_URI_TAG_KEY)
            expected_hash = tags.get(PREPROCESS_HASH_TAG_KEY)
            expected_schema_version = tags.get(PREPROCESS_SCHEMA_VERSION_TAG_KEY)

            if not preprocess_state_uri or not expected_hash or not expected_schema_version:
                run = client.get_run(mv.run_id)
                run_tags = run.data.tags
                preprocess_state_uri = preprocess_state_uri or run_tags.get(PREPROCESS_URI_TAG_KEY)
                expected_hash = expected_hash or run_tags.get(PREPROCESS_HASH_TAG_KEY)
                expected_schema_version = (
                    expected_schema_version or run_tags.get(PREPROCESS_SCHEMA_VERSION_TAG_KEY)
                )
                logger.warning(
                    "ModelVersion tags missing; falling back to Run tags for preprocess state."
                )
        elif model_uri.startswith("runs:/"):
            parts = model_uri.split("/")
            run_id = parts[1] if len(parts) > 1 else None
            if run_id:
                run = client.get_run(run_id)
                preprocess_state_uri = run.data.tags.get(PREPROCESS_URI_TAG_KEY)
                expected_hash = run.data.tags.get(PREPROCESS_HASH_TAG_KEY)
                expected_schema_version = run.data.tags.get(PREPROCESS_SCHEMA_VERSION_TAG_KEY)
            else:
                raise RuntimeError("MLFLOW_MODEL_URI 에서 run_id 를 찾을 수 없습니다.")
        else:
            raise RuntimeError(f"지원하지 않는 MLFLOW_MODEL_URI 형식입니다: {model_uri}")

        if not preprocess_state_uri:
            raise RuntimeError("MLflow tag preprocess_state_uri 가 필요합니다.")
        if not expected_hash:
            raise RuntimeError("MLflow tag feature_schema_hash 가 필요합니다.")
        if not expected_schema_version:
            raise RuntimeError("MLflow tag preprocess_schema_version 가 필요합니다.")
        if not preprocess_state_uri.startswith("runs:/"):
            raise RuntimeError("preprocess_state_uri 는 runs:/ 형태여야 합니다.")

        preprocess_local_path = download_artifacts(artifact_uri=preprocess_state_uri)
        with open(preprocess_local_path, "r", encoding="utf-8") as f:
            preprocess_state = json.load(f)

        actual_schema_version = preprocess_state.get("schema_version")
        if str(actual_schema_version) != str(expected_schema_version):
            raise RuntimeError(
                "preprocess_state schema_version 이 MLflow tag 와 일치하지 않습니다."
            )

        actual_hash = _compute_feature_schema_hash(preprocess_state)
        if actual_hash != expected_hash:
            raise RuntimeError("preprocess_state hash 가 MLflow tag 와 일치하지 않습니다.")
    except Exception:
        logger.exception("Failed to load preprocess state")
        raise

    if not preprocess_local_path or not preprocess_state or not Path(preprocess_local_path).exists():
        raise RuntimeError("preprocess state is required but missing.")

    preprocessor = Preprocessor.from_state_file(preprocess_local_path)
    preprocess_model_name = f"{bento_name}_preprocess"
    bento_preprocess_model = bentoml.picklable_model.save_model(
        preprocess_model_name,
        preprocessor,
        signatures={"transform": {"batchable": True, "batchdim": 0}},
    )
    logger.info("Imported Bento preprocess model: %s", bento_preprocess_model)


if __name__ == "__main__":
    main()
