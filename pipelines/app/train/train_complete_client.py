import json
import os
import sys
import httpx
from app.core.config import get_api_settings
from app.core.storage import resolve_data_path


def _load_preprocess_state(preprocess_state_uri: str) -> dict:
    local_path = resolve_data_path(preprocess_state_uri)
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def _build_input_schema(state: dict) -> dict:
    return {
        "schema_version": state.get("schema_version"),
        "feature_schema_hash": state.get("feature_schema_hash"),
        "label_column": state.get("label_column"),
        "drop_columns": state.get("drop_columns", []),
        "categorical_columns": state.get("categorical_columns", []),
        "feature_columns": state.get("feature_columns", []),
        "config_name": state.get("config_name"),
        "problem_type": state.get("problem_type"),
    }


def _build_output_schema(state: dict) -> dict:
    problem_type = state.get("problem_type")
    if problem_type == "regression":
        return {"problem_type": problem_type, "type": "number", "shape": [1]}
    if problem_type == "binary":
        return {
            "problem_type": problem_type,
            "type": "number",
            "shape": [1],
            "encoding": "probability",
        }
    if problem_type in ("multiclass", "classification"):
        return {
            "problem_type": problem_type,
            "type": "array",
            "shape": ["num_classes"],
            "encoding": "probabilities",
        }
    return {"problem_type": problem_type}


def main():
    if len(sys.argv) < 8:
        print(
            "Usage: train_complete_client '<payload-json>' '<best-hparams-uri>' '<model-uri>' "
            "'<mlflow-model-name>' '<mlflow-model-version>' '<mlflow-run-id>' "
            "'<preprocess-state-uri>'",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_json = sys.argv[1]
    best_hparams_uri = sys.argv[2]
    model_uri = sys.argv[3]
    mlflow_model_name = sys.argv[4]
    mlflow_model_version = sys.argv[5]
    mlflow_run_id = sys.argv[6]
    preprocess_state_uri = sys.argv[7]

    payload = json.loads(raw_json)
    training_job_id = payload["training_job_id"]

    api_settings = get_api_settings()
    api_base_url = api_settings.API_BASE_URL.rstrip("/")
    cb_url = f"{api_base_url}/api/v1/internal/train/train-complete"

    preprocess_state = _load_preprocess_state(preprocess_state_uri)
    feature_schema_hash = preprocess_state.get("feature_schema_hash")
    if not feature_schema_hash:
        raise ValueError("preprocess_state에 feature_schema_hash가 필요합니다.")
    input_schema = _build_input_schema(preprocess_state)
    output_schema = _build_output_schema(preprocess_state)

    callback_payload = {
        "training_job_id": training_job_id,
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "best_hparams_uri": best_hparams_uri,
        "model_uri": model_uri,
        "mlflow_model_name": mlflow_model_name,
        "mlflow_model_version": mlflow_model_version,
        "mlflow_run_id": mlflow_run_id,
        "preprocess_state_uri": preprocess_state_uri,
        "feature_schema_hash": feature_schema_hash,
        "input_schema": input_schema,
        "output_schema": output_schema,
    }

    with httpx.Client(timeout=None) as client:
        resp = client.post(cb_url, json=callback_payload)
        resp.raise_for_status()

    print(json.dumps({"status": "ok", "training_job_id": training_job_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
