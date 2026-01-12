import json
import sys
import httpx
from app.core.config import get_api_settings


def main():
    if len(sys.argv) < 7:
        print(
            "Usage: train_complete_client '<payload-json>' '<best-hparams-uri>' '<model-uri>' "
            "'<mlflow-model-name>' '<mlflow-model-version>' '<mlflow-run-id>'",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_json = sys.argv[1]
    best_hparams_uri = sys.argv[2]
    model_uri = sys.argv[3]
    mlflow_model_name = sys.argv[4]
    mlflow_model_version = sys.argv[5]
    mlflow_run_id = sys.argv[6]

    payload = json.loads(raw_json)
    training_job_id = payload["training_job_id"]

    api_settings = get_api_settings()
    api_base_url = api_settings.API_BASE_URL.rstrip("/")
    cb_url = f"{api_base_url}/api/v1/internal/train/train-complete"

    callback_payload = {
        "training_job_id": training_job_id,
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "best_hparams_uri": best_hparams_uri,
        "model_uri": model_uri,
        "mlflow_model_name": mlflow_model_name,
        "mlflow_model_version": mlflow_model_version,
        "mlflow_run_id": mlflow_run_id,
    }

    with httpx.Client(timeout=None) as client:
        resp = client.post(cb_url, json=callback_payload)
        resp.raise_for_status()

    print(json.dumps({"status": "ok", "training_job_id": training_job_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
