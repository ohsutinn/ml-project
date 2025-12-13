import json
import sys

import httpx

from app.core.config import settings


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: train_callback_client '<payload-json>' '<train-path>' '<eval-path>'",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_json = sys.argv[1]
    train_path = sys.argv[2]
    eval_path = sys.argv[3]

    payload = json.loads(raw_json)
    training_job_id = payload["training_job_id"]

    api_base_url = settings.API_BASE_URL.rstrip("/")
    cb_url = f"{api_base_url}/api/v1/internal/train/preprocess-complete"

    callback_payload = {
        "training_job_id": training_job_id,
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "train_path": train_path,
        "eval_path": eval_path,
    }

    with httpx.Client(timeout=None) as client:
        resp = client.post(cb_url, json=callback_payload)
        resp.raise_for_status()

    print(
        json.dumps(
            {"status": "ok", "training_job_id": training_job_id},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()