import json
import sys

import httpx

from core.config import settings


def main():
    if len(sys.argv) < 2:
        print("Usage: dv_client '<payload-json>'", file=sys.stderr)
        sys.exit(1)

    raw_json = sys.argv[1]
    payload = json.loads(raw_json)

    dv_base_url = settings.DV_SERVER_URL.rstrip("/")
    api_base_url = settings.API_BASE_URL.rstrip("/")

    dv_url = f"{dv_base_url}/api/v1/validation/run"
    cb_url = f"{api_base_url}/api/v1/datasets/baseline-complete"

    # 1) DV 서버 payload
    dv_request = {
        "data_path": payload["data_path"],
        "split": payload["split"],
        "label_column": payload.get("label_column"),
        "mode": payload["mode"],
        "dataset_id": payload["dataset_id"],
        "dataset_version": payload["dataset_version"],
    }

    with httpx.Client(timeout=None) as client:
        # 2) DV 서버 호출
        dv_resp = client.post(dv_url, json=dv_request)
        dv_resp.raise_for_status()
        body = dv_resp.json()
        dv_result = body["data"]

        # 3) 콜백 payload 
        callback_payload = {
            "dataset_id": payload["dataset_id"],
            "dataset_version_id": payload["dataset_version_id"],
            "dv_result": dv_result,
        }

        cb_resp = client.post(cb_url, json=callback_payload)
        cb_resp.raise_for_status()

    print(json.dumps({"dv_result": dv_result}, ensure_ascii=False))


if __name__ == "__main__":
    main()