import os
import json
import sys

import httpx

from app.core.config import get_data_validation_settings


def main():
    if len(sys.argv) < 2:
        print("Usage: dv_validate_client '<payload-json>'", file=sys.stderr)
        sys.exit(1)

    raw_json = sys.argv[1]
    payload = json.loads(raw_json)

    dv_settings = get_data_validation_settings()
    dv_base_url = dv_settings.DV_SERVER_URL.rstrip("/")
    dv_url = f"{dv_base_url}/api/v1/validation/run"

    # DV 요청 바디 (필요한 값만 넣기)
    dv_request = {
        "data_path": payload["data_path"],
        "split": payload["split"],
        "label_column": payload.get("label_column"),
        "mode": payload["mode"],
        "dataset_id": payload["dataset_id"],
        "dataset_version": payload["dataset_version"],
        "schema_path": payload.get("schema_path"),
        "baseline_stats_path": payload.get("baseline_stats_path"),
    }

    with httpx.Client(timeout=None) as client:
        dv_resp = client.post(dv_url, json=dv_request)
        dv_resp.raise_for_status()
        body = dv_resp.json()
        dv_result = body["data"]

    summary = dv_result.get("summary", {})
    block_pipeline = bool(summary.get("block_pipeline"))

    # Argo outputs parameter용 파일 생성
    os.makedirs("/tmp", exist_ok=True)
    with open("/tmp/block_pipeline.txt", "w", encoding="utf-8") as f:
        f.write("true" if block_pipeline else "false")

    # 디버깅 로그
    print(
        json.dumps(
            {
                "block_pipeline": block_pipeline,
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
