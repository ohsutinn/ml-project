import os
import sys
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main():
    if len(sys.argv) < 4:
        print("Usage: promote_client '<payload-json>' '<src-alias>' '<dst-alias>'", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(sys.argv[1])
    src_alias = sys.argv[2]
    dst_alias = sys.argv[3]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI env is required")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = payload["registered_model_name"]

    mv = client.get_model_version_by_alias(name=model_name, alias=src_alias)
    version = mv.version

    client.set_registered_model_alias(name=model_name, alias=dst_alias, version=version)

    Path("/tmp").mkdir(parents=True, exist_ok=True)
    Path("/tmp/promoted_model_name.txt").write_text(model_name, encoding="utf-8")
    Path("/tmp/promoted_version.txt").write_text(str(version), encoding="utf-8")
    Path("/tmp/promoted_alias.txt").write_text(dst_alias, encoding="utf-8")

    print(json.dumps({"model_name": model_name, "version": version, "alias": dst_alias}, ensure_ascii=False))


if __name__ == "__main__":
    main()