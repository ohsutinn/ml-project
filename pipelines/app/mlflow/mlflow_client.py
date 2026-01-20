import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient

from app.core.storage import resolve_data_path
from app.core.load_config import load_config


def load_xy(csv_path: str, label_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"label_column={label_column} 이(가) CSV에 없습니다.")
    X = df.drop(columns=[label_column])
    y = df[label_column].to_numpy()
    return X, y


def compute_metrics(problem_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    - regression: val_rmse
    - classification: val_accuracy (y_pred는 softmax(2D) 가정)
    """
    if problem_type == "regression":
        y_true = y_true.astype("float32")
        y_pred = y_pred.reshape(-1).astype("float32")
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"val_rmse": rmse}

    y_true = y_true.astype("int64")
    y_cls = np.argmax(y_pred, axis=1)
    acc = float((y_cls == y_true).mean())
    return {"val_accuracy": acc}


def main() -> None:
    if len(sys.argv) < 7:
        print(
            (
                "Usage: register_client '<payload-json>' '<train-path>' '<eval-path>' "
                "'<preprocess-state-uri>' '<best-hparams-uri>' '<model-uri>'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    payload = json.loads(sys.argv[1])
    _train_uri = sys.argv[2]  # 지금은 안 씀(필요해지면 signature/샘플 등에 활용 가능)
    eval_uri = sys.argv[3]
    preprocess_state_uri = sys.argv[4]
    best_hparams_uri = sys.argv[5]
    model_uri = sys.argv[6]

    training_job_id = int(payload["training_job_id"])
    model_id = int(payload["model_id"])
    dataset_id = int(payload["dataset_id"])
    dataset_version = int(payload["dataset_version"])
    label_column = str(payload["label_column"])
    config_name = payload.get("config_name")
    if not config_name:
        raise ValueError("payload.config_name 가 필요합니다.")

    # -------------------------
    # MLflow 설정
    # -------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI 환경변수가 필요합니다.")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "automl")
    mlflow.set_experiment(experiment_name)

    # Registered Model naming rule
    registered_model_name = f"automl_model_{model_id}"

    # -------------------------
    # MinIO -> local
    # -------------------------
    local_eval = resolve_data_path(eval_uri)
    local_hparams = resolve_data_path(best_hparams_uri)
    local_model = resolve_data_path(model_uri)
    local_preprocess_state = resolve_data_path(preprocess_state_uri)

    # config 로드 (problem_type)
    cfg = load_config(config_name)
    problem_type = str(cfg["target"]["problem_type"])

    # eval 로드 + metric 계산
    x_eval_df, y_eval = load_xy(local_eval, label_column)
    model = tf.keras.models.load_model(local_model)

    y_pred = model.predict(x_eval_df.to_numpy().astype("float32"), verbose=0)
    metrics = compute_metrics(problem_type, y_eval, y_pred)

    # hparams 로드
    with open(local_hparams, "r", encoding="utf-8") as f:
        best_hparams: Dict[str, Any] = json.load(f)

    # preprocess 상태 로드
    with open(local_preprocess_state, "r", encoding="utf-8") as f:
        preprocess_state: Dict[str, Any] = json.load(f)

    preprocess_schema_version = preprocess_state.get("schema_version")
    feature_schema_hash = preprocess_state.get("feature_schema_hash")
    if preprocess_schema_version is None or not feature_schema_hash:
        raise ValueError("preprocess_state에 schema_version/feature_schema_hash가 필요합니다.")

    # -------------------------
    # Tracking(run) + 모델 로깅 + Registry(버전 생성)
    # -------------------------
    with mlflow.start_run(run_name=f"train_job_{training_job_id}") as run:
        run_id = run.info.run_id

        # 전처리 상태 아티팩트
        preprocess_artifact_name = Path(local_preprocess_state).name
        preprocess_artifact_path = f"preprocess/{preprocess_artifact_name}"
        mlflow.log_artifact(local_preprocess_state, artifact_path="preprocess")
        preprocess_state_uri = f"runs:/{run_id}/{preprocess_artifact_path}"

        # UI에서 찾기 쉬운 tags
        mlflow.set_tags(
            {
                "training_job_id": str(training_job_id),
                "model_id": str(model_id),
                "dataset_id": str(dataset_id),
                "dataset_version": str(dataset_version),
                "config_name": str(config_name),
                "problem_type": str(problem_type),
                "best_hparams_uri": str(best_hparams_uri),
                "best_model_uri": str(model_uri),
                "preprocess_state_uri": str(preprocess_state_uri),
                "preprocess_schema_version": str(preprocess_schema_version),
                "feature_schema_hash": str(feature_schema_hash),
            }
        )

        # params / metrics
        mlflow.log_params({k: str(v) for k, v in best_hparams.items()})
        mlflow.log_metrics(metrics)

        # hparams.json도 artifact로 남김
        mlflow.log_artifact(local_hparams, artifact_path="hpo")

        # 모델 아티팩트(Tracking)
        mlflow.keras.log_model(
            model,
            artifact_path="model",
        )

        # Registry 버전 생성
        model_source = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_source, registered_model_name)

    # -------------------------
    # (선택) candidate alias만 설정 (승격은 사람)
    # -------------------------
    client = MlflowClient()
    if os.getenv("MLFLOW_SET_CANDIDATE_ALIAS", "true").lower() in ("1", "true", "yes"):
        client.set_registered_model_alias(name=registered_model_name, alias="candidate", version=mv.version)

    # 전처리 아티팩트 경로/해시를 모델 버전 태그로 남김 (서빙 시 사용)
    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="preprocess_artifact_path",
        value=preprocess_artifact_path,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="preprocess_state_uri",
        value=preprocess_state_uri,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="preprocess_schema_version",
        value=str(preprocess_schema_version),
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="feature_schema_hash",
        value=str(feature_schema_hash),
    )

    # -------------------------
    # Argo outputs
    # -------------------------
    Path("/tmp").mkdir(parents=True, exist_ok=True)
    Path("/tmp/mlflow_model_name.txt").write_text(registered_model_name, encoding="utf-8")
    Path("/tmp/mlflow_model_version.txt").write_text(str(mv.version), encoding="utf-8")
    Path("/tmp/mlflow_run_id.txt").write_text(run_id, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "tracking_uri": tracking_uri,
                "experiment": experiment_name,
                "registered_model_name": registered_model_name,
                "model_version": mv.version,
                "run_id": run_id,
                "metrics": metrics,
                "preprocess_artifact_path": preprocess_artifact_path,
                "preprocess_state_uri": preprocess_state_uri,
                "preprocess_schema_version": preprocess_schema_version,
                "feature_schema_hash": feature_schema_hash,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
