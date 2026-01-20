import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from app.core.config import get_wandb_settings
from app.core.load_config import load_config
from app.core.storage import resolve_data_path, ensure_bucket, to_minio_uri
from app.core.minio import BUCKET_NAME, minio_client


# -------------------------
# MinIO Artifact Upload
# -------------------------
def upload_hpo_artifact(
    local_path: str,
    *,
    model_id: int,
    dataset_id: int,
    dataset_version_number: int,
    training_job_id: int,
    kind: str,
) -> str:
    """
    HPO 관련 아티팩트를 MinIO에 업로드하고 "minio://bucket/object" URI를 리턴.
    """
    ensure_bucket()
    bucket = BUCKET_NAME

    file_name = Path(local_path).name
    object_name = (
        f"hpo/model_{model_id}/job_{training_job_id}/"
        f"dataset_{dataset_id}/v{dataset_version_number}/{kind}/{file_name}"
    )

    minio_client.fput_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=local_path,
    )

    return to_minio_uri(bucket, object_name)


# -------------------------
# Dataset Loading
# -------------------------
def load_datasets(
    train_uri: str,
    eval_uri: str,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MinIO URI (train_uri, eval_uri)를 로컬로 내려받고
    (X_train, y_train, X_eval, y_eval)을 numpy로 리턴.
    """
    local_train = resolve_data_path(train_uri)
    local_eval = resolve_data_path(eval_uri)

    train_df = pd.read_csv(local_train)
    eval_df = pd.read_csv(local_eval)

    if label_column not in train_df.columns:
        raise ValueError(f"train 데이터에 label_column={label_column} 이 없습니다.")
    if label_column not in eval_df.columns:
        raise ValueError(f"eval 데이터에 label_column={label_column} 이 없습니다.")

    X_train = train_df.drop(columns=[label_column]).to_numpy().astype("float32")
    y_train = train_df[label_column].to_numpy()

    X_eval = eval_df.drop(columns=[label_column]).to_numpy().astype("float32")
    y_eval = eval_df[label_column].to_numpy()

    return X_train, y_train, X_eval, y_eval


# -------------------------
# Model Builder
# -------------------------
def build_keras_model(
    input_shape: Tuple[int, ...],
    problem_type: str,
    num_classes: int | None,
    run_cfg: Dict[str, Any],
) -> tf.keras.Model:
    """
    run_cfg(= wandb.config 최종값) 기반으로 모델 생성.
    ✅ 여기서는 .get() 안 씀: YAML/스윕에 키 없으면 KeyError로 바로 터지게(명시적)
    """
    hidden_units = int(run_cfg["hidden_units"])
    dropout = float(run_cfg["dropout"])
    activation = str(run_cfg["activation"])
    learning_rate = float(run_cfg["learning_rate"])
    optimizer_name = str(run_cfg["optimizer"]).lower()

    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(hidden_units, activation=activation),
        tf.keras.layers.Dropout(dropout),
    ]

    if problem_type == "regression":
        layers.append(tf.keras.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = ["mae"]
    else:
        if num_classes is None or num_classes < 2:
            raise ValueError("classification 문제인데 num_classes 가 올바르지 않습니다.")
        layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = tf.keras.Sequential(layers)

    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def main() -> None:
    # argv: 1) payload-json  2) train-uri  3) eval-uri
    if len(sys.argv) < 4:
        print("Usage: hpo_train_client '<payload-json>' '<train-path>' '<eval-path>'", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(sys.argv[1])
    train_uri = sys.argv[2]
    eval_uri = sys.argv[3]

    model_id = int(payload["model_id"])
    dataset_id = int(payload["dataset_id"])
    dataset_version_number = int(payload["dataset_version_number"])
    training_job_id = int(payload["training_job_id"])
    label_column = str(payload["label_column"])

    # ✅ config_name = "tips" → tips.yaml 로드
    config_name = payload.get("config_name")
    if not config_name:
        raise ValueError("payload.config_name 가 필요합니다. 예: 'tips'")

    cfg = load_config(config_name)

    target_cfg = cfg["target"]
    problem_type = str(target_cfg["problem_type"])  

    hpo_cfg = cfg["hpo"]
    base_hparams: Dict[str, Any] = hpo_cfg["base_hparams"]
    sweep_config: Dict[str, Any] = hpo_cfg["sweep_config"]

    # 데이터 로드
    X_train, y_train_raw, X_eval, y_eval_raw = load_datasets(train_uri, eval_uri, label_column)
    input_shape = (X_train.shape[1],)

    if problem_type == "regression":
        y_train = y_train_raw.astype("float32")
        y_eval = y_eval_raw.astype("float32")
        num_classes = None
        metric_name = "val_rmse"
        default_metric = {"name": metric_name, "goal": "minimize"}
    else:
        y_train = y_train_raw.astype("int64")
        y_eval = y_eval_raw.astype("int64")
        num_classes = int(pd.Series(y_train).nunique())
        metric_name = "val_accuracy"
        default_metric = {"name": metric_name, "goal": "maximize"}

    # W&B 프로젝트/엔티티
    wandb_settings = get_wandb_settings()
    entity = wandb_settings.WANDB_ENTITY
    project = wandb_settings.WANDB_PROJECT

    # ✅ sweep 생성용 config (계획표)
    final_sweep_config: Dict[str, Any] = {
        "method": sweep_config["method"],
        "metric": sweep_config.get("metric", default_metric),
        "parameters": sweep_config["parameters"],
    }

    # -------------------------
    # train() : agent가 조합별로 반복 실행
    # -------------------------
    def train() -> None:
        wandb.init(project=project, entity=entity, config=base_hparams)

        run_cfg = dict(wandb.config)  # ✅ base_hparams + sweep override 최종 결과

        model = build_keras_model(
            input_shape=input_shape,
            problem_type=problem_type,
            num_classes=num_classes,
            run_cfg=run_cfg,
        )

        epochs = int(run_cfg["epochs"])
        batch_size = int(run_cfg["batch_size"])

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_eval, y_eval),
            callbacks=[WandbMetricsLogger(log_freq="epoch")],
            verbose=0,
        )

        # metric 계산/로그
        y_pred = model.predict(X_eval, verbose=0)

        if problem_type == "regression":
            y_pred = y_pred.squeeze().astype("float32")
            rmse = float(np.sqrt(np.mean((y_eval - y_pred) ** 2)))
            wandb.log({"val_rmse": rmse})
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
            acc = float((y_pred_classes == y_eval).mean())
            wandb.log({"val_accuracy": acc})

        wandb.finish()

    # 1) sweep 등록(계획표 생성)
    if entity:
        sweep_id = wandb.sweep(final_sweep_config, project=project, entity=entity)
    else:
        sweep_id = wandb.sweep(final_sweep_config, project=project)

    # 2) agent 실행(계획표 조합대로 train() 반복)
    wandb.agent(sweep_id, function=train)

    # -------------------------
    # Best Run 선택
    # -------------------------
    api = wandb.Api()
    sweep_path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)

    runs = list(sweep.runs)
    if not runs:
        print("No runs found in sweep. HPO 실패.", file=sys.stderr)
        sys.exit(1)

    goal = final_sweep_config["metric"]["goal"]
    if goal == "minimize":
        best_run = min(runs, key=lambda r: r.summary.get(metric_name, float("inf")))
    else:
        best_run = max(runs, key=lambda r: r.summary.get(metric_name, 0.0))

    best_config = {k: v for k, v in best_run.config.items() if not k.startswith("_")}

    # ✅ best 재학습은 "base_hparams + best_config" 로 최종 고정
    best_train_cfg = {**base_hparams, **best_config}

    # best hparams JSON 저장
    fd_hparams, tmp_hparams = tempfile.mkstemp(suffix=".json")
    os.close(fd_hparams)
    with open(tmp_hparams, "w", encoding="utf-8") as f:
        json.dump(best_train_cfg, f, ensure_ascii=False, indent=2)

    # best 모델 재학습 + 저장
    best_model = build_keras_model(
        input_shape=input_shape,
        problem_type=problem_type,
        num_classes=num_classes,
        run_cfg=best_train_cfg,
    )

    best_model.fit(
        X_train,
        y_train,
        epochs=int(best_train_cfg["epochs"]),
        batch_size=int(best_train_cfg["batch_size"]),
        validation_data=(X_eval, y_eval),
        verbose=0,
    )

    fd_model, tmp_model = tempfile.mkstemp(suffix=".keras")
    os.close(fd_model)
    best_model.save(tmp_model)

    # MinIO 업로드
    best_hparams_uri = upload_hpo_artifact(
        tmp_hparams,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version_number,
        training_job_id=training_job_id,
        kind="hparams",
    )
    best_model_uri = upload_hpo_artifact(
        tmp_model,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version_number,
        training_job_id=training_job_id,
        kind="model",
    )

    os.remove(tmp_hparams)
    os.remove(tmp_model)

    # Argo output parameter 파일
    os.makedirs("/tmp", exist_ok=True)
    with open("/tmp/best_hparams_uri.txt", "w", encoding="utf-8") as f:
        f.write(best_hparams_uri)
    with open("/tmp/best_model_uri.txt", "w", encoding="utf-8") as f:
        f.write(best_model_uri)

    print(
        json.dumps(
            {
                "sweep_id": sweep_id,
                "best_run_id": best_run.id,
                "metric_name": metric_name,
                "best_metric": best_run.summary.get(metric_name),
                "best_hparams_uri": best_hparams_uri,
                "best_model_uri": best_model_uri,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
