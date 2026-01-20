import hashlib
import json
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.load_config import load_config
from app.core.storage import resolve_data_path, save_preprocessed_split, save_preprocess_artifact

PREPROCESS_STATE_FILENAME = "preprocess_state.json"
PREPROCESS_STATE_SCHEMA_VERSION = 1


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _feature_schema_hash(schema_payload: dict) -> str:
    canonical = json.dumps(
        schema_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_preprocess_state(state: dict) -> None:
    feature_columns = state["feature_columns"]
    label_column = state.get("label_column")
    all_columns = set(state["all_columns"])

    if label_column and label_column in feature_columns:
        raise ValueError("label_column 이 feature_columns 에 포함되어 있습니다.")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns 에 중복 컬럼이 있습니다.")
    if len(state.get("categorical_columns", [])) != len(set(state.get("categorical_columns", []))):
        raise ValueError("categorical_columns 에 중복 컬럼이 있습니다.")
    for col in feature_columns:
        if col not in all_columns:
            raise ValueError(f"feature_columns 에 all_columns 에 없는 컬럼이 있습니다: {col}")


def run_preprocess(
    local_path: str,
    *,
    dataset_id: int,
    dataset_version_id: int,
    dataset_version_number: int,
    label_column: str | None,
    config_name: str,
) -> tuple[str, str, str]:
    """
    1) CSV 로드
    2) config 기반 컬럼 드롭
    3) train / eval split
    4) train 기준으로 전처리 스키마 생성
    5) train/eval 동일 스키마로 정렬
    6) MinIO에 업로드 후 URI 반환
    """
    cfg = load_config(config_name)

    target_cfg = cfg.get("target", {}) or {}
    target = target_cfg.get("name")
    problem_type = target_cfg.get("problem_type", "regression")
    drop_columns = cfg.get("drop_columns", [])

    split_cfg = cfg.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))
    stratify_by = split_cfg.get("stratify_by")

    # 1) CSV 읽기
    df = pd.read_csv(local_path)

    # 2) drop 컬럼 제거
    if drop_columns:
        df = df.drop(
            columns=[c for c in drop_columns if c in df.columns],
            errors="ignore",
        )

    # 3) stratify 설정 (split 전에 결정)
    stratify = None
    if problem_type in ("binary", "multiclass"):
        base_col = stratify_by or label_column or target
        if base_col and base_col in df.columns:
            y_for_strat = df[base_col]
            value_counts = y_for_strat.value_counts()
            # 각 클래스 최소 2개 이상일 때만 stratify 사용 (sklearn 제약)
            if (value_counts >= 2).all():
                stratify = y_for_strat

    # 4) train / eval split
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # 5) 결측치 row 제거 (split 이후 각각 적용)
    train_df = train_df.dropna().reset_index(drop=True)
    eval_df = eval_df.dropna().reset_index(drop=True)

    # 6) train 기준으로 범주형 컬럼 결정
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_column and label_column in cat_cols:
        cat_cols.remove(label_column)

    # 7) train/eval에 동일한 범주형 인코딩 적용
    if cat_cols:
        train_df = pd.get_dummies(train_df, columns=cat_cols, dummy_na=False)
        eval_df = pd.get_dummies(eval_df, columns=cat_cols, dummy_na=False)

    # 8) train 기준 컬럼 정렬(누락 컬럼은 0으로 채움)
    eval_df = eval_df.reindex(columns=train_df.columns, fill_value=0)

    suffix = Path(local_path).suffix or ".csv"

    # 6) temp 파일로 저장
    fd_train, tmp_train = tempfile.mkstemp(suffix=suffix)
    os.close(fd_train)
    train_df.to_csv(tmp_train, index=False)

    fd_eval, tmp_eval = tempfile.mkstemp(suffix=suffix)
    os.close(fd_eval)
    eval_df.to_csv(tmp_eval, index=False)

    # 7) MinIO 업로드
    train_uri = save_preprocessed_split(
        tmp_train,
        split="train",
        dataset_id=dataset_id,
        dataset_version=dataset_version_number,
    )
    eval_uri = save_preprocessed_split(
        tmp_eval,
        split="eval",
        dataset_id=dataset_id,
        dataset_version=dataset_version_number,
    )

    os.remove(tmp_train)
    os.remove(tmp_eval)

    # 8) 전처리 상태 저장 + 업로드
    feature_columns = (
        [c for c in train_df.columns if c != label_column]
        if label_column
        else list(train_df.columns)
    )
    schema_payload = {
        "schema_version": PREPROCESS_STATE_SCHEMA_VERSION,
        "label_column": label_column,
        "drop_columns": drop_columns,
        "categorical_columns": cat_cols,
        "feature_columns": feature_columns,
        "config_name": config_name,
        "problem_type": problem_type,
    }
    feature_schema_hash = _feature_schema_hash(schema_payload)

    preprocess_state = {
        "schema_version": PREPROCESS_STATE_SCHEMA_VERSION,
        "created_at_utc": _now_utc_iso(),
        "feature_schema_hash": feature_schema_hash,
        "dataset_id": dataset_id,
        "dataset_version_id": dataset_version_id,
        "dataset_version_number": dataset_version_number,
        "label_column": label_column,
        "drop_columns": drop_columns,
        "categorical_columns": cat_cols,
        "feature_columns": feature_columns,
        "all_columns": train_df.columns.tolist(),
        "config_name": config_name,
        "problem_type": problem_type,
    }
    _validate_preprocess_state(preprocess_state)

    fd_state, tmp_state = tempfile.mkstemp(suffix=".json")
    os.close(fd_state)
    with open(tmp_state, "w", encoding="utf-8") as f:
        json.dump(preprocess_state, f, ensure_ascii=False, indent=2)

    preprocess_state_uri = save_preprocess_artifact(
        tmp_state,
        dataset_id=dataset_id,
        dataset_version=dataset_version_number,
        file_name=PREPROCESS_STATE_FILENAME,
    )
    os.remove(tmp_state)

    return train_uri, eval_uri, preprocess_state_uri


def main():
    if len(sys.argv) < 2:
        print("Usage: preprocess_client '<payload-json>'", file=sys.stderr)
        sys.exit(1)

    raw_json = sys.argv[1]
    payload = json.loads(raw_json)

    dataset_id = payload["dataset_id"]
    dataset_version_id = payload["dataset_version_id"]
    dataset_version_number = payload["dataset_version_number"]
    data_path = payload["data_path"]
    label_column = payload.get("label_column")
    config_name = payload.get("config_name")

    if not config_name:
        raise ValueError("payload.config_name 가 필요합니다.")
        
    # 1) MinIO → 로컬 CSV 다운로드
    local_data_path = resolve_data_path(data_path)

    # 2) 전처리 + split + 업로드
    train_uri, eval_uri, preprocess_state_uri = run_preprocess(
        local_path=local_data_path,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        dataset_version_number=dataset_version_number,
        label_column=label_column,
        config_name=config_name,
    )

    # 3) Argo outputs 로 넘길 파일 작성
    with open("/tmp/train_path.txt", "w", encoding="utf-8") as f:
        f.write(train_uri)

    with open("/tmp/eval_path.txt", "w", encoding="utf-8") as f:
        f.write(eval_uri)

    with open("/tmp/preprocess_state_uri.txt", "w", encoding="utf-8") as f:
        f.write(preprocess_state_uri)

    # 디버깅 출력
    print(
        json.dumps(
            {
                "train_path": train_uri,
                "eval_path": eval_uri,
                "preprocess_state_uri": preprocess_state_uri,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
