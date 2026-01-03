import os
import tempfile
from pathlib import Path
from typing import Optional

from app.core.minio import BUCKET_NAME, minio_client


def ensure_bucket(bucket: Optional[str] = None) -> None:
    """
    MinIO 버킷이 없으면 생성.
    """
    if bucket is None:
        bucket = BUCKET_NAME

    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)


def resolve_data_path(path: str) -> str:
    # 1) minio:// 프리픽스 있으면 제거
    if path.startswith("minio://"):
        path = path[len("minio://") :]

    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"지원하지 않는 data_path 형식입니다: {path}")

    bucket, object_name = parts

    suffix = Path(object_name).suffix or ".csv"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # MinIO → 로컬 다운로드
    minio_client.fget_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=tmp_path,
    )

    return tmp_path


def save_preprocessed_split(
    local_path: str,
    *,
    split: str,
    dataset_id: Optional[int] = None,
    dataset_version: Optional[int] = None,
) -> str:
    ensure_bucket()
    bucket = BUCKET_NAME

    file_name = Path(local_path).name

    if dataset_id is not None and dataset_version is not None:
        object_name = (
            f"preprocessed/datasets/{dataset_id}/v{dataset_version}/{split}/{file_name}"
        )
    else:
        object_name = f"preprocessed/{split}/{file_name}"

    minio_client.fput_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=local_path,
    )

    return f"{bucket}/{object_name}"


def save_preprocess_artifact(
    local_path: str,
    *,
    dataset_id: int,
    dataset_version: int,
    file_name: Optional[str] = None,
) -> str:
    """
    전처리 상태(JSON 등)를 MinIO에 업로드하고 URI를 반환한다.
    """
    ensure_bucket()
    bucket = BUCKET_NAME

    file_name = file_name or Path(local_path).name
    object_name = f"preprocess_state/datasets/{dataset_id}/v{dataset_version}/{file_name}"

    minio_client.fput_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=local_path,
    )

    return f"{bucket}/{object_name}"
