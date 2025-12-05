from io import BytesIO
from pathlib import Path
from typing import Optional

import os
import tempfile

from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.minio import BUCKET_NAME, minio_client


async def ensure_bucket(bucket: str = BUCKET_NAME) -> None:
    """
    TFDV 아티팩트를 저장할 버킷이 없으면 생성.
    """
    def _ensure():
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)

    await run_in_threadpool(_ensure)


async def upload_bytes(
    data: bytes,
    object_name: str,
    content_type: Optional[str] = None,
    bucket: str = BUCKET_NAME,
) -> str:
    """
    bytes → MinIO 바로 업로드.
    """
    await ensure_bucket(bucket)

    def _upload():
        minio_client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    await run_in_threadpool(_upload)

    return f"{bucket}/{object_name}"


def build_tfdv_object_name(
    dataset_id: int,
    version: int,
    split: str,
    kind: str,  
    ext: str,   
) -> str:
    """
    TFDV 아티팩트용 object_name 규칙.

    예:
      tfdv/datasets/42/v3/train/schema.binpb
    """
    return f"tfdv/datasets/{dataset_id}/v{version}/{split}/{kind}.{ext}"


async def download_bytes(uri: str) -> bytes:
    """
    'bucket/object' 형식의 URI에서 MinIO 객체를 읽어 bytes로 반환.
    (스키마/통계/아노말리 로딩용)
    """
    parts = uri.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MinIO URI 형식이 올바르지 않습니다: {uri}",
        )

    bucket, object_name = parts

    def _download() -> bytes:
        resp = minio_client.get_object(bucket, object_name)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    data: bytes = await run_in_threadpool(_download)
    return data


async def resolve_data_path(path: str) -> str:
    """
    TFDV 입력 데이터(data_path)용 경로 해석.
    """

    parts = path.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 data_path 형식입니다: {path}",
        )

    bucket, object_name = parts
    suffix = Path(object_name).suffix

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    def _download():
        minio_client.fget_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=tmp_path,
        )

    await run_in_threadpool(_download)
    return tmp_path