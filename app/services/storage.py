from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
import uuid

from fastapi.concurrency import run_in_threadpool

from app.core.minio import minio_client, BUCKET_NAME


async def ensure_bucket(bucket: str = BUCKET_NAME) -> None:
    """
    버킷이 없으면 생성.
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
) -> None:
    """
    바이트 배열을 MinIO/S3에 업로드 (threadpool에서 동기 클라이언트 호출).
    """
    def _upload():
        minio_client.put_object(
            bucket,
            object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    await run_in_threadpool(_upload)

def build_dataset_object_name(
    dataset_id: int,
    version: int,
    file_name: str,
) -> str:
    """
    Dataset / DatasetVersion 용 object_name 규칙.

    예시:
      datasets/42/v3/2025/11/19/abcdefgh_original.csv
    """

    date = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    safe_name = Path(file_name).name  # 경로 제거, 파일명만
    u8 = uuid.uuid4().hex[:8]

    return f"datasets/{dataset_id}/v{version}/{date}/{u8}_{safe_name}"