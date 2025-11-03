from io import BytesIO
from typing import Optional

from fastapi.concurrency import run_in_threadpool

from app.core.minio import minio_client, BUCKET_NAME


async def ensure_bucket(bucket: str = BUCKET_NAME) -> None:
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
    def _upload():
        minio_client.put_object(
            bucket,
            object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    await run_in_threadpool(_upload)
