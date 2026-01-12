from minio import Minio
from app.core.config import get_minio_settings

minio_settings = get_minio_settings()

minio_client = Minio(
    endpoint=minio_settings.MINIO_ENDPOINT,
    access_key=minio_settings.MINIO_ACCESS_KEY,
    secret_key=minio_settings.MINIO_SECRET_KEY,
    secure=False,
)
BUCKET_NAME = "pipelines-artifacts"