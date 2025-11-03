from minio import Minio
from app.core.config import settings  # 여기에 MINIO_ENDPOINT 등 넣어두세요

minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False,                     
)
BUCKET_NAME = "demo-bucket"