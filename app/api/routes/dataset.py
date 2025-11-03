from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any
import uuid
from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import SessionDep
from app.constants import DatasetStatus
from app.core.minio import BUCKET_NAME
from app.models.models import ApiResponse, Dataset, DatasetCreate, DatasetPublic
from app.services.storage import ensure_bucket, upload_bytes


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/", response_model=ApiResponse[DatasetPublic], status_code=status.HTTP_201_CREATED
)
async def upload_datasets(
    session: SessionDep,
    file: UploadFile = File(...),
) -> Any:
    """
    파일 업로드
    """

    # object name 만들기
    date = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    u8 = uuid.uuid4().hex[:8]
    object_name = f"datasets/{date}/{file.filename}/{u8}"

    # 버킷 생성
    await ensure_bucket()

    # 파일 업로드
    file_bytes = await file.read()
    await upload_bytes(
        data=file_bytes,
        object_name=object_name,
        content_type=file.content_type,
    )

    storage_path = f"{BUCKET_NAME}/{object_name}"

    create_in = DatasetCreate(
        file_name=file.filename,
        size_bytes=len(file_bytes),
        file_type=Path(file.filename).suffix.lower(),
        status=DatasetStatus.PENDING,
        storage_path=storage_path,
    )

    dataset = Dataset.model_validate(create_in)

    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)

    return ApiResponse(
        code=HTTPStatus.CREATED,
        message="파일 업로드 성공",
        data=dataset,
    )
