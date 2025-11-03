from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.deps import SessionDep
from app.constants import DatasetStatus
from app.core.minio import BUCKET_NAME
from app.models.models import ApiResponse, Dataset, DatasetCreate, DatasetsPublic
from app.services.storage import ensure_bucket, upload_bytes


router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024


@router.post(
    "/",
    response_model=ApiResponse[DatasetsPublic],
    status_code=status.HTTP_201_CREATED,
)
async def upload_datasets(
    session: SessionDep,
    files: list[UploadFile] = File(...),
) -> Any:
    """
    파일 업로드
    """
    # 버킷 생성
    await ensure_bucket()

    saved_datasets: list[Dataset] = []

    for file in files:

        file_bytes = await file.read()

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename} 이(가) 2GB를 초과합니다.",
            )

        date = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        u8 = uuid.uuid4().hex[:8]
        object_name = f"datasets/{date}/{file.filename}/{u8}"

        # 파일 업로드
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
        saved_datasets.append(dataset)

    await session.commit()

    for ds in saved_datasets:
        await session.refresh(ds)

    return ApiResponse(
        code=HTTPStatus.CREATED,
        message="파일 업로드 성공",
        data=DatasetsPublic(
            results=saved_datasets,
            count=len(saved_datasets)
        ),
    )
