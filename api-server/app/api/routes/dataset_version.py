from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlmodel import func, select

from app.api.deps import SessionDep
from app.constants import DatasetStatus
from app.core.minio import BUCKET_NAME
from app.models.common import ApiResponse
from app.models.dataset import (
    Dataset,
    DatasetVersion,
    DatasetVersionCreate,
    DatasetVersionPublic,
    DatasetVersionsPublic,
)
from app.services.storage import build_dataset_object_name, ensure_bucket, upload_bytes

router = APIRouter(prefix="/datasets", tags=["dataset-versions"])

MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024


@router.post(
    "/{dataset_id}/versions",
    response_model=ApiResponse[DatasetVersionPublic],
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset_version(
    dataset_id: int,
    session: SessionDep,
    file: UploadFile = File(...),
) -> Any:
    """
    특정 Dataset 아래에 새로운 버전을 업로드.
    """
    dataset = await session.get(Dataset, dataset_id)
    if not dataset or dataset.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="데이터셋을 찾을 수 없습니다.",
        )

    await ensure_bucket()

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{file.filename} 이(가) 2GB를 초과합니다.",
        )

    # 마지막 버전 조회
    stmt_last = (
        select(DatasetVersion)
        .where(
            (DatasetVersion.dataset_id == dataset_id)
            & (DatasetVersion.deleted_at.is_(None))
        )
        .order_by(DatasetVersion.version.desc())
    )
    result = await session.exec(stmt_last)
    last_version = result.first()
    next_version = 1 if last_version is None else last_version.version + 1  # TODO: 동시성 개선 여지

    object_name = build_dataset_object_name(
        dataset_id=dataset_id,
        version=next_version,
        file_name=file.filename,
    )

    await upload_bytes(
        data=file_bytes,
        object_name=object_name,
        content_type=file.content_type,
    )

    storage_path = f"{BUCKET_NAME}/{object_name}"

    dataset_version_in = DatasetVersionCreate(
        file_name=file.filename,
        size_bytes=len(file_bytes),
        file_type=Path(file.filename).suffix.lower(),
        storage_path=storage_path,
        status=DatasetStatus.PENDING,
    )

    dataset_version = DatasetVersion.model_validate(
        dataset_version_in,
        update={
            "dataset_id": dataset_id,
            "version": next_version,
        },
    )

    session.add(dataset_version)
    await session.commit()
    await session.refresh(dataset_version)

    return ApiResponse(
        code=HTTPStatus.CREATED,
        message="데이터셋 버전 업로드 성공",
        data=dataset_version,
    )


@router.get(
    "/{dataset_id}/versions",
    response_model=ApiResponse[DatasetVersionsPublic],
)
async def read_dataset_versions(
    dataset_id: int,
    session: SessionDep,
) -> Any:
    """
    특정 Dataset 의 버전 목록 조회.
    """
    dataset = await session.get(Dataset, dataset_id)
    if not dataset or dataset.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="데이터셋을 찾을 수 없습니다.",
        )

    count_stmt = (
        select(func.count())
        .select_from(DatasetVersion)
        .where(
            (DatasetVersion.dataset_id == dataset_id)
            & (DatasetVersion.deleted_at.is_(None))
        )
    )
    result = await session.exec(count_stmt)
    count = result.one()

    stmt = (
        select(DatasetVersion)
        .where(
            (DatasetVersion.dataset_id == dataset_id)
            & (DatasetVersion.deleted_at.is_(None))
        )
        .order_by(DatasetVersion.version.asc())
    )
    result = await session.exec(stmt)
    versions = result.all() or []

    data = DatasetVersionsPublic(
        results=versions,
        count=count,
    )

    return ApiResponse(
        code=HTTPStatus.OK,
        message="데이터셋 버전 목록 조회 성공",
        data=data,
    )