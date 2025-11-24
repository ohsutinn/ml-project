from http import HTTPStatus
from typing import Any
from fastapi import APIRouter, HTTPException, status
from sqlmodel import func, select

from app.api.deps import SessionDep
from app.models.models import (
    ApiResponse,
    Dataset,
    DatasetCreate,
    DatasetDetailPublic,
    DatasetPublic,
    DatasetVersion,
    DatasetVersionsPublic,
    DatasetsPublic,
)


router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024

# ------------------------------------------------------------------------------
# 1) 논리 Dataset 생성 / 조회
# ------------------------------------------------------------------------------


@router.post(
    "/",
    response_model=ApiResponse[DatasetPublic],
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset(
    session: SessionDep,
    dataset_in: DatasetCreate,
) -> Any:
    """
    논리 데이터셋(Dataset) 생성.
    """
    dataset = Dataset.model_validate(dataset_in)
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)

    return ApiResponse(
        code=HTTPStatus.CREATED,
        message="데이터셋 생성 성공",
        data=dataset,
    )


@router.get(
    "/",
    response_model=ApiResponse[DatasetsPublic],
)
async def read_datasets(session: SessionDep, skip: int = 0, limit: int = 100) -> Any:
    """
    데이터셋 리스트 조회.
    """

    count_statement = (
        select(func.count()).select_from(Dataset).where(Dataset.deleted_at.is_(None))
    )
    result = await session.exec(count_statement)
    count = result.one()

    statement = (
        select(Dataset).where(Dataset.deleted_at.is_(None)).offset(skip).limit(limit)
    )
    result = await session.exec(statement)
    datasets = result.all()

    data = DatasetsPublic(
        results=datasets,
        count=count,
    )

    return ApiResponse(
        code=HTTPStatus.OK,
        message="데이터셋 목록 조회 성공",
        data=data,
    )


@router.get(
    "/{dataset_id}",
    response_model=ApiResponse[DatasetDetailPublic],
)
async def read_dataset(
    dataset_id: int, session: SessionDep, skip: int = 0, limit: int = 100
) -> Any:
    """
    단일 데이터셋 상세 조회 + 해당 버전 목록까지 함께 반환.
    """

    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    count_statement = (
        select(func.count())
        .select_from(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
    )
    result = await session.exec(count_statement)
    count = result.one()

    statement = (
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version.asc())
        .offset(skip)
        .limit(limit)
    )
    result = await session.exec(statement)
    versions = all()

    version_results = DatasetVersionsPublic(results=versions, count=count)

    detail = DatasetDetailPublic(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        versions=version_results,
    )

    return ApiResponse(
        code=HTTPStatus.OK,
        message="데이터셋 상세 조회 성공",
        data=detail,
    )
