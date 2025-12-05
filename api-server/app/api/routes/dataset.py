from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlmodel import func, select

from app.api.deps import SessionDep
from app.models.common import ApiResponse
from app.models.dataset import (
    Dataset,
    DatasetCreate,
    DatasetDetailPublic,
    DatasetPublic,
    DatasetVersionsPublic,
    DatasetsPublic,
    DatasetVersion,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


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
async def read_datasets(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    데이터셋 리스트 조회.
    """
    count_stmt = (
        select(func.count())
        .select_from(Dataset)
        .where(Dataset.deleted_at.is_(None))
    )
    result = await session.exec(count_stmt)
    count = result.one()

    stmt = (
        select(Dataset)
        .where(Dataset.deleted_at.is_(None))
        .offset(skip)
        .limit(limit)
    )
    result = await session.exec(stmt)
    datasets = result.all()

    data = DatasetsPublic(results=datasets, count=count)

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
    dataset_id: int,
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    단일 데이터셋 상세 조회 + 해당 버전 목록까지 함께 반환.
    """
    dataset = await session.get(Dataset, dataset_id)
    if not dataset or dataset.deleted_at is not None:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    # AND 조건은 파이썬 and 말고 & 연산자로
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
        .offset(skip)
        .limit(limit)
    )
    result = await session.exec(stmt)
    versions = result.all()

    version_results = DatasetVersionsPublic(results=versions, count=count)

    data = DatasetDetailPublic(
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
        data=data,
    )