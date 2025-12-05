from http import HTTPStatus
from pathlib import Path
from typing import Any
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlmodel import func, select

from app.api.deps import SessionDep
from app.clients.dv_client import validate_dataset_on_dv_server
from app.constants import DVMode, DVSplit, DatasetStatus
from app.core.minio import BUCKET_NAME
from app.models.models import (
    ApiResponse,
    Dataset,
    DatasetBaseLineRequest,
    DatasetBaseline,
    DatasetBaselineCreate,
    DatasetBaselinePublic,
    DatasetCreate,
    DatasetDetailPublic,
    DatasetPublic,
    DatasetVersion,
    DatasetVersionCreate,
    DatasetVersionPublic,
    DatasetVersionsPublic,
    DatasetsPublic,
)
from app.services.storage import build_dataset_object_name, ensure_bucket, upload_bytes


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
        .where(
            DatasetVersion.dataset_id == dataset_id
            and DatasetVersion.deleted_at.is_(None)
        )
    )
    result = await session.exec(count_statement)
    count = result.one()

    statement = (
        select(DatasetVersion)
        .where(
            DatasetVersion.dataset_id == dataset_id
            and DatasetVersion.deleted_at.is_(None)
        )
        .order_by(DatasetVersion.version.asc())
        .offset(skip)
        .limit(limit)
    )
    result = await session.exec(statement)
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


# ------------------------------------------------------------------------------
# 2) DatasetVersion 업로드 / 조회
# ------------------------------------------------------------------------------


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

    - 파일을 MinIO/S3에 스냅샷으로 저장
    - 해당 스냅샷에 대해 버전 번호를 자동으로 증가시켜 DatasetVersion 레코드 생성
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

    # 현재 dataset 의 마지막 버전 번호 조회
    statement = (
        select(DatasetVersion)
        .where(
            DatasetVersion.dataset_id == dataset_id
            and DatasetVersion.deleted_at.is_(None)
        )
        .order_by(DatasetVersion.version.desc())
    )
    result = await session.exec(statement)
    last_version = result.first()
    next_version = (
        1 if last_version is None else last_version.version + 1
    )  # 동시성 문제 고려해야 함

    # object_name 구성 (dataset_id + version 기반)
    object_name = build_dataset_object_name(
        dataset_id=dataset_id,
        version=next_version,
        file_name=file.filename,
    )

    # 파일 업로드
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

    count_statement = (
        select(func.count())
        .select_from(DatasetVersion)
        .where(
            DatasetVersion.dataset_id == dataset_id
            and DatasetVersion.deleted_at.is_(None)
        )
    )
    result = await session.exec(count_statement)
    count = result.one()

    statement = (
        select(DatasetVersion)
        .where(
            DatasetVersion.dataset_id == dataset_id
            and DatasetVersion.deleted_at.is_(None)
        )
        .order_by(DatasetVersion.version.asc())
    )
    result = await session.exec(statement)
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


# ------------------------------------------------------------------------------
# 3) Dataset Baseline 생성
# ------------------------------------------------------------------------------


@router.post(
    "/versions/{dataset_version_id}/baseline",
    response_model=ApiResponse[DatasetBaselinePublic],
)
async def create_baseline(
    dataset_version_id: int,
    body: DatasetBaseLineRequest,
    session: SessionDep,
) -> Any:

    # 1) DatasetVersion 조회
    dataset_version = await session.get(DatasetVersion, dataset_version_id)
    if not dataset_version or dataset_version.deleted_at is not None:
        raise HTTPException(404, "데이터셋을 찾을 수 없습니다.")

    # 2) Data Validation 서버에 보낼 payload 구성
    dv_result = await validate_dataset_on_dv_server(
        data_path=dataset_version.storage_path,
        split=DVSplit.TRAIN,
        schema_path=None,
        baseline_stats_path=None,
        label_column=body.label_column,
        mode=DVMode.INITIAL_BASELINE,
        dataset_id=dataset_version.dataset_id,
        dataset_version=dataset_version.version,
    )

    # 3) 응답에서 schema/statistics 저장 위치를 DB에 기록
    baseline_in = DatasetBaselineCreate(
        version=dataset_version.version,
        schema_path=dv_result["schema_path"],
        baseline_stats_path=dv_result["baseline_stats_path"],
    )

    baseline = DatasetBaseline.model_validate(
        baseline_in,
        update={
            "dataset_id": dataset_version.dataset_id,
        },
    )
    session.add(baseline)

    # DatasetVersion 상태 업데이트
    dataset_version.status = DatasetStatus.READY

    # Dataset 테이블에도 baseline 정보 세팅
    dataset = await session.get(Dataset, dataset_version.dataset_id)
    dataset.baseline_version_id = dataset_version.version
    dataset.schema_path = baseline.schema_path
    dataset.baseline_stats_path = baseline.baseline_stats_path

    await session.commit()
    await session.refresh(baseline)

    return ApiResponse(
        code=200,
        message="초기 베이스라인 생성 완료",
        data=baseline,
    )
