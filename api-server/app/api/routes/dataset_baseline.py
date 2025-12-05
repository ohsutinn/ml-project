from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.clients.dv_client import validate_dataset_on_dv_server
from app.constants import DVMode, DVSplit, DatasetStatus
from app.models.common import ApiResponse
from app.models.dataset import (
    BaselineWithValidationResult,
    Dataset,
    DatasetBaseLineRequest,
    DatasetBaseline,
    DatasetBaselineCreate,
    DatasetVersion,
)

router = APIRouter(prefix="/datasets", tags=["dataset-baselines"])


@router.post(
    "/{dataset_id}/versions/{dataset_version_id}/baseline",
    response_model=ApiResponse[BaselineWithValidationResult],
)
async def create_baseline(
    dataset_id: int,
    dataset_version_id: int,
    body: DatasetBaseLineRequest,
    session: SessionDep,
) -> Any:
    """
    특정 DatasetVersion 에 대한 초기 베이스라인 생성 + DV 서버 호출.
    """
    # 1) DatasetVersion 조회
    dataset_version = await session.get(DatasetVersion, dataset_version_id)
    if (
        not dataset_version
        or dataset_version.deleted_at is not None
        or dataset_version.dataset_id != dataset_id
    ):
        raise HTTPException(404, "데이터셋 버전을 찾을 수 없습니다.")

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
    if not dataset or dataset.deleted_at is not None:
        raise HTTPException(404, "데이터셋을 찾을 수 없습니다.")

    dataset.baseline_version_id = dataset_version.version
    dataset.schema_path = baseline.schema_path
    dataset.baseline_stats_path = baseline.baseline_stats_path

    await session.commit()
    await session.refresh(baseline)

    return ApiResponse(
        code=HTTPStatus.OK,
        message="초기 베이스라인 생성 완료",
        data=BaselineWithValidationResult(
            baseline=baseline,
            validation=dv_result,
        ),
    )
