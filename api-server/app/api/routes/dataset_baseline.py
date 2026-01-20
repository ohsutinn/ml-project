from http import HTTPStatus
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.constants import DVMode, DVSplit, DatasetStatus
from app.models.common import ApiResponse
from app.models.dataset import (
    BaselineCallback,
    BaselineJobCreated,
    Dataset,
    DatasetBaseLineRequest,
    DatasetBaseline,
    DatasetBaselineCreate,
    DatasetVersion,
)
from app.services.argo import create_dv_initial_baseline_workflow

router = APIRouter(prefix="/datasets", tags=["dataset-baselines"])


@router.post(
    "/{dataset_id}/versions/{dataset_version_id}/baseline",
    response_model=ApiResponse[BaselineJobCreated],
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
    dv_payload = {
        # DV 서버용
        "data_path": dataset_version.storage_path,
        "split": DVSplit.TRAIN.value,
        "label_column": body.label_column,
        "mode": DVMode.INITIAL_BASELINE.value,
        "dataset_id": dataset_version.dataset_id,
        "dataset_version_number": dataset_version.version,
        # 콜백용
        "dataset_version_id": dataset_version.id,
    }

    # 3) Argo Workflow 생성
    workflow_name = create_dv_initial_baseline_workflow(dv_payload)

    # 4) DB 상태 업데이트
    dataset_version.status = DatasetStatus.PROFILING
    await session.commit()
    await session.refresh(dataset_version)

    return ApiResponse(
        code=HTTPStatus.ACCEPTED,
        message="초기 베이스라인 생성 워크플로우를 시작했습니다.",
        data=BaselineJobCreated(
            workflow_name=workflow_name,
            dataset_id=dataset_id,
            dataset_version_number=dataset_version.version,
            status=dataset_version.status,
        ),
    )


@router.post(
    "/baseline-complete",
    response_model=ApiResponse[Dict[str, Any]],
)
async def baseline_complete(
    payload: BaselineCallback,
    session: SessionDep,
) -> Any:
    dataset_version = await session.get(DatasetVersion, payload.dataset_version_id)
    if (
        not dataset_version
        or dataset_version.deleted_at is not None
        or dataset_version.dataset_id != payload.dataset_id
    ):
        raise HTTPException(400, "데이터셋 버전을 찾을 수 없습니다.")

    dv_result = payload.dv_result

    baseline_in = DatasetBaselineCreate(
        version=dataset_version.version,
        schema_path=dv_result["schema_path"],
        baseline_stats_path=dv_result["baseline_stats_path"],
    )

    baseline = DatasetBaseline.model_validate(
        baseline_in,
        update={"dataset_id": dataset_version.dataset_id},
    )
    session.add(baseline)

    dataset_version.status = DatasetStatus.READY

    dataset = await session.get(Dataset, dataset_version.dataset_id)
    if not dataset or dataset.deleted_at is not None:
        raise HTTPException(400, "데이터셋을 찾을 수 없습니다.")

    dataset.baseline_version_id = dataset_version.version
    dataset.schema_path = baseline.schema_path
    dataset.baseline_stats_path = baseline.baseline_stats_path

    await session.commit()
    await session.refresh(baseline)

    return ApiResponse(
        code=HTTPStatus.OK,
        message="초기 베이스라인 생성 완료",
        data={
            "dataset_id": dataset.id,
            "dataset_version_number": dataset_version.version,
            "baseline_id": baseline.id,
            "schema_path": baseline.schema_path,
            "baseline_stats_path": baseline.baseline_stats_path,
        },
    )
