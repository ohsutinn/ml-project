from http import HTTPStatus
from typing import Any
from fastapi import APIRouter, HTTPException, status

from app.api.deps import SessionDep
from app.constants import ModelStatus, TrainingJobStatus
from app.models.common import ApiResponse
from app.models.dataset import Dataset, DatasetVersion
from app.models.model import (
    Model,
    ModelCreate,
    ModelPublic,
    PreprocessCompletePayload,
    TrainModelRequest,
    TrainingJob,
)
from app.services.argo import create_model_train_workflow


router = APIRouter(prefix="/models", tags=["models"])


@router.post(
    "/",
    response_model=ApiResponse[ModelPublic],
    status_code=status.HTTP_201_CREATED,
)
async def create_model(session: SessionDep, model_in: ModelCreate) -> Any:
    """
    모델 정의 생성
    - 모델 이름, 태스크 등록
    """

    model = Model.model_validate(model_in)
    session.add(model)
    await session.commit()
    await session.refresh(model)

    return ApiResponse[ModelPublic](
        code=HTTPStatus.CREATED,
        message="모델 정의 생성 성공",
        data=model,
    )


@router.post(
    "/{model_id}/versions",
    response_model=ApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def train_model(
    session: SessionDep, model_id: int, train_in: TrainModelRequest
) -> Any:
    """
    모델 학습 요청
    - TFDV VALIDATE + 전처리 파이프라인(Argo Workflow) 트리거
    """

    # 1) 모델 존재 검증
    model = await session.get(Model, model_id)
    if not model or model.deleted_at is not None or model.status == ModelStatus.DELETED:
        raise HTTPException(status_code=404, detail="존재하지 않는 모델입니다.")

    # 2) DatasetVersion 조회
    dataset_version = await session.get(DatasetVersion, train_in.dataset_version_id)
    if not dataset_version:
        raise HTTPException(
            status_code=404, detail="존재하지 않는 데이터셋 버전입니다."
        )

    dataset = await session.get(Dataset, dataset_version.dataset_id)
    if not dataset or dataset.deleted_at is not None:
        raise HTTPException(status_code=404, detail="존재하지 않는 데이터셋입니다.")

    # 3) 베이스라인 정보 확인 (없으면 학습 시작하면 안 됨)
    if not dataset.schema_path or not dataset.baseline_stats_path:
        raise HTTPException(
            status_code=400,
            detail="해당 데이터셋에는 베이스라인이 없습니다. 먼저 베이스라인을 생성하세요.",
        )

    # 4) Argo Workflow 에 넘길 payload 구성
    workflow_payload = {
        "model_id": model.id,
        "dataset_id": dataset.id,
        "dataset_version_id": dataset_version.id,
        "dataset_version": dataset_version.version,
        "data_path": dataset_version.storage_path,
        "label_column": train_in.label_column,
        "split": train_in.split,
        "schema_path": dataset.schema_path,
        "baseline_stats_path": dataset.baseline_stats_path,
    }

    # 5) Argo Workflow 생성
    workflow_name = create_model_train_workflow(workflow_payload)

    # 6) TrainingJob 레코드 생성
    job = TrainingJob(
        model_id=model.id,
        dataset_id=dataset.id,
        dataset_version_id=dataset_version.id,
        workflow_name=workflow_name,
        status=TrainingJobStatus.RUNNING,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    return ApiResponse(
        code=HTTPStatus.ACCEPTED,
        message="학습 파이프라인(전처리 전까지)을 시작했습니다.",
        data={
            "training_job_id": job.id,
            "workflow_name": workflow_name,
        },
    )


internal_router = APIRouter(
    prefix="/internal/train",
    tags=["internal-train"],
    include_in_schema=False,
)


@internal_router.post(
    "/preprocess-complete",
    response_model=ApiResponse[dict],
)
async def preprocess_complete(
    payload: PreprocessCompletePayload,
    session: SessionDep,
) -> Any:
    """
    Argo train-client 가 전처리 완료 후 호출하는 콜백.
    train / eval 경로를 TrainingJob 에 저장.
    """
    job = await session.get(TrainingJob, payload.training_job_id)
    if not job:
        raise HTTPException(404, "TrainingJob 을 찾을 수 없습니다.")

    job.train_path = payload.train_path
    job.eval_path = payload.eval_path
    job.status = TrainingJobStatus.PREPROCESSED

    await session.commit()
    await session.refresh(job)

    return ApiResponse(
        code=HTTPStatus.OK,
        message="전처리 결과 저장 완료",
        data={
            "training_job_id": job.id,
            "train_path": job.train_path,
            "eval_path": job.eval_path,
            "status": job.status,
        },
    )
