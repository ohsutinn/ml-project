from http import HTTPStatus
from typing import Any
from fastapi import APIRouter, HTTPException, status

from app.api.deps import SessionDep
from app.constants import DVMode, ModelStatus, TrainingJobStatus
from app.core.config import settings
from app.models.common import ApiResponse
from app.models.dataset import Dataset, DatasetVersion
from app.models.model import (
    Model,
    ModelCreate,
    ModelPublic,
    PreprocessCompletePayload,
    PromoteModelRequest,
    PromoteModelResponse,
    TrainCompletePayload,
    TrainModelRequest,
    TrainingJob,
)
from app.services.argo import create_model_deploy_workflow, create_model_train_workflow


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

    # 3) 베이스라인 정보 확인
    if not dataset.schema_path or not dataset.baseline_stats_path:
        raise HTTPException(
            status_code=400,
            detail="해당 데이터셋에는 베이스라인이 없습니다. 먼저 베이스라인을 생성하세요.",
        )

    # 3-1) config 필수 체크
    if not train_in.config_name:
        raise HTTPException(
            status_code=400,
            detail="config 를 지정해야 합니다.",
        )
    config_name = train_in.config_name

    # 4) TrainingJob 레코드 먼저 생성
    job = TrainingJob(
        model_id=model.id,
        dataset_id=dataset.id,
        dataset_version_id=dataset_version.id,
        workflow_name="",
        status=TrainingJobStatus.PENDING,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # 5) Argo Workflow 에 넘길 payload 구성
    workflow_payload = {
        "training_job_id": job.id,
        "model_id": model.id,
        "dataset_id": dataset.id,
        "dataset_version_id": dataset_version.id,
        "dataset_version": dataset_version.version,
        "data_path": dataset_version.storage_path,
        "label_column": train_in.label_column,
        "mode": DVMode.VALIDATE,
        "split": train_in.split,
        "schema_path": dataset.schema_path,
        "baseline_stats_path": dataset.baseline_stats_path,
        "config_name": config_name,
    }

    # 6) Argo Workflow 생성
    workflow_name = create_model_train_workflow(workflow_payload)

    job.workflow_name = workflow_name
    job.status = TrainingJobStatus.RUNNING
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
    if payload.preprocess_state_uri:
        job.preprocess_state_uri = payload.preprocess_state_uri
    if payload.feature_schema_hash:
        job.feature_schema_hash = payload.feature_schema_hash
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
            "preprocess_state_uri": job.preprocess_state_uri,
            "feature_schema_hash": job.feature_schema_hash,
            "status": job.status,
        },
    )


@internal_router.post(
    "/train-complete",
    response_model=ApiResponse[dict],
)
async def train_complete(payload: TrainCompletePayload, session: SessionDep) -> Any:
    job = await session.get(TrainingJob, payload.training_job_id)
    if not job:
        raise HTTPException(404, "TrainingJob 을 찾을 수 없습니다.")

    job.best_hparams_uri = payload.best_hparams_uri
    job.best_model_uri = payload.model_uri
    if payload.preprocess_state_uri:
        job.preprocess_state_uri = payload.preprocess_state_uri
    if payload.feature_schema_hash:
        job.feature_schema_hash = payload.feature_schema_hash
    job.status = TrainingJobStatus.DONE

    await session.commit()
    await session.refresh(job)

    return ApiResponse(
        code=HTTPStatus.OK,
        message="모델 학습 완료",
        data={
            "training_job_id": job.id,
            "best_hparams_uri": job.best_hparams_uri,
            "model_uri": job.best_model_uri,
            "preprocess_state_uri": job.preprocess_state_uri,
            "feature_schema_hash": job.feature_schema_hash,
            "status": job.status,
        },
    )


@router.post(
    "/{model_id}/promote",
    response_model=ApiResponse[PromoteModelResponse],
)
async def promote_model(model_id: int, body: PromoteModelRequest, session: SessionDep) -> Any:
    model = await session.get(Model, model_id)
    if not model or getattr(model, "deleted_at", None) is not None:
        raise HTTPException(status_code=404, detail="존재하지 않는 모델입니다.")

    registered_model_name = f"automl_model_{model_id}"
    deploy_name = "automl-model-"+{model_id}+"-serving"

    wf_name = create_model_deploy_workflow(
        {
            "model_id": model_id,
            "registered_model_name": registered_model_name,
            "src_alias": body.src_alias,
            "dst_alias": body.dst_alias,
            "serving_namespace": body.serving_namespace,
            "deploy_name": deploy_name,
            "serving_image": body.serving_image,
            "bento_signature_method": body.bento_signature_method,
        }
    )

    return ApiResponse(
        code=HTTPStatus.OK,
        message="모델 승격/배포 워크플로우를 시작했습니다.",
        data=PromoteModelResponse(
            workflow_name=wf_name,
            deploy_name=deploy_name,
            registered_model_name=registered_model_name,
            dst_alias=body.dst_alias,
        ),
    )
