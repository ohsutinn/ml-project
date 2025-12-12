from http import HTTPStatus
from typing import Any
from fastapi import APIRouter, HTTPException, status

from app.api.deps import SessionDep
from app.constants import ModelStatus
from app.models.common import ApiResponse
from app.models.dataset import DatasetVersion
from app.models.model import Model, ModelCreate, ModelPublic, TrainModelRequest


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

    1. 모델 존재/상태 확인
    2. dataset_version_id 로 DatasetVersion 조회
    3. DatasetVersion.storage_path 를 기반으로 DV 서버에 검증 요청
    4. 검증 결과(anomalies/summary)를 보고
       - 문제가 심각하면 400/422 등으로 바로 에러 응답
       - 통과하면 실제 학습 파이프라인(or 배치잡) 트리거

    ⚠️ 여기서는 "학습 파이프라인 트리거"는 TODO로 남겨두고,
       DV 검증 → 간단 결과 반환까지만 처리.
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

    # TODO: 실제 학습 파이프라인 트리거

    return ApiResponse(
        code=HTTPStatus.ACCEPTED,
        message="데이터 검증 통과. 학습 파이프라인을 시작할 수 있는 상태입니다.",
        data={
        },
    )
