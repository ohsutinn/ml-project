from typing import Any
from fastapi import APIRouter, HTTPException, status

from app.api.deps import SessionDep
from app.constants import ModelStatus
from app.models.models import ApiResponse, Model, ModelCreate, ModelPublic


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

    model = Model.mmodel_validate(model_in)
    session.add(model)
    session.commit()
    session.refresh(model)

    return ApiResponse[ModelPublic](
        code=0,
        message="ok",
        data=model,
    )


@router.post(
    "/{model_id}/versions",
    response_model=ApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def train_model(session: SessionDep, model_id: int) -> Any:
    """
    모델 학습 요청
    """

    # 모델 존재 검증
    model = session.get(Model, model_id)
    if not model or model.status == ModelStatus.DELETED:
        raise HTTPException(status_code=404, detail="존재하지 않는 모델입니다.")

    
    # TODO: 학습 시작

    return ApiResponse(
        code=0,
        message="학습 요청이 수행됩니다.",
        data=None,
    )
