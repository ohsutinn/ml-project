from http import HTTPStatus
from typing import Any
from fastapi import APIRouter, HTTPException, status

from app.api.deps import SessionDep
from app.clients.dv_client import validate_dataset_on_dv_server
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

    # 3) DV 서버로 보낼 data_path 준비
    #
    #   - 지금 storage_path에는 예를 들어 "argmax-datasets/datasets/1/v1/xxx.csv" 가 들어있다고 하면
    #   - DV 서버 쪽 run_data_validation에서는
    #       * 로컬 경로
    #       * 또는 s3://, minio:// 같은 URI를 받아 처리하도록 구현해두면 됩니다.
    #
    #   예제: "s3://<bucket>/<object_key>" 형태로 넘기고,
    #        DV 서버의 _resolve_path에서 MinIO/S3에서 내려받도록 구현.
    #
    #   여기서는 간단히 "s3://{storage_path}" 로 감싼다고 가정합니다.
    #
    data_path = dataset_version.storage_path
    # 예: MinIO/S3 스타일 URI로 바꾸고 싶다면:
    # data_path = f"s3://{dataset_version.storage_path}"

    # 4) DV 서버 호출
    dv_result = await validate_dataset_on_dv_server(
        data_path=data_path,
        split=train_in.split,
        schema_path=None,  # 스키마/베이스라인은 나중에 붙여도 됨
        baseline_stats_path=None,
        label_column=train_in.label_column,
    )

    summary = dv_result.get("summary") or {}
    anomalies = dv_result.get("anomalies") or []
    categories = dv_result.get("categories") or {}
    metrics = dv_result.get("metrics") or {}

    error_count = summary.get("error_count", 0)
    new_col = categories.get("NEW_COLUMN", 0)
    missing_col = categories.get("MISSING_COLUMN", 0)

    if error_count > 0 or new_col > 0 or missing_col > 0:
        # 자세한 이유를 응답 본문에 같이 내려주면, 프론트/사용자가 확인하기 좋음
        raise HTTPException(
            status_code=422,
            detail={
                "message": "데이터 검증 실패: 학습을 시작할 수 없습니다.",
                "summary": summary,
                "categories": categories,
                "metrics": metrics,
                "anomalies": anomalies,
            },
        )

    # TODO: 실제 학습 파이프라인 트리거 (예: Celery job, KFP run 등)

    return ApiResponse(
        code=HTTPStatus.ACCEPTED,
        message="데이터 검증 통과. 학습 파이프라인을 시작할 수 있는 상태입니다.",
        data={
            "validation_summary": summary,
            "validation_categories": categories,
            "validation_metrics": metrics,
        },
    )
