from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from http import HTTPStatus


from app.models.models import (
    ApiResponse,
    DataValidationRequest,
    DataValidationResult,
)
from app.services.data_validation import run_data_validation


router = APIRouter(prefix="/validation", tags=["data-validation"])


@router.post(
    "/run",
    response_model=ApiResponse[DataValidationResult],
)
async def validate_data(payload: DataValidationRequest) -> Any:
    """
    데이터 검증 실행 엔드포인트.
    """
    result = await run_data_validation(payload)
    return ApiResponse(
        code=HTTPStatus.OK,
        message="데이터 검증 완료",
        data=result,
    )