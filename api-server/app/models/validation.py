from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.constants import AnomalySeverity, DVSplit


class ValidationAnomaly(BaseModel):
    """
    TFDV가 검출한 단일 이상 징후(Anomaly)에 대한 요약 정보.
    """

    feature_name: str = Field(
        ..., description="특징(컬럼) 이름, 글로벌 이슈는 '__GLOBAL__' 등"
    )
    short_description: str = Field(..., description="짧은 설명")
    description: Optional[str] = Field(
        default=None,
        description="상세 설명 (TFDV가 제공하는 텍스트)",
    )
    severity: AnomalySeverity = Field(
        ...,
        description="심각도 (ERROR / WARNING / UNKNOWN)",
    )
    reason: Optional[str] = Field(
        default=None,
        description="주요 reason 요약 (optional)",
    )
    category: str = Field(
        default="OTHER",
        description="간단한 카테고리 (예: NEW_COLUMN, MISSING_COLUMN, DISTRIBUTION_CHANGE 등)",
    )


class ValidationSummary(BaseModel):
    """
    전체 검증 결과에 대한 요약.
    """

    total_count: int = Field(..., description="전체 anomaly 개수")
    error_count: int = Field(..., description="ERROR 수준 anomaly 개수")
    warning_count: int = Field(..., description="WARNING 수준 anomaly 개수")
    unknown_count: int = Field(..., description="UNKNOWN 수준 anomaly 개수")
    block_pipeline: bool = Field(
        ...,
        description="ERROR가 하나라도 있으면 True. 정책은 상위에서 재해석 가능.",
    )


class DataValidationResult(BaseModel):
    """
    API 서버에서 사용하는 TFDV 검증 결과 요약.

    - split: 어느 split(TRAIN/EVAL/SERVING)에 대한 검증인지
    - anomalies: feature별 anomaly 리스트
    - summary: 개수 요약 + block_pipeline 플래그
    - categories: anomaly 카테고리별 카운트
    - metrics: 샘플 수, 라벨 분포/스큐 등 정량 지표
    """

    split: DVSplit

    anomalies: List[ValidationAnomaly] = []
    summary: ValidationSummary

    categories: Dict[str, int] = Field(
        default_factory=dict,
        description="카테고리별 anomaly 개수 (예: {'NEW_COLUMN': 1, 'MISSING_COLUMN': 2})",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="샘플 수, 라벨 분포/스큐 등 정량 지표",
    )

    validated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="검증 수행 시각(UTC)",
    )