from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

from app.constants import AnomalySeverity, DVMode, DVSplit

T = TypeVar("T")


class ApiResponse(GenericModel, Generic[T]):

    code: int
    message: str
    data: T | None = None


class DataValidationRequest(BaseModel):
    """
    데이터 검증 요청 바디.
    """

    # 검증 대상 데이터 파일 경로 (리눅스 VM 로컬 경로, NFS 등)
    data_path: str = Field(..., description="검증 대상 데이터 경로 (CSV 등)")

    # train / eval / serving
    split: DVSplit = Field(..., description="데이터 스플릿 종류")

    # 기존 스키마 파일 경로
    schema_path: Optional[str] = Field(
        default=None,
        description="TFDV 스키마 텍스트 파일 경로 (.pbtxt). 없으면 스키마 없이 통계/분포 기반만 확인.",
    )

    # 베이스라인 통계 파일 경로
    baseline_stats_path: Optional[str] = Field(
        default=None,
        description="베이스라인 통계(.binarypb 등) 경로. 없으면 drift 비교 없이 스키마 검증만.",
    )

    # 이진 분류 등에서 레이블 컬럼명
    label_column: Optional[str] = Field(
        default=None,
        description="이진 분류 등에서 타깃(label) 컬럼명. 분포/언밸런스 메트릭 계산에 사용.",
    )

    mode: DVMode = DVMode.VALIDATE

    dataset_id: int = Field(
        ...,
        description="논리 Dataset ID",
    )

    dataset_version_number: int = Field(
        ...,
        description="DatasetVersion number.",
    )


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
    데이터 검증 전체 결과.

    - anomalies: feature별 anomaly 리스트
    - summary  : 개수 요약 + 기본 block_pipeline 플래그
    - categories:
        anomaly를 간단한 타입별로 집계한 카운트 (NEW_COLUMN, MISSING_COLUMN 등)
    - metrics:
        샘플 수, 라벨 분포, 라벨 스큐 등 정량 지표 (플랫폼 정책에서 사용)
    """

    dataset_id: Optional[int] = Field(
        default=None,
        description="해당 검증이 속한 Dataset ID",
    )

    dataset_version_number: Optional[int] = Field(
        default=None,
        description="해당 검증이 속한 DatasetVersion number",
    )

    split: DVSplit
    data_path: str
    schema_path: Optional[str] = None
    baseline_stats_path: Optional[str] = None

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
