from datetime import datetime
from typing import List, Optional

import sqlalchemy as sa
from sqlmodel import Field, Relationship, SQLModel

from app.constants import DatasetStatus
from app.models.validation import DataValidationResult


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------


class DatasetBase(SQLModel):
    """
    논리적인 데이터셋 자체에 대한 정보 (이름/설명).
    """

    name: str = Field(max_length=255, index=True)
    description: Optional[str] = Field(default=None, max_length=1000)


class DatasetCreate(DatasetBase):
    """Dataset 생성 시 요청 바디용 DTO"""

    pass


class DatasetPublic(DatasetBase):
    """단일 Dataset 조회용 응답 DTO"""

    id: int
    created_at: datetime
    updated_at: datetime


class DatasetsPublic(SQLModel):
    """다수 Dataset 리스트 응답 래퍼"""

    results: List[DatasetPublic]
    count: int


class DatasetDetailPublic(DatasetPublic):
    """
    Dataset + 해당 버전 목록까지 포함한 상세 응답 DTO.
    """

    versions: "DatasetVersionsPublic"


class Dataset(SQLModel, table=True):
    """
    논리 데이터셋 테이블.
    실제 파일/스냅샷은 DatasetVersion에서 관리.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=255, index=True)
    description: Optional[str] = Field(default=None, max_length=1000)

    baseline_version_id: Optional[int] = Field(default=None)
    schema_path: Optional[str] = Field(default=None)
    baseline_stats_path: Optional[str] = Field(default=None)

    created_at: datetime = Field(
        default=None,
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    updated_at: datetime = Field(
        default=None,
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    deleted_at: Optional[datetime] = None  # 소프트 삭제용

    versions: list["DatasetVersion"] = Relationship(back_populates="dataset")


# ------------------------------------------------------------------------------
# DatasetBaseline
# ------------------------------------------------------------------------------


class DatasetBaselineBase(SQLModel):
    """
    특정 Dataset 에 대해 사용 중인 스키마/베이스라인 통계 정보.
    (dataset_id 는 table / Public 에서만 포함)
    """

    version: int = Field(index=True)

    # MinIO 등 오브젝트 스토리지 상의 경로
    schema_path: str = Field(max_length=1024)
    baseline_stats_path: str = Field(max_length=1024)


class DatasetBaselineCreate(DatasetBaselineBase):
    """
    서버 내부에서 DatasetBaseline 레코드 생성 시 사용하는 DTO.
    """

    pass


class DatasetBaselinePublic(DatasetBaselineBase):
    """
    베이스라인 조회용 DTO.
    """

    id: int
    dataset_id: int
    created_at: datetime
    updated_at: datetime


class DatasetBaseLineRequest(SQLModel):
    """
    베이스라인 생성 시 API 요청 바디.
    """

    label_column: str | None = None


class DatasetBaseline(DatasetBaselineBase, table=True):
    """
    베이스라인 테이블 정의.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)

    created_at: datetime = Field(
        default=None,
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    updated_at: datetime = Field(
        default=None,
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )


# ------------------------------------------------------------------------------
# DatasetVersion
# ------------------------------------------------------------------------------


class DatasetVersionBase(SQLModel):
    """
    실제 저장된 데이터에 대한 메타데이터.
    """

    version: int = Field(index=True)

    # 파일 정보
    file_name: str = Field(max_length=255)
    size_bytes: int
    file_type: str = Field(max_length=255)

    # 오브젝트 스토리지 경로
    storage_path: str = Field(max_length=1024)

    # 상태 (PENDING/READY/FAILED 등)
    status: DatasetStatus = Field(
        default=DatasetStatus.PENDING,
        sa_column=sa.Column(
            sa.Enum(DatasetStatus, name="dataset_version_status"),
            nullable=False,
            server_default=DatasetStatus.PENDING,
        ),
    )


class DatasetVersionCreate(SQLModel):
    """
    서버 내부에서 DatasetVersion을 생성할 때 사용할 DTO.
    (파일 업로드 후 메타데이터 구성용).
    """

    file_name: str
    size_bytes: int
    file_type: str
    storage_path: str
    status: DatasetStatus = DatasetStatus.PENDING


class DatasetVersionPublic(DatasetVersionBase):
    """단일 DatasetVersion 조회용 응답 DTO"""

    id: int
    dataset_id: int
    created_at: datetime
    updated_at: datetime


class DatasetVersionsPublic(SQLModel):
    """버전 리스트 응답 래퍼"""

    results: List[DatasetVersionPublic]
    count: int


class DatasetVersion(DatasetVersionBase, table=True):
    """
    실제 데이터 버전 테이블.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)

    created_at: datetime = Field(
        default=None,
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    updated_at: datetime = Field(
        default=None,
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    deleted_at: Optional[datetime] = None

    dataset: Dataset = Relationship(back_populates="versions")


# ------------------------------------------------------------------------------
# Baseline + Validation
# ------------------------------------------------------------------------------


class BaselineWithValidationResult(SQLModel):
    baseline: DatasetBaselinePublic
    validation: DataValidationResult
