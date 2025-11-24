from datetime import datetime, timezone
from typing import Generic, List, Optional, TypeVar
from pydantic.generics import GenericModel
from sqlmodel import Column, Enum, Field, Relationship, SQLModel
from app.constants import DatasetStatus, ModelStatus
import sqlalchemy as sa


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


T = TypeVar("T")


class ApiResponse(GenericModel, Generic[T]):
    code: int
    message: str
    data: T | None = None


# ------------------------------------------------------------------------------
# Dataset / DatasetVersion 도메인
#   - Dataset  : 논리적인 데이터셋 (이름/설명)
#   - DatasetVersion : 실제 파일 스냅샷 + 버전 정보
# ------------------------------------------------------------------------------

# ---- Dataset (논리 데이터셋) ---------------------------------------------------

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

    # 관계: 하나의 Dataset 에 여러 DatasetVersion
    versions: List["DatasetVersion"] = Relationship(back_populates="dataset")


# ---- DatasetVersion (실제 버전/스냅샷) ---------------------------------------


class DatasetVersionBase(SQLModel):
    """
    실제 저장된 데이터 스냅샷(파일)에 대한 메타데이터.
    """

    # 버전 번호 (1, 2, 3 ...)
    version: int = Field(index=True)

    # 파일 정보
    file_name: str = Field(max_length=255)
    size_bytes: int
    file_type: str = Field(max_length=255)

    # 오브젝트 스토리지 경로 (예: "my-bucket/datasets/1/v1/...")
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
    실제 스냅샷(파일) 버전 테이블.
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

    dataset: Dataset = Relationship(back_populates="versions")


class ModelBase(SQLModel):
    name: str = Field(max_length=255)
    task: str = Field(max_length=64)
    status: ModelStatus = Field(
        default=ModelStatus.ACTIVE,
        sa_column=Column(
            Enum(ModelStatus, name="model_status"),
            nullable=False,
            server_default=ModelStatus.ACTIVE,
        ),
    )


class ModelCreate(ModelBase):
    pass


class ModelPublic(ModelBase):
    id: int
    created_at: datetime
    updated_at: datetime


class ModelsPubic(SQLModel):
    results: list[ModelPublic]
    count: int


class Model(ModelBase, table=True):
    id: int = Field(default=None, primary_key=True, index=True)
    created_at: datetime = Field(
        default=None,
        sa_column=Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    updated_at: datetime = Field(
        default=None,
        sa_column=Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    deleted_at: Optional[datetime] = None
