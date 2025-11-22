from datetime import datetime, timezone
from typing import Generic, Optional, TypeVar
from pydantic.generics import GenericModel
from sqlmodel import Column, Enum, Field, SQLModel, table
from app.constants import DatasetStatus, ModelStatus
import sqlalchemy as sa


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


T = TypeVar("T")


class ApiResponse(GenericModel, Generic[T]):
    code: int
    message: str
    data: T | None = None


class DatasetBase(SQLModel):
    file_name: str = Field(max_length=255)
    size_bytes: int
    file_type: str = Field(max_length=255)
    status: DatasetStatus = Field(
        default=DatasetStatus.PENDING,
        sa_column=Column(
            Enum(DatasetStatus, name="status"),
            nullable=False,
            server_default=DatasetStatus.PENDING,
        ),
    )


class DatasetCreate(DatasetBase):
    storage_path: str


class DatasetPublic(DatasetBase):
    id: int
    created_at: datetime
    updated_at: datetime


class DatasetsPublic(SQLModel):
    results: list[DatasetPublic]
    count: int


class Dataset(DatasetBase, table=True):
    id: int = Field(default=None, primary_key=True, index=True)
    storage_path: Optional[str] = None
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


class ModelBase(SQLModel):
    name: str = Field(max_length=255)
    task: str = Field(max_length=64)
    status: ModelStatus = Field(
        default=ModelStatus.ACTIVE,
        sa_column=Column(
            Enum(ModelStatus, name="status"),
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


class ModelsPubic(ModelPublic):
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
