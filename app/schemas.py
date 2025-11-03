from datetime import datetime, timezone
from typing import Generic, Optional, TypeVar
from pydantic.generics import GenericModel
from sqlmodel import Column, Enum, Field, SQLModel
from app.constants import DatasetStatus
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
