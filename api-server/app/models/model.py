from datetime import datetime
from typing import List, Optional

import sqlalchemy as sa
from pydantic import BaseModel
from sqlmodel import Column, Enum, Field, SQLModel

from app.constants import ModelStatus, TrainingJobStatus


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


class ModelsPublic(SQLModel):
    results: List[ModelPublic]
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


class TrainingJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    model_id: int = Field(index=True)
    dataset_id: int = Field(index=True)
    dataset_version_id: int = Field(index=True)

    workflow_name: str = Field(default="", max_length=255)

    train_path: Optional[str] = Field(default=None, max_length=1024)
    eval_path: Optional[str] = Field(default=None, max_length=1024)

    status: TrainingJobStatus = Field(
        default=TrainingJobStatus.PENDING,
        sa_column=Column(
            Enum(TrainingJobStatus, name="training_job_status"),
            nullable=False,
            server_default=TrainingJobStatus.PENDING,
        ),
    )

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


class TrainModelRequest(BaseModel):
    """
    모델 학습 요청 바디.

    - dataset_version_id: 어떤 데이터셋 버전으로 학습할지
    - split: 검증 시 사용할 split 정보 (기본 "train")
    - label_column: TFDV에서 라벨 분포 확인용 컬럼명
    """

    dataset_version_id: int
    split: str
    label_column: Optional[str] = None


class PreprocessCompletePayload(BaseModel):
    training_job_id: int
    train_path: str
    eval_path: str
