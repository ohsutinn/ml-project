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

    # 전처리 결과
    train_path: Optional[str] = Field(default=None, max_length=1024)
    eval_path: Optional[str] = Field(default=None, max_length=1024)
    preprocess_state_uri: Optional[str] = Field(default=None, max_length=1024)
    feature_schema_hash: Optional[str] = Field(default=None, max_length=128)

    # HPO + 학습 결과 아티팩트
    best_hparams_uri: Optional[str] = Field(default=None, max_length=1024)
    best_model_uri: Optional[str] = Field(default=None, max_length=1024)

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
    config_name: str 


class PreprocessCompletePayload(BaseModel):
    training_job_id: int
    train_path: str
    eval_path: str
    preprocess_state_uri: Optional[str] = None
    feature_schema_hash: Optional[str] = None


class TrainCompletePayload(SQLModel):
    training_job_id: int
    dataset_id: int
    dataset_version_id: int
    best_hparams_uri: str
    model_uri: str
    preprocess_state_uri: Optional[str] = None
    feature_schema_hash: Optional[str] = None

class PromoteModelRequest(BaseModel):
    src_alias: str = "candidate"
    dst_alias: str = "production"
    serving_namespace: str = "automl-serving"
    deploy_name: str | None = None
    serving_image: str = "ohsepang/automl-serving:0.1.0"
    bento_signature_method: str = "predict"


class PromoteModelResponse(BaseModel):
    workflow_name: str
    deploy_name: str
    registered_model_name: str
    dst_alias: str
