from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


class ApiSettings(_BaseSettings):
    API_V1_STR: str = "/api/v1"
    API_BASE_URL: str


class DataValidationSettings(_BaseSettings):
    DV_SERVER_URL: str


class MinioSettings(_BaseSettings):
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str


class BentoSettings(_BaseSettings):
    MLFLOW_MODEL_URI: str
    BENTO_MODEL_TAG: str
    BENTO_MODEL_NAME: str
    BENTO_SIGNATURE_METHOD: str


class WandbSettings(_BaseSettings):
    WANDB_ENTITY: str | None = None
    WANDB_PROJECT: str = "automl-hpo"


@lru_cache
def get_api_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache
def get_data_validation_settings() -> DataValidationSettings:
    return DataValidationSettings()


@lru_cache
def get_minio_settings() -> MinioSettings:
    return MinioSettings()


@lru_cache
def get_bento_settings() -> BentoSettings:
    return BentoSettings()


@lru_cache
def get_wandb_settings() -> WandbSettings:
    return WandbSettings()
