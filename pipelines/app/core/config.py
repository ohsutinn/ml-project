from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"

    DV_SERVER_URL: str
    API_BASE_URL: str

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str

    MLFLOW_MODEL_URI: str
    BENTO_MODEL_TAG: str
    BENTO_MODEL_NAME: str
    BENTO_SIGNATURE_METHOD: str




settings = Settings()
