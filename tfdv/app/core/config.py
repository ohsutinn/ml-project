from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # 공통 API 정보
    API_V1_STR: str = "/api/v1"

    # ---- MinIO / 인프라 관련 ----
    REGION: str
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str

    # ---- TFDV / 파일 경로 관련 ----
    DV_BASE_DIR: Path = Path("/opt/data_validation")

    @property
    def DV_OUTPUT_DIR(self) -> Path:
        return self.DV_BASE_DIR / "output"

    @property
    def DV_SCHEMA_DIR(self) -> Path:
        return self.DV_OUTPUT_DIR / "schema"

    @property
    def DV_STATS_DIR(self) -> Path:
        return self.DV_OUTPUT_DIR / "stats"

    @property
    def DV_ANOMALIES_DIR(self) -> Path:
        return self.DV_OUTPUT_DIR / "anomalies"


settings = Settings()
