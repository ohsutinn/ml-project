from sqlalchemy import Column, DateTime, Integer, String, func, Enum as SAEnum, text
from app.constants import DatasetStatus
from app.core.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)

    # 원본 파일 메타데이터
    file_name = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)

    # 저장된 파일 경로
    storage_path = Column(String, nullable=True)

    # 상태값
    status = Column[DatasetStatus](
        SAEnum,
        nullable=False,
        default=DatasetStatus.PENDING,
        server_default=text("'PENDING'"),
    )

    # 시간 정보
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)
