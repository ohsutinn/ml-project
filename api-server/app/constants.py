import enum


class DatasetStatus(str, enum.Enum):
    PENDING = "PENDING"  # 파일 업로드 완료 but TFDV 후처리 진행x
    UPLOADING = "UPLOADING"  # 업로드 세션 open, 파일 업로드 중인 상태
    READY = "READY"  # TFDV 검증 완료 -> 학습 및 추론 사용 가능
    PROFILING = "PROFILING"  # 프로파일링 및 검증 진행 중
    FAILED = "FAILED"  # 해당 버전 사용x
    DELETED = "DELETED"  # soft 삭제


class ModelStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class DVMode(str, enum.Enum):
    INITIAL_BASELINE = "INITIAL_BASELINE"
    VALIDATE = "VALIDATE"


class DVSplit(str, enum.Enum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"
    SERVING = "SERVING"


class AnomalySeverity(str, enum.Enum):
    UNKNOWN = "UNKNOWN"
    WARNING = "WARNING"
    ERROR = "ERROR"
