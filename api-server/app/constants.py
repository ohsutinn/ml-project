from enum import Enum


class DatasetStatus(str, Enum):
    PENDING = "PENDING"  # 파일 업로드 완료 but TFDV 후처리 진행x
    UPLOADING = "UPLOADING"  # 업로드 세션 open, 파일 업로드 중인 상태
    READY = "READY"  # TFDV 검증 완료 -> 학습 및 추론 사용 가능
    PROFILING = "PROFILING"  # 프로파일링 및 검증 진행 중
    FAILED = "FAILED"  # 해당 버전 사용x
    DELETED = "DELETED"  # soft 삭제


class ModelStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class TrainingJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PREPROCESSED = "PREPROCESSED"
    FAILED = "FAILED"
    DONE = "DONE"


class DVMode(str, Enum):
    INITIAL_BASELINE = "INITIAL_BASELINE"
    VALIDATE = "VALIDATE"


class DVSplit(str, Enum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"
    SERVING = "SERVING"


class AnomalySeverity(str, Enum):
    UNKNOWN = "UNKNOWN"
    WARNING = "WARNING"
    ERROR = "ERROR"
