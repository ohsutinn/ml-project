import enum

class DatasetStatus(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    READY = "READY"
    PROFILING = "PROFILING"
    FAILED = "FAILED"
    DELETED = "DELETED"

class ModelStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"