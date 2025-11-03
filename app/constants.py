import enum

class DatasetStatus(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    READY = "READY"
    PROFILING = "PROFILING"
    FAILED = "FAILED"
    DELETED = "DELETED"