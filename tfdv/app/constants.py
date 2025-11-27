from enum import Enum


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
