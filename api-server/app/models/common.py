from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic.generics import GenericModel

T = TypeVar("T")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ApiResponse(GenericModel, Generic[T]):
    code: int
    message: str
    data: T | None = None
