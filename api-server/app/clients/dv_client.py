from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.core.config import settings  # DV 서버 URL을 설정으로 빼두었다고 가정


class DVClientError(Exception):
    """DV 서버 호출 시 발생하는 에러 래퍼."""

    pass


async def validate_dataset_on_dv_server(
    *,
    data_path: str,
    split: str = "train",
    schema_path: Optional[str] = None,
    baseline_stats_path: Optional[str] = None,
    label_column: Optional[str] = None,
    mode: Optional[str] = None,
    dataset_id: Optional[int] = None,
    dataset_version: Optional[int] = None,
) -> Dict[str, Any]:
    """
    DV 서버(/api/v1/validation/run)에 데이터 검증을 요청하고,
    ApiResponse.data(DataValidationResult)를 Dict로 반환.
    """

    dv_base_url = settings.DV_SERVER_URL.rstrip("/")
    url = f"{dv_base_url}/api/v1/validation/run"

    payload: Dict[str, Any] = {
        "data_path": data_path,
        "split": split,
        "schema_path": schema_path,
        "baseline_stats_path": baseline_stats_path,
        "label_column": label_column,
        "mode": mode,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            resp = await client.post(url, json=payload)
        except Exception as e:
            raise DVClientError(f"DV 서버 호출 실패: {e!r}") from e

    if resp.status_code != 200:
        raise DVClientError(
            f"DV 서버 응답 오류: status={resp.status_code}, body={resp.text}"
        )

    body = resp.json()
    data = body.get("data")
    if data is None:
        raise DVClientError("DV 서버 응답에 data 필드가 없습니다.")

    return data
