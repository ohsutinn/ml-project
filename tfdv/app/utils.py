import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException, status
from app.core.config import settings
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2, schema_pb2, statistics_pb2
from urllib.parse import urlparse
from app.constants import AnomalySeverity, DataSplit
from app.core.minio import minio_client
from app.models.models import ValidationAnomaly, ValidationSummary


BASE_DIR = settings.DV_BASE_DIR.resolve()
OUTPUT_DIR = settings.DV_OUTPUT_DIR
SCHEMA_DIR = settings.DV_SCHEMA_DIR
STATS_DIR = settings.DV_STATS_DIR
ANOMALIES_DIR = settings.DV_ANOMALIES_DIR


def ensure_output_dirs() -> None:
    """데이터 검증 결과를 저장할 디렉토리들을 모두 만들어 둔다."""
    for d in (OUTPUT_DIR, SCHEMA_DIR, STATS_DIR, ANOMALIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_schema_if_exists(schema_path: str | None) -> schema_pb2.Schema | None:
    if not schema_path:
        return None
    if not os.path.exists(schema_path):
        return None
    return tfdv.load_schema_text(schema_path)


def load_stats_if_exists(
    stats_path: str | None,
) -> statistics_pb2.DatasetFeatureStatisticsList | None:
    if not stats_path:
        return None
    if not os.path.exists(stats_path):
        return None
    return tfdv.load_statistics(stats_path)


def generate_stats_from_csv(path: str) -> statistics_pb2.DatasetFeatureStatisticsList:
    return tfdv.generate_statistics_from_csv(data_location=path)


def save_stats(
    stats: statistics_pb2.DatasetFeatureStatisticsList,
    split: DataSplit,
) -> str:
    """
    현재 통계를 디스크에 저장 (디버깅/TFDV HTML용 등).
    ArgMax Mini에서는 dataset_version_id 기반으로 관리해도 무방.
    """
    stats_path = STATS_DIR / f"stats_{split.value}.binarypb"
    tfdv.write_stats_text(stats, str(stats_path))
    return str(stats_path)


def save_anomalies(
    anomalies: anomalies_pb2.Anomalies,
    split: DataSplit,
) -> str:
    """
    anomalies proto를 디스크에 저장 (디버깅/HTML 렌더링 용도).
    """
    anomalies_path = ANOMALIES_DIR / f"anomalies_{split.value}.pbtxt"
    tfdv.write_anomalies_text(anomalies, str(anomalies_path))
    return str(anomalies_path)


def categorize_anomaly(info: anomalies_pb2.AnomalyInfo) -> str:
    """
    간단한 문자열 매칭 기반 카테고리 분류.

    - NEW_COLUMN
    - MISSING_COLUMN
    - DISTRIBUTION_CHANGE
    - 기타는 OTHER
    """
    text = ((info.short_description or "") + " " + (info.description or "")).upper()

    if "NEW" in text and "COLUMN" in text:
        return "NEW_COLUMN"
    if "MISSING" in text and "COLUMN" in text:
        return "MISSING_COLUMN"
    if "DISTRIBUTION" in text or "DRIFT" in text or "SKEW" in text:
        return "DISTRIBUTION_CHANGE"
    return "OTHER"


def parse_anomalies(
    anomalies: anomalies_pb2.Anomalies,
) -> Tuple[List[ValidationAnomaly], ValidationSummary, Dict[str, int]]:
    """
    TFDV Anomalies proto를 ValidationAnomaly 리스트와 요약/카테고리로 변환.
    """
    anomaly_list: List[ValidationAnomaly] = []
    category_counts: Dict[str, int] = {}

    error_count = 0
    warning_count = 0
    unknown_count = 0

    for feature_name, info in anomalies.anomaly_info.items():
        severity_enum = info.severity

        if severity_enum == anomalies_pb2.AnomalyInfo.ERROR:
            error_count += 1
            severity = AnomalySeverity.ERROR
        elif severity_enum == anomalies_pb2.AnomalyInfo.WARNING:
            warning_count += 1
            severity = AnomalySeverity.WARNING
        else:
            unknown_count += 1
            severity = AnomalySeverity.UNKNOWN

        # 가장 대표적인 reason 하나 정도 요약 (있을 때만)
        reason_msg = None
        if info.reason:
            r = info.reason[0]
            reason_msg = f"{r.type}: {r.short_description or ''}".strip()

        category = categorize_anomaly(info)
        category_counts[category] = category_counts.get(category, 0) + 1

        anomaly_list.append(
            ValidationAnomaly(
                feature_name=feature_name,
                short_description=info.short_description or "",
                description=info.description or None,
                severity=severity,
                reason=reason_msg,
                category=category,
            )
        )

    total_count = error_count + warning_count + unknown_count

    summary = ValidationSummary(
        total_count=total_count,
        error_count=error_count,
        warning_count=warning_count,
        unknown_count=unknown_count,
        block_pipeline=(error_count > 0),
    )

    return anomaly_list, summary, category_counts


def compute_label_distribution(
    stats: statistics_pb2.DatasetFeatureStatisticsList,
    label_column: str,
) -> Dict[str, float] | None:
    if not stats.datasets:
        return None

    ds = stats.datasets[0]
    target_feat = None
    for f in ds.features:
        if not f.path.step:
            continue

        first_step = f.path.step[0]
        fname = first_step.value if hasattr(first_step, "value") else first_step

        if fname == label_column:
            target_feat = f
            break

    if target_feat is None:
        return None

    if not target_feat.HasField("string_stats"):
        return None

    cs = target_feat.string_stats.common_stats
    total = cs.num_non_missing
    if total == 0:
        return None

    value_counts: Dict[str, float] = {}
    for v in target_feat.string_stats.top_values:
        value_counts[v.value] = value_counts.get(v.value, 0.0) + v.frequency

    dist = {k: v / total for k, v in value_counts.items()}
    return dist


def compute_label_skew(
    current: Dict[str, float] | None,
    baseline: Dict[str, float] | None,
) -> float | None:
    """
    두 라벨 분포 간 L1 distance 계산 (간단 스큐 지표).
    """
    if current is None or baseline is None:
        return None

    keys = set(current.keys()) | set(baseline.keys())
    return sum(abs(current.get(k, 0.0) - baseline.get(k, 0.0)) for k in keys)


def resolve_data_path(path: str) -> str:
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 data_path 형식입니다: {path}",
        )

    bucket, object_name = parts

    suffix = os.path.splitext(object_name)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        minio_client.fget_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=tmp_path,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MinIO에서 파일을 다운로드하는 중 오류가 발생했습니다: {e}",
        )

    return tmp_path


def save_schema(schema: schema_pb2.Schema, split: DataSplit) -> str:
    schema_path = SCHEMA_DIR / f"schema_{split.value}.pbtxt"
    tfdv.write_schema_text(schema, str(schema_path))
    return str(schema_path)


def build_metrics(
    stats, label_column: str | None, baseline_stats=None
) -> dict[str, Any]:
    num_examples = stats.datasets[0].num_examples if stats.datasets else 0
    num_features = len(stats.datasets[0].features) if stats.datasets else 0

    label_distribution = None
    baseline_label_distribution = None
    label_skew_l1 = None

    if label_column:
        label_distribution = compute_label_distribution(stats, label_column)
        if baseline_stats is not None:
            baseline_label_distribution = compute_label_distribution(
                baseline_stats, label_column
            )
            label_skew_l1 = compute_label_skew(
                current=label_distribution,
                baseline=baseline_label_distribution,
            )

    return {
        "num_examples": num_examples,
        "num_features": num_features,
        "label_column": label_column,
        "label_distribution": label_distribution,
        "baseline_label_distribution": baseline_label_distribution,
        "label_skew_l1": label_skew_l1,
    }
