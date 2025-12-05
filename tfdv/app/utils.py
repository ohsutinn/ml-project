# app/utils.py
from typing import Any, Dict, List, Tuple

import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2, schema_pb2, statistics_pb2

from app.constants import AnomalySeverity, DVSplit
from app.core.storage import (
    build_tfdv_object_name,
    download_bytes,
    upload_bytes,
)
from app.models.models import ValidationAnomaly, ValidationSummary


async def save_stats(
    stats: statistics_pb2.DatasetFeatureStatisticsList,
    split: DVSplit,
    dataset_id: int,
    dataset_version: int,
) -> str:
    """
    통계 proto → bytes 직렬화 → MinIO 업로드.
    반환: 'bucket/object' URI.
    """
    data = stats.SerializeToString()
    object_name = build_tfdv_object_name(
        dataset_id=dataset_id,
        version=dataset_version,
        split=split.value,
        kind="stats",
        ext="binpb",
    )
    uri = await upload_bytes(
        data=data,
        object_name=object_name,
        content_type="application/x-protobuf",
    )
    return uri


async def save_schema(
    schema: schema_pb2.Schema,
    split: DVSplit,
    dataset_id: int,
    dataset_version: int,
) -> str:
    """
    스키마 proto → bytes 직렬화 → MinIO 업로드.
    반환: 'bucket/object' URI.
    """
    data = schema.SerializeToString()
    object_name = build_tfdv_object_name(
        dataset_id=dataset_id,
        version=dataset_version,
        split=split.value,
        kind="schema",
        ext="binpb",
    )
    uri = await upload_bytes(
        data=data,
        object_name=object_name,
        content_type="application/x-protobuf",
    )
    return uri


async def save_anomalies(
    anomalies: anomalies_pb2.Anomalies,
    split: DVSplit,
    dataset_id: int,
    dataset_version: int,
) -> str:
    """
    아노말리 proto → bytes 직렬화 → MinIO 업로드.
    """
    data = anomalies.SerializeToString()
    object_name = build_tfdv_object_name(
        dataset_id=dataset_id,
        version=dataset_version,
        split=split.value,
        kind="anomalies",
        ext="binpb",
    )
    uri = await upload_bytes(
        data=data,
        object_name=object_name,
        content_type="application/x-protobuf",
    )
    return uri


async def load_schema_if_exists(schema_uri: str | None) -> schema_pb2.Schema | None:
    """
    MinIO URI('bucket/object')에서 스키마 로드.
    """
    if not schema_uri:
        return None

    data = await download_bytes(schema_uri)
    schema = schema_pb2.Schema()
    schema.ParseFromString(data)
    return schema


async def load_stats_if_exists(
    stats_uri: str | None,
) -> statistics_pb2.DatasetFeatureStatisticsList | None:
    """
    MinIO URI('bucket/object')에서 통계 로드.
    """
    if not stats_uri:
        return None

    data = await download_bytes(stats_uri)
    stats = statistics_pb2.DatasetFeatureStatisticsList()
    stats.ParseFromString(data)
    return stats


def generate_stats_from_csv(path: str) -> statistics_pb2.DatasetFeatureStatisticsList:
    """
    로컬 CSV 파일 경로에서 통계 생성.
    (CSV 자체는 storage.resolve_data_path에서 MinIO → temp 파일로 내려받은 상태라고 가정)
    """
    return tfdv.generate_statistics_from_csv(data_location=path)


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


def build_metrics(
    stats: statistics_pb2.DatasetFeatureStatisticsList,
    label_column: str | None,
    baseline_stats: statistics_pb2.DatasetFeatureStatisticsList | None = None,
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
