from app.constants import DVMode, DVSplit
from app.core.storage import resolve_data_path
from app.models.models import DataValidationRequest, DataValidationResult
from app.utils import (
    build_metrics,
    generate_stats_from_csv,
    load_schema_if_exists,
    load_stats_if_exists,
    parse_anomalies,
    save_schema,
    save_stats,
)
import tensorflow_data_validation as tfdv


async def run_data_validation(payload: DataValidationRequest) -> DataValidationResult:
    # 1) data_path (MinIO URI)
    data_path = await resolve_data_path(payload.data_path)
    split = DVSplit(payload.split)

    # 2) 현재 통계 계산
    current_stats = generate_stats_from_csv(data_path)
    mode = payload.mode

    dataset_id = payload.dataset_id
    dataset_version = payload.dataset_version

    if mode == DVMode.INITIAL_BASELINE:
        # ── 최초 베이스라인 생성 ─────────────────
        schema = tfdv.infer_schema(current_stats)
        anomalies = tfdv.validate_statistics(
            statistics=current_stats,
            schema=schema,
        )

        # 통계 / 스키마를 MinIO에 저장
        stats_uri = await save_stats(
            current_stats,
            split,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )

        schema_uri = await save_schema(
            schema,
            split,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )

        anomaly_list, summary, category_counts = parse_anomalies(anomalies)
        metrics = build_metrics(current_stats, payload.label_column, None)

        return DataValidationResult(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            split=payload.split,
            data_path=data_path,
            schema_path=schema_uri,
            baseline_stats_path=stats_uri,
            anomalies=anomaly_list,
            summary=summary,
            categories=category_counts,
            metrics=metrics,
        )

    # ── VALIDATE 모드 ─────────────────────────────

    # 기존 베이스라인 스키마/통계를 MinIO에서 로드
    schema = await load_schema_if_exists(payload.schema_path)
    baseline_stats = await load_stats_if_exists(payload.baseline_stats_path)

    anomalies = tfdv.validate_statistics(
        statistics=current_stats,
        schema=schema,
        previous_statistics=baseline_stats,
    )

    anomaly_list, summary, category_counts = parse_anomalies(anomalies)
    metrics = build_metrics(current_stats, payload.label_column, baseline_stats)

    return DataValidationResult(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        split=payload.split,
        data_path=data_path,
        schema_path=payload.schema_path,
        baseline_stats_path=payload.baseline_stats_path,
        anomalies=anomaly_list,
        summary=summary,
        categories=category_counts,
        metrics=metrics,
    )
