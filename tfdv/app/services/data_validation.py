from app.constants import DVMode, DVSplit
from app.models.models import DataValidationRequest, DataValidationResult
from app.utils import (
    build_metrics,
    generate_stats_from_csv,
    load_schema_if_exists,
    load_stats_if_exists,
    parse_anomalies,
    resolve_data_path,
    save_schema,
    save_stats,
)
import tensorflow_data_validation as tfdv


async def run_data_validation(payload: DataValidationRequest) -> DataValidationResult:
    data_path = resolve_data_path(payload.data_path)
    split = DVSplit(payload.split)

    current_stats = generate_stats_from_csv(data_path)
    mode = payload.mode

    if mode == DVMode.INITIAL_BASELINE:
        schema = tfdv.infer_schema(current_stats)
        anomalies = tfdv.validate_statistics(
            statistics=current_stats,
            schema=schema,
        )

        stats_path = save_stats(current_stats, split)
        schema_path = save_schema(schema, split)

        anomaly_list, summary, category_counts = parse_anomalies(anomalies)
        metrics = build_metrics(current_stats, payload.label_column, None)

        return DataValidationResult(
            mode=DVMode.INITIAL_BASELINE,
            split=payload.split,
            data_path=data_path,
            schema_path=schema_path,
            baseline_stats_path=stats_path,
            anomalies=anomaly_list,
            summary=summary,
            categories=category_counts,
            metrics=metrics,
        )

    # ---------- VALIDATE 모드 ----------
    schema = load_schema_if_exists(payload.schema_path)
    baseline_stats = load_stats_if_exists(payload.baseline_stats_path)

    anomalies = tfdv.validate_statistics(
        statistics=current_stats,
        schema=schema,
        previous_statistics=baseline_stats,
    )

    anomaly_list, summary, category_counts = parse_anomalies(anomalies)
    metrics = build_metrics(current_stats, payload.label_column, baseline_stats)

    return DataValidationResult(
        mode=DVMode.VALIDATE,
        split=payload.split,
        data_path=data_path,
        schema_path=payload.schema_path,
        baseline_stats_path=payload.baseline_stats_path,
        anomalies=anomaly_list,
        summary=summary,
        categories=category_counts,
        metrics=metrics,
    )
