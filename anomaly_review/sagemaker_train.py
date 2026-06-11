"""
SageMaker training entrypoint for anomaly review (per-well RF + autoencoder).

Input channel (recommended):
  s3://bucket/{org_id}/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet

Legacy fallback:
  s3://bucket/{org_id}/sensor_data/daily_production.parquet

Env:
  SM_CHANNEL_PRODUCTION_DATA (preferred) or SM_CHANNEL_SENSOR_DATA
  SM_MODEL_DIR, SM_HP_ORGANIZATION_ID, REVIEW_TRAINING_PARTITION_DAYS
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    production_channel = os.environ.get(
        "SM_CHANNEL_PRODUCTION_DATA",
        os.environ.get("SM_CHANNEL_SENSOR_DATA", "/opt/ml/input/data/production_data"),
    )
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    org_id = os.environ.get("SM_HP_ORGANIZATION_ID", "")

    max_days_raw = (
        os.environ.get("SM_HP_REVIEW_TRAINING_PARTITION_DAYS")
        or os.getenv("REVIEW_TRAINING_PARTITION_DAYS")
        or "250"
    ).strip()
    max_days = int(max_days_raw) if max_days_raw.isdigit() else 250

    from anomaly_review.partition_channel import (
        materialize_partition_snapshot,
        resolve_partition_channel_dir,
    )

    channel_dir = resolve_partition_channel_dir(production_channel, "daily_production")
    parquet_path = materialize_partition_snapshot(
        channel_dir,
        legacy_filename="daily_production.parquet",
        snapshot_name="_merged_daily_production.parquet",
        max_days=max_days,
    )

    logger.info(
        "Review training: data=%s -> model=%s (org=%s, partition_days=%s)",
        parquet_path,
        model_dir,
        org_id,
        max_days,
    )

    os.makedirs(model_dir, exist_ok=True)

    from anomaly_review.models import train_all_review_models_from_parquet

    train_all_review_models_from_parquet(parquet_path, model_dir)
    logger.info("Anomaly review training job finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Review training failed: %s", e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
