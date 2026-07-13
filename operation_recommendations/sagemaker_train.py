"""
SageMaker training entrypoint for operation recommendations (action models).

- Fetches data from S3 parquet: SageMaker downloads input channels from
  s3://bucket/{org_id}/sensor_data/ (well_sensor_readings, daily_production)
  to the container.
- Stores model on S3: training writes to /opt/ml/model; SageMaker uploads that
  to s3://bucket/{org_id}/model/{TrainingJobName}/output/model.tar.gz when the job completes.

Env:
  - SM_CHANNEL_SENSOR_DATA (required): local channel dir for sensor readings
  - SM_CHANNEL_PRODUCTION_DATA (required): local channel dir for daily production
  - SM_MODEL_DIR, SM_HP_ORGANIZATION_ID (optional)

Supports both layouts:
  - Legacy snapshots: well_sensor_readings.parquet + daily_production.parquet in channel root
  - Partitioned: date=YYYY-MM-DD/part-0000.parquet under channel root
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    sensor_channel = os.environ.get("SM_CHANNEL_SENSOR_DATA", "/opt/ml/input/data/sensor_data")
    production_channel = os.environ.get(
        "SM_CHANNEL_PRODUCTION_DATA", "/opt/ml/input/data/production_data"
    )
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    org_id = os.environ.get("SM_HP_ORGANIZATION_ID", "")
    if org_id:
        os.environ["ORGANIZATION_ID"] = str(org_id).strip()

    max_days_raw = (os.getenv("OPS_TRAINING_PARTITION_DAYS") or "250").strip()
    max_days = int(max_days_raw) if max_days_raw.isdigit() else 250

    from operation_recommendations.partition_channel import (
        materialize_partition_snapshot,
        resolve_partition_channel_dir,
    )

    sensor_dir = resolve_partition_channel_dir(sensor_channel, "well_sensor_readings")
    sensor_path = materialize_partition_snapshot(
        sensor_dir,
        legacy_filename="well_sensor_readings.parquet",
        snapshot_name="_merged_well_sensor_readings.parquet",
        max_days=max_days,
    )

    production_dir = resolve_partition_channel_dir(production_channel, "daily_production")
    production_path = materialize_partition_snapshot(
        production_dir,
        legacy_filename="daily_production.parquet",
        snapshot_name="_merged_daily_production.parquet",
        max_days=max_days,
    )

    logger.info(
        "Operations training: sensor=%s production=%s -> %s (org=%s)",
        sensor_path,
        production_path,
        model_dir,
        org_id,
    )

    os.makedirs(model_dir, exist_ok=True)

    from operation_recommendations.train_models import train_operations_from_parquet

    summary = train_operations_from_parquet(sensor_path, production_path, model_dir)
    expected = os.path.join(model_dir, "decision_tree_alldata.joblib")
    category_expected = os.path.join(model_dir, "category_classifier.joblib")
    if not os.path.isfile(expected):
        logger.error("Training did not produce required artifact: %s", expected)
        sys.exit(1)
    if not os.path.isfile(category_expected):
        logger.error("Training did not produce required artifact: %s", category_expected)
        sys.exit(1)
    logger.info(
        "Operations training finished: train_accuracy=%s category_accuracy=%s trained_on=%s artifacts=%s,%s",
        summary.get("train_accuracy"),
        summary.get("category_train_accuracy"),
        summary.get("trained_on"),
        expected,
        category_expected,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Operations training failed: %s", e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
