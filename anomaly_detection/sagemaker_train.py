"""
SageMaker training entrypoint for anomaly detection.

Input channel (recommended S3 prefix):
  s3://bucket/{org_id}/sensor_data/well_sensor_readings/

Files appear under the channel as:
  date=YYYY-MM-DD/part-0000.parquet

Legacy layouts also supported (sensor_data/ root with well_sensor_readings/ subdir).

Writes artifacts to /opt/ml/model (anomaly_detection/wells/...).
"""

import json
import os
import sys
import logging

_APP_ROOT = os.environ.get("SAGEMAKER_PROGRAM_DIR", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_HYPERPARAMETERS_PATH = "/opt/ml/input/config/hyperparameters.json"


def _load_hyperparameters() -> dict:
    if os.path.isfile(_HYPERPARAMETERS_PATH):
        with open(_HYPERPARAMETERS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    hp = _load_hyperparameters()
    sensor_channel = os.environ.get("SM_CHANNEL_SENSOR_DATA", "/opt/ml/input/data/sensor_data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    per_well_limit = int(
        hp.get("PER_WELL_LIMIT")
        or os.environ.get("SM_HP_PER_WELL_LIMIT")
        or "5000"
    )
    org_id = hp.get("ORGANIZATION_ID") or os.environ.get("SM_HP_ORGANIZATION_ID", "")

    partition_days = (
        hp.get("ANOMALY_SENSOR_PARTITION_DAYS")
        or os.environ.get("ANOMALY_SENSOR_PARTITION_DAYS")
        or "250"
    )
    os.environ["ANOMALY_SENSOR_PARTITION_DAYS"] = str(partition_days)

    logger.info(
        "Training: channel=%s -> model=%s (per_well_limit=%s, org=%s, partition_days=%s)",
        sensor_channel,
        model_dir,
        per_well_limit,
        org_id,
        partition_days,
    )

    from anomaly_detection.sensor_data import load_well_sensor_readings_from_local_channel
    from anomaly_detection.train_models import train_all_models_from_dataframe

    full_df = load_well_sensor_readings_from_local_channel(sensor_channel)
    if full_df.empty:
        raise RuntimeError(f"No sensor rows loaded from channel {sensor_channel}")

    logger.info("Loaded %s sensor rows for training", len(full_df))
    train_all_models_from_dataframe(full_df, model_dir, per_well_limit=per_well_limit)
    logger.info("Training job finished successfully.")


if __name__ == "__main__":
    main()
