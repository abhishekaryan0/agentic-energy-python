"""
SageMaker training entrypoint for anomaly detection.

Reads sensor data from the SageMaker input channel (s3://bucket/{org_id}/sensor_data/
downloaded to /opt/ml/input/data/sensor_data/), runs per-well training, writes
artifacts to /opt/ml/model (anomaly_detection/wells/...). SageMaker uploads that to
s3://bucket/{org_id}/model/<job_name>/output/model.tar.gz.

Env (set by SageMaker or hyperparameters):
  SM_CHANNEL_SENSOR_DATA - path to sensor_data channel (e.g. /opt/ml/input/data/sensor_data)
  SM_MODEL_DIR - output path (e.g. /opt/ml/model)
  SM_HP_ORGANIZATION_ID - optional, for logging
  SM_HP_PER_WELL_LIMIT - optional, default 5000
"""

import os
import sys
import logging

# Container app is at /app
_APP_ROOT = os.environ.get("SAGEMAKER_PROGRAM_DIR", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    sensor_channel = os.environ.get("SM_CHANNEL_SENSOR_DATA", "/opt/ml/input/data/sensor_data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    per_well_limit = int(os.environ.get("SM_HP_PER_WELL_LIMIT", "5000"))
    org_id = os.environ.get("SM_HP_ORGANIZATION_ID", "")

    parquet_name = "well_sensor_readings.parquet"
    parquet_path = os.path.join(sensor_channel, parquet_name)
    if not os.path.isfile(parquet_path):
        # Allow single file in channel (SageMaker may copy the object as the file name)
        candidates = [os.path.join(sensor_channel, f) for f in os.listdir(sensor_channel) if f.endswith(".parquet")]
        if not candidates:
            logger.error("No parquet file in %s; expected %s", sensor_channel, parquet_name)
            sys.exit(1)
        parquet_path = candidates[0]
        logger.info("Using parquet: %s", parquet_path)

    logger.info("Training: data=%s -> model=%s (per_well_limit=%s, org=%s)",
                parquet_path, model_dir, per_well_limit, org_id)

    from anomaly_detection.train_models import train_all_models_from_parquet

    train_all_models_from_parquet(parquet_path, model_dir, per_well_limit=per_well_limit)
    logger.info("Training job finished successfully.")


if __name__ == "__main__":
    main()
