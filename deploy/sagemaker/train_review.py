"""
SageMaker training entrypoint for anomaly review (per-well RF + autoencoder).

Reads daily_production.parquet from the SageMaker input channel
(s3://bucket/{org_id}/sensor_data/ downloaded to /opt/ml/input/data/sensor_data/),
trains per-well models, writes to /opt/ml/model (Well_001/, Well_002/, ... by asset_id).

Env: SM_CHANNEL_SENSOR_DATA, SM_MODEL_DIR, SM_HP_ORGANIZATION_ID (optional).
"""

import os
import sys
import logging
import traceback

_APP_ROOT = os.environ.get("SAGEMAKER_PROGRAM_DIR", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    sensor_channel = os.environ.get("SM_CHANNEL_SENSOR_DATA", "/opt/ml/input/data/sensor_data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    org_id = os.environ.get("SM_HP_ORGANIZATION_ID", "")

    parquet_name = "daily_production.parquet"
    parquet_path = os.path.join(sensor_channel, parquet_name)
    if not os.path.isfile(parquet_path):
        candidates = [os.path.join(sensor_channel, f) for f in os.listdir(sensor_channel) if f.endswith(".parquet")]
        if not candidates:
            logger.error("No parquet in %s; expected %s", sensor_channel, parquet_name)
            sys.exit(1)
        parquet_path = candidates[0]
        logger.info("Using parquet: %s", parquet_path)

    logger.info("Review training: data=%s -> model=%s (org=%s)", parquet_path, model_dir, org_id)

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
