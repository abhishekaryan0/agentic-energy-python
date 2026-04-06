"""
SageMaker training entrypoint for operations (action recommendation models).

- Fetches data from S3 parquet: SageMaker downloads the input channel from
  s3://bucket/{org_id}/sensor_data/ (well_sensor_readings.parquet, daily_production.parquet)
  to the container. No Snowflake.
- Stores model on S3: training writes to /opt/ml/model; SageMaker uploads that
  to s3://bucket/{org_id}/model/{TrainingJobName}/output/model.tar.gz when the job completes.

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

    sensor_path = os.path.join(sensor_channel, "well_sensor_readings.parquet")
    production_path = os.path.join(sensor_channel, "daily_production.parquet")
    if not os.path.isfile(sensor_path) or not os.path.isfile(production_path):
        parquets = sorted([os.path.join(sensor_channel, f) for f in os.listdir(sensor_channel) if f.endswith(".parquet")])
        if "well_sensor_readings.parquet" in [os.path.basename(p) for p in parquets]:
            sensor_path = next(p for p in parquets if os.path.basename(p) == "well_sensor_readings.parquet")
        elif parquets:
            sensor_path = parquets[0]
        if "daily_production.parquet" in [os.path.basename(p) for p in parquets]:
            production_path = next(p for p in parquets if os.path.basename(p) == "daily_production.parquet")
        elif len(parquets) >= 2:
            production_path = parquets[1]
        if not os.path.isfile(sensor_path) or not os.path.isfile(production_path):
            logger.error("Need well_sensor_readings.parquet and daily_production.parquet in %s", sensor_channel)
            sys.exit(1)

    logger.info("Operations training: sensor=%s production=%s -> %s (org=%s)",
                sensor_path, production_path, model_dir, org_id)

    os.makedirs(model_dir, exist_ok=True)

    from operations.train_models import train_operations_from_parquet

    summary = train_operations_from_parquet(sensor_path, production_path, model_dir)
    expected = os.path.join(model_dir, "decision_tree_alldata.joblib")
    if not os.path.isfile(expected):
        logger.error("Training did not produce required artifact: %s", expected)
        sys.exit(1)
    logger.info("Operations training finished: train_accuracy=%s trained_on=%s artifact=%s",
                summary.get("train_accuracy"), summary.get("trained_on"), expected)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Operations training failed: %s", e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
