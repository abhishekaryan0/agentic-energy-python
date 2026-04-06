"""
SageMaker training entrypoint for financial model (implementation cost).

Reads:
- daily_production.parquet from SM_CHANNEL_SENSOR_DATA
- invoice parquet(s) from SM_CHANNEL_FINANCIAL_DATA (prefix downloaded)

Writes:
- /opt/ml/model/financial_implementation_cost.joblib

SageMaker will upload /opt/ml/model as:
s3://bucket/{org_id}/model/{TrainingJobName}/output/model.tar.gz
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


def _pick_invoice_parquets(financial_channel: str) -> list[str]:
    if not os.path.isdir(financial_channel):
        raise FileNotFoundError(financial_channel)
    candidates = []
    for root, _dirs, files in os.walk(financial_channel):
        for f in files:
            if f.endswith(".parquet") and ("invoices_v1" in f or "invoices_v2" in f):
                candidates.append(os.path.join(root, f))
    # prefer latest by filename sort; keep at most one v1 and one v2
    candidates.sort(reverse=True)
    v1 = next((p for p in candidates if "invoices_v1" in os.path.basename(p)), None)
    v2 = next((p for p in candidates if "invoices_v2" in os.path.basename(p)), None)
    out = [p for p in (v1, v2) if p]
    if not out:
        raise FileNotFoundError(f"No invoices_v1/v2 parquet found under {financial_channel}")
    return out


def main():
    sensor_channel = os.environ.get("SM_CHANNEL_SENSOR_DATA", "/opt/ml/input/data/sensor_data")
    financial_channel = os.environ.get("SM_CHANNEL_FINANCIAL_DATA", "/opt/ml/input/data/financial_data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    production_path = os.path.join(sensor_channel, "daily_production.parquet")
    if not os.path.isfile(production_path):
        raise FileNotFoundError(f"Missing daily_production.parquet in {sensor_channel}")

    invoice_paths = _pick_invoice_parquets(financial_channel)

    logger.info("Financial training: production=%s invoices=%s -> %s", production_path, invoice_paths, model_dir)
    os.makedirs(model_dir, exist_ok=True)

    from financial.train_models import train_financial_from_parquet

    res = train_financial_from_parquet(production_path, invoice_paths, model_dir)
    expected = os.path.join(model_dir, "financial_implementation_cost.joblib")
    if not os.path.isfile(expected):
        logger.error("Training did not produce required artifact: %s", expected)
        sys.exit(1)
    logger.info("Financial training finished: %s (artifact=%s)", res, expected)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Financial training failed: %s", e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

