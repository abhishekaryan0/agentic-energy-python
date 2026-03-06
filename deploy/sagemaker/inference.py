"""
SageMaker inference entrypoint for anomaly detection (Option B: SageMaker + Lambda).

SageMaker calls: model_fn(model_dir) -> input_fn(body, content_type) -> predict_fn(input, model) -> output_fn(prediction, accept).

Expects model tarball to extract so that model_dir contains:
  model_dir/anomaly_detection/wells/...
  model_dir/anomaly_detection/well_types/...
(i.e. pack with: tar -czvf anomaly_detection.tar.gz -C models anomaly_detection)
"""

import os
import sys
import json
import logging

# Ensure project root is on path (container app is at /app)
_APP_ROOT = os.environ.get("SAGEMAKER_PROGRAM_DIR", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import so model_dir is set before detector loads
_detector = None


def model_fn(model_dir: str):
    """
    Load the anomaly detector. SageMaker extracts the model tarball to model_dir.
    We expect: model_dir/anomaly_detection/wells/ and model_dir/anomaly_detection/well_types/
    """
    global _detector
    os.environ["ANOMALY_MODELS_DIR"] = model_dir  # so detector._load_trained_models uses it
    from anomaly_detection.detector import AnomalyDetector, AnomalyConfig

    # model_dir is e.g. /opt/ml/model; tarball has anomaly_detection/ at top level
    config = AnomalyConfig(
        models_dir=model_dir,
        anomaly_detection_subdir="anomaly_detection",
    )
    _detector = AnomalyDetector(config=config)
    logger.info("Anomaly detector loaded for SageMaker")
    return _detector


def input_fn(request_body, content_type):
    """Parse JSON request body. Tolerate empty/invalid body for health checks."""
    if content_type and content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")
    if not request_body or (isinstance(request_body, (bytes, str)) and not str(request_body).strip()):
        return {}
    try:
        return json.loads(request_body) if isinstance(request_body, (bytes, str)) else request_body
    except (TypeError, ValueError):
        return {}


def predict_fn(input_data, model):
    """
    Run anomaly detection. input_data must have:
      - readings: dict of sensor values
      - lift_type: str (required)
      - well_id: str (optional)
      - historical_data: list of dicts (optional, for LSTM)
    Health-check payload (empty/missing) returns a safe response so the endpoint passes.
    """
    if not input_data:
        return {"error": "readings and lift_type are required", "is_anomaly": False, "anomaly_score": 0.0}
    readings = input_data.get("readings")
    lift_type = input_data.get("lift_type")
    well_id = input_data.get("well_id")
    historical_data = input_data.get("historical_data")

    if not readings or not lift_type:
        return {
            "error": "readings and lift_type are required",
            "is_anomaly": False,
            "anomaly_score": 0.0,
        }

    result = model.check_anomaly(
        readings,
        historical_data=historical_data,
        well_id=well_id,
        lift_type=lift_type,
    )
    return result


def output_fn(prediction, accept):
    """Return prediction as JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction)
