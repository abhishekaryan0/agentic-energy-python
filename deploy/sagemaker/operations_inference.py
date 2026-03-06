"""
SageMaker inference entrypoint for operational recommendations (detector).

SageMaker calls: model_fn(model_dir) -> input_fn(body, content_type) -> predict_fn(input, model) -> output_fn(prediction, accept).

Expects model tarball to contain at top level:
  model_dir/category_classifier.joblib
  model_dir/decision_tree_alldata.joblib
  model_dir/decision_tree_rod_pump.joblib (optional per lift-type)
  ...

Input JSON: { "well_id": "Well_001", "lift_type": "ESP", "metrics": { "motor_temp_slope": 0.1, ... } }
Output: { "well_id": "...", "recommendations": [...], "count": N }
No Snowflake required; metrics must be provided in the request.
"""

import os
import sys
import json
import logging

_APP_ROOT = os.environ.get("SAGEMAKER_PROGRAM_DIR", "/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_dir = None


def model_fn(model_dir: str):
    """Set OPERATIONS_MODELS_DIR so operations load models from model_dir (e.g. /opt/ml/model)."""
    global _model_dir
    _model_dir = model_dir
    os.environ["OPERATIONS_MODELS_DIR"] = model_dir
    logger.info("Operations model_dir set to %s", model_dir)
    return {"model_dir": model_dir}


def input_fn(request_body, content_type):
    """Parse JSON: { well_id, lift_type, metrics }. Tolerate empty body for health checks."""
    if content_type and content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")
    if not request_body or (isinstance(request_body, (bytes, str)) and not str(request_body).strip()):
        return {"well_id": "__ping__", "lift_type": "ESP", "metrics": {}}
    try:
        data = json.loads(request_body) if isinstance(request_body, (bytes, str)) else request_body
    except (TypeError, ValueError):
        return {"well_id": "__ping__", "lift_type": "ESP", "metrics": {}}
    if not data.get("well_id") or not data.get("lift_type") or not data.get("metrics"):
        return {"well_id": "__ping__", "lift_type": "ESP", "metrics": {}}
    return data


def predict_fn(input_data, model):
    """Run operational recommendation detector from metrics only (no DB). Ping returns empty."""
    well_id = input_data.get("well_id") or ""
    if well_id == "__ping__":
        return {"well_id": "__ping__", "recommendations": [], "count": 0}
    lift_type = input_data["lift_type"]
    metrics = input_data["metrics"]
    confidence_threshold = float(input_data.get("confidence_threshold", 0.6))
    model_dir = model.get("model_dir") or _model_dir
    if model_dir:
        os.environ["OPERATIONS_MODELS_DIR"] = model_dir

    from operations.recommendations.persistence import recommend_from_metrics

    try:
        results = recommend_from_metrics(
            well_id=well_id,
            lift_type=lift_type,
            metrics=metrics,
            confidence_threshold=confidence_threshold,
        )
    except Exception as e:
        logger.exception("recommend_from_metrics failed for %s", well_id)
        return {
            "error": str(e),
            "well_id": well_id,
            "recommendations": [],
            "count": 0,
        }

    def _clean(obj):
        if obj is None:
            return None
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return None
        return obj

    return {"well_id": well_id, "recommendations": _clean(results), "count": len(results)}


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction)
