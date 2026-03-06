"""
SageMaker inference entrypoint for anomaly review (per-well pipeline).

SageMaker calls: model_fn(model_dir) -> input_fn(body, content_type) -> predict_fn(input, model) -> output_fn(prediction, accept).

Expects model tarball to extract so that model_dir contains per-well subdirs:
  model_dir/Well_001/
  model_dir/Well_002/
  ...
(i.e. pack with: tar -czf review_models.tar.gz -C models $(ls models) and upload)

Input JSON: { "well_id": "Well_001", "save_to_db": false }
Output: list of anomaly review items (or error). Container requires Snowflake, Postgres, OpenAI env for full pipeline.
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

# No heavy model to keep in memory; we set MODEL_DIR per request from model_dir
_model_dir = None


def model_fn(model_dir: str):
    """
    Set MODEL_DIR so anomaly_review loads per-well models from model_dir (e.g. /opt/ml/model).
    Tarball must have well_id subdirs at top level: model_dir/Well_001/, ...
    """
    global _model_dir
    try:
        _model_dir = model_dir
        os.environ["MODEL_DIR"] = str(model_dir)
        logger.info("Anomaly review model_dir set to %s", model_dir)
        return {"model_dir": model_dir}
    except Exception as e:
        logger.exception("model_fn failed: %s", e)
        raise


def input_fn(request_body, content_type):
    """Parse JSON: { well_id, save_to_db (optional, default false) }. Tolerate empty/missing well_id for health checks."""
    if content_type and content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")
    if not request_body or (isinstance(request_body, (bytes, str)) and not str(request_body).strip()):
        return {"well_id": "__ping__", "save_to_db": False}
    try:
        data = json.loads(request_body) if isinstance(request_body, (bytes, str)) else request_body
    except (TypeError, ValueError):
        return {"well_id": "__ping__", "save_to_db": False}
    if not data or not data.get("well_id"):
        # Health check often sends {} or body without well_id - treat as ping so endpoint passes
        return {"well_id": "__ping__", "save_to_db": False}
    return data


def predict_fn(input_data, model):
    """
    Run anomaly review for one well: detect_anomalies(well_id, save_to_db=...).
    Requires Snowflake, Postgres, OpenAI env in the container for full pipeline.
    Health-check / ping requests (well_id __ping__) return empty success without calling Snowflake.
    """
    well_id = input_data.get("well_id") or ""
    if well_id == "__ping__":
        return {"well_id": "__ping__", "anomalies": [], "count": 0}
    save_to_db = input_data.get("save_to_db", False)
    # Ensure MODEL_DIR is set (model is dict from model_fn)
    model_dir = model.get("model_dir") if model else None or _model_dir
    if model_dir:
        os.environ["MODEL_DIR"] = str(model_dir)

    try:
        from anomaly_review.service import detect_anomalies
        results = detect_anomalies(well_id, df=None, save_to_db=save_to_db)
    except Exception as e:
        logger.exception("detect_anomalies failed for %s", well_id)
        return {
            "error": str(e),
            "well_id": well_id,
            "anomalies": [],
            "count": 0,
        }

    # Sanitize for JSON (NaN, numpy types)
    def _clean(obj):
        if obj is None:
            return None
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):  # NaN, Inf
            return None
        return obj

    return {"well_id": well_id, "anomalies": _clean(results), "count": len(results)}


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction)
