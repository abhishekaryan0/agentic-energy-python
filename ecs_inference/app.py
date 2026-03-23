from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# Detection imports are lazy so we can handle missing deps if needed
from anomaly_detection.detector import AnomalyConfig, AnomalyDetector
from anomaly_review.service import detect_anomalies
from operations.recommendations.persistence import recommend_from_metrics


app = FastAPI(title="Agentic Energy ML Inference", version="1.0.0")


# Global model holders
_detector: Optional[AnomalyDetector] = None


class DetectionRequest(BaseModel):
    readings: Dict[str, Any]
    lift_type: str
    well_id: Optional[str] = None
    historical_data: Optional[List[Dict[str, Any]]] = None


class ReviewRequest(BaseModel):
    well_id: str
    save_to_db: bool = False


class OperationsRequest(BaseModel):
    well_id: str
    lift_type: str
    metrics: Dict[str, Any]
    confidence_threshold: float = 0.6


@app.on_event("startup")
def load_models() -> None:
    """
    Initialize shared detectors / env for ECS.

    Models are expected to be present on the container filesystem and wired via env:
      - ANOMALY_MODELS_DIR: base dir with anomaly_detection/ subdir.
      - MODEL_DIR: base dir for anomaly_review per-well models (optional).
      - OPERATIONS_MODELS_DIR: base dir for operations models (joblibs).

    Training jobs (SageMaker) should upload tarballs to S3; CI/CD or an init
    step should fetch and extract them into these dirs before or during task start.
    """
    global _detector

    # Anomaly detection
    anomaly_models_dir = os.environ.get("ANOMALY_MODELS_DIR")
    if anomaly_models_dir:
        os.environ["ANOMALY_MODELS_DIR"] = anomaly_models_dir
        cfg = AnomalyConfig(models_dir=anomaly_models_dir, anomaly_detection_subdir="anomaly_detection")
        _detector = AnomalyDetector(config=cfg)

    # Anomaly review: just ensure MODEL_DIR is set; detect_anomalies reads it
    review_model_dir = os.environ.get("MODEL_DIR")
    if review_model_dir:
        os.environ["MODEL_DIR"] = review_model_dir

    # Operations recommendations: models dir for recommend_from_metrics
    ops_models_dir = os.environ.get("OPERATIONS_MODELS_DIR")
    if ops_models_dir:
        os.environ["OPERATIONS_MODELS_DIR"] = ops_models_dir


@app.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    """
    Basic health check.
    """
    return {
        "status": "ok",
        "anomaly_detection_loaded": _detector is not None,
        "model_dir_review": os.environ.get("MODEL_DIR"),
        "operations_models_dir": os.environ.get("OPERATIONS_MODELS_DIR"),
    }


@app.post("/anomaly-detection", tags=["detection"])
def anomaly_detection(req: DetectionRequest) -> Dict[str, Any]:
    """
    Anomaly detection endpoint (rules + ML).

    Input matches deploy/sagemaker/inference.py:
      - readings: dict of sensor values
      - lift_type: well type (Rod Pump/ESP/Gas Lift)
      - well_id: optional
      - historical_data: optional list of past readings
    """
    if _detector is None:
        return {
            "error": "AnomalyDetector not initialized (set ANOMALY_MODELS_DIR)",
            "is_anomaly": False,
            "anomaly_score": 0.0,
        }
    result = _detector.check_anomaly(
        readings=req.readings,
        historical_data=req.historical_data,
        well_id=req.well_id,
        lift_type=req.lift_type,
    )
    return result


@app.post("/anomaly-review", tags=["review"])
def anomaly_review(req: ReviewRequest) -> Dict[str, Any]:
    """
    Anomaly review endpoint: wraps anomaly_review.service.detect_anomalies.

    Requires Snowflake/Postgres/OpenAI env when full pipeline is enabled.
    """
    if not req.well_id:
        return {"error": "well_id is required", "anomalies": [], "count": 0}
    try:
        anomalies = detect_anomalies(req.well_id, df=None, save_to_db=req.save_to_db)
    except Exception as exc:
        return {"error": str(exc), "well_id": req.well_id, "anomalies": [], "count": 0}

    # Ensure JSON-serializable
    def _clean(obj: Any) -> Any:
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
        return obj

    cleaned = _clean(anomalies)
    return {"well_id": req.well_id, "anomalies": cleaned, "count": len(cleaned) if isinstance(cleaned, list) else 0}


@app.post("/operations-recommendations", tags=["operations"])
def operations_recommendations(req: OperationsRequest) -> Dict[str, Any]:
    """
    Operations recommendations endpoint: metrics-in, recommendations-out.

    Input matches deploy/sagemaker/operations_inference.py:
      - well_id
      - lift_type
      - metrics: dict of precomputed metrics (slopes, roll_means, etc.)
      - confidence_threshold (optional)
    """
    if not req.well_id or not req.lift_type or not req.metrics:
        return {"error": "well_id, lift_type and metrics are required", "recommendations": [], "count": 0}
    try:
        recs = recommend_from_metrics(
            well_id=req.well_id,
            lift_type=req.lift_type,
            metrics=req.metrics,
            confidence_threshold=req.confidence_threshold,
        )
    except Exception as exc:
        return {"error": str(exc), "well_id": req.well_id, "recommendations": [], "count": 0}

    def _clean(obj: Any) -> Any:
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
        return obj

    cleaned = _clean(recs)
    return {"well_id": req.well_id, "recommendations": cleaned, "count": len(cleaned) if isinstance(cleaned, list) else 0}


