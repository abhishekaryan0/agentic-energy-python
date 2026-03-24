"""
Persistence Module
==================

Provides loading of ML models (from disk or S3) and recommend_from_metrics for
operational recommendations. Uses S3 only (no Snowflake).

Usage:
    >>> from operations.recommendations.persistence import recommend_from_metrics, load_model
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from operations.utils import (
    get_project_root,
    get_models_dir,
    get_model_path as _get_model_path,
    retry_db_operation as _retry_db_operation,
    to_native_type,
)

from operations.config import (
    CANONICAL_FEATURES,
    FEATURES,
    get_features_for_lift_type,
)

from operations.recommendations.analyzer import TrendAnalyzer
from operations.recommendations.rules import (
    RuleEngine,
    ImpactEstimator,
    ACTION_CATEGORY_MAP,
    ACTION_MAPPING,
)

logger = logging.getLogger(__name__)

# Map ML class (action) -> rule config for detailed_analysis, citations, etc.
CLASS_TO_ACTION_CONFIG: Dict[str, Dict[str, Any]] = {}
for _sensor, _cfg in (ACTION_MAPPING or {}).items():
    cls_name = _cfg.get("class")
    if cls_name and cls_name not in CLASS_TO_ACTION_CONFIG:
        CLASS_TO_ACTION_CONFIG[cls_name] = _cfg

# Project paths
_PROJECT_ROOT = get_project_root()
MODELS_DIR = get_models_dir()

# Optional ML imports
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
except Exception:
    DecisionTreeClassifier = None
    train_test_split = None
    classification_report = None
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    RandomForestClassifier = None
    CalibratedClassifierCV = None


# =============================================================================
# MODEL LOADING (models from disk or S3 via get_models_dir())
# =============================================================================

def load_model(model_path: str = 'models/decision_tree.joblib') -> Dict[str, Any]:
    """
    Load a model from path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dict with 'model' and 'features' keys
        
    Raises:
        RuntimeError: If joblib not available
        FileNotFoundError: If model file not found
    """
    if joblib is None:
        raise RuntimeError('joblib not available')
    
    resolved_path = _get_model_path(model_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(resolved_path)
    
    obj = joblib.load(resolved_path)
    if isinstance(obj, dict) and 'model' in obj:
        return obj
    return {'model': obj, 'features': CANONICAL_FEATURES}


def get_model_for_lift_type(
    lift_type: str, 
    models_dir: Optional[str] = None, 
    base_name: str = 'decision_tree_alldata', 
    fallback_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model for lift type; fall back to combined model if not found.
    
    Args:
        lift_type: Lift type name
        models_dir: Directory containing models
        base_name: Base name for model files
        fallback_path: Path to fallback model
        
    Returns:
        Dict with 'model' and 'features' keys
        
    Raises:
        RuntimeError: If joblib not available
        FileNotFoundError: If no suitable model found
    """
    if joblib is None:
        raise RuntimeError('joblib not available')
    
    if models_dir is None:
        models_dir = str(MODELS_DIR)
    elif not os.path.isabs(models_dir):
        models_dir = str(_PROJECT_ROOT / models_dir)
    
    lt_safe = (lift_type or 'unknown').strip().lower().replace(' ', '_').replace('-', '_')
    lt_path = os.path.join(models_dir, f'{base_name}_{lt_safe}.joblib')
    
    if os.path.exists(lt_path):
        obj = joblib.load(lt_path)
        return {
            'model': obj.get('model', obj) if isinstance(obj, dict) else obj,
            'features': (obj.get('features') or CANONICAL_FEATURES) if isinstance(obj, dict) else CANONICAL_FEATURES
        }
    
    fallback = fallback_path or os.path.join(models_dir, f'{base_name}.joblib')
    if os.path.exists(fallback):
        obj = joblib.load(fallback)
        return {
            'model': obj.get('model', obj) if isinstance(obj, dict) else obj,
            'features': (obj.get('features') or CANONICAL_FEATURES) if isinstance(obj, dict) else CANONICAL_FEATURES
        }
    
    raise FileNotFoundError(f"No model found for lift_type={lift_type} or fallback at {fallback}")


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_category(
    metrics: Dict[str, Any], 
    model_path: str = 'models/category_classifier.joblib'
) -> Dict[str, Any]:
    """
    Predict recommendation category from metrics using trained model.
    
    Args:
        metrics: Dict of computed metrics
        model_path: Path to category classifier model
        
    Returns:
        Dict with 'category' and 'probability' or empty dict if unavailable
    """
    if joblib is None:
        return {}
    
    try:
        resolved_path = _get_model_path(model_path)
        if not os.path.exists(resolved_path):
            return {}
        
        mdl = joblib.load(resolved_path)
        if isinstance(mdl, dict):
            clf = mdl.get('model')
            features = mdl.get('features', CANONICAL_FEATURES)
            le = mdl.get('le')
        else:
            clf = mdl
            features = CANONICAL_FEATURES
            le = None

        fv = [metrics.get(f, 0.0) for f in features]
        probs = clf.predict_proba([fv])[0]
        idx = probs.argmax()
        pred = clf.classes_[idx]
        
        if le is not None:
            try:
                pred = le.inverse_transform([pred])[0]
            except Exception:
                pass
        
        return {'category': pred, 'probability': float(probs[idx])}
    except Exception:
        return {}


def predict_with_model(
    model_path: str, 
    metrics: Dict[str, Any], 
    lift_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make prediction using specified model.
    
    Args:
        model_path: Path to model file
        metrics: Metrics dict
        lift_type: Optional lift type for feature selection
        
    Returns:
        Dict with 'prediction' and 'probability'
    """
    mdl = load_model(model_path)
    clf = mdl['model'] if isinstance(mdl, dict) else mdl
    model_features = mdl.get('features') if isinstance(mdl, dict) else None
    feats, _ = _features_from_metrics(metrics, lift_type=lift_type, feature_list=model_features)
    
    pred = clf.predict([feats])[0]
    proba = None
    try:
        proba = clf.predict_proba([feats])[0].max()
    except Exception:
        pass
    
    return {'prediction': pred, 'probability': float(proba) if proba is not None else None}


def _features_from_metrics(
    metrics: Dict[str, Any], 
    lift_type: Optional[str] = None, 
    feature_list: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert metrics dict to a flat numeric feature vector.
    
    Args:
        metrics: Dict of computed metrics
        lift_type: Optional lift type for feature selection
        feature_list: Optional explicit list of features
        
    Returns:
        Tuple of (feature_array, feature_names)
    """
    if feature_list:
        keys = feature_list
    elif lift_type:
        keys = get_features_for_lift_type(lift_type)
    else:
        keys = [k for k in metrics.keys() if k.endswith('_slope') or k.endswith('_roll_mean') or k in ['missing_rate']]
        keys.extend([k for k in metrics.keys() if k.endswith('_corr')])
        seen = set()
        keys = [k for k in keys if not (k in seen or seen.add(k))]
        if not keys:
            keys = FEATURES
    
    features = []
    for k in keys:
        v = metrics.get(k)
        try:
            features.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
        except Exception:
            features.append(0.0)
    
    return np.array(features), keys


def _get_prediction_candidates(clf, feats, threshold: float) -> List[Tuple[str, Optional[float]]]:
    """Get prediction candidates above threshold."""
    candidates = []
    try:
        probas = clf.predict_proba([feats])[0]
        classes = clf.classes_
        for cls_name, p in zip(classes, probas):
            if cls_name != 'NoAction' and p >= threshold:
                candidates.append((cls_name, float(p)))
        if not candidates:
            pass
    except Exception:
        pass
        pred = clf.predict([feats])[0]
        if pred != 'NoAction':
            candidates.append((pred, None))
    return candidates


def _has_detection_signal(metrics: Dict[str, Any]) -> bool:
    """Check if there's a detection signal in the metrics.
    
    Relaxed logic: Accept well if it has any metric data, even if no trends.
    This allows recommendations for stable wells that still need optimization.
    """
    # Check if we have any meaningful metrics
    has_prod_data = (
        metrics.get('oil_volume_slope') is not None or 
        metrics.get('gas_volume_slope') is not None
    )
    
    has_sensor_data = any(
        k.endswith('_slope') and v is not None
        for k, v in metrics.items()
    )
    
    has_correlations = any(
        k.endswith('_corr') and v is not None
        for k, v in metrics.items()
    )
    
    # Accept if we have production data OR sensor data OR correlations (relaxed check)
    # This allows stable wells to get recommendations
    return has_prod_data or has_sensor_data or has_correlations


def _create_prediction_record(
    well: str, action: str, proba: Optional[float], 
    metrics: Dict[str, Any], lift_type: str
) -> Dict[str, Any]:
    """Create a prediction record."""
    confidence = proba or 0
    if confidence > 0.55:
        priority = 'HIGH'
    elif confidence > 0.40:
        priority = 'MEDIUM'
    else:
        priority = 'LOW'
    
    rec = {
        'well_id': well,
        'action': action,
        'priority': priority,
        'status': 'New',
        'reason': f'ML model prediction (confidence: {int(confidence*100)}%)',
        'expected_impact': '',
        'confidence': proba or 0.0,
        # Align with Prisma-style schema fields when present downstream
        'confidence_percent': float(confidence * 100.0),
        'prediction_source': 'ml_model',
        'probability': proba,
        'metrics': metrics,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'id': str(uuid.uuid4()),
        'category': ACTION_CATEGORY_MAP.get(action, 'Production')
    }

    # Derive impact fields for downstream Prisma mapping.
    # Use available metrics (e.g., oil_volume_slope) as a proxy for production uplift.
    try:
        oil_slope = metrics.get('oil_volume_slope')
        oil_roll_mean = metrics.get('oil_volume_roll_mean')
        # Approximate production increase as "avoided decline" or uplift
        if oil_slope is not None:
            rec['production_increase_bbl_day'] = round(float(abs(oil_slope)), 2)
        if oil_roll_mean is not None and oil_roll_mean:
            base = float(oil_roll_mean)
            uplift = float(rec.get('production_increase_bbl_day') or 0.0)
            if base > 0:
                rec['production_increase_percent'] = round((uplift / base) * 100.0, 2)
    except Exception:
        rec.setdefault('production_increase_bbl_day', None)
        rec.setdefault('production_increase_percent', None)

    # Map confidence_percent into a rough financial impact estimate.
    # Assumes $70/bbl and simple payback with 30 days of benefit.
    try:
        bbl_per_day = float(rec.get('production_increase_bbl_day') or 0.0)
        price_per_bbl = float(os.getenv("OPERATIONS_OIL_PRICE_USD", "70") or 70)
        daily_benefit = bbl_per_day * price_per_bbl
        rec['daily_expense_benefit_usd'] = round(daily_benefit, 2) if daily_benefit else 0.0

        # Implementation cost: simple heuristic as a few days of benefit
        rec['implementation_cost_usd'] = round(daily_benefit * 3, 2) if daily_benefit else 0.0
        rec['net_daily_benefit_usd'] = round(
            daily_benefit - (rec['implementation_cost_usd'] / 30.0), 2
        ) if daily_benefit else 0.0

        # Asset value increase as 1 year of net daily benefit
        rec['asset_value_increase_usd'] = round(
            rec['net_daily_benefit_usd'] * 365.0, 2
        ) if rec['net_daily_benefit_usd'] else 0.0
    except Exception:
        rec.setdefault('daily_expense_benefit_usd', 0.0)
        rec.setdefault('implementation_cost_usd', 0.0)
        rec.setdefault('net_daily_benefit_usd', 0.0)
        rec.setdefault('asset_value_increase_usd', 0.0)

    # Enrich from rule templates (ACTION_MAPPING) so citations and narrative fields are populated.
    cfg = CLASS_TO_ACTION_CONFIG.get(action)
    if cfg:
        rec['detailed_analysis'] = cfg.get('high_action') or cfg.get('low_action') or action
        templates = cfg.get('templates') or []
        if templates:
            tpl = templates[0]
            try:
                rec['optimal_performance'] = tpl.format(current=metrics.get('oil_volume_roll_mean') or '—', target='optimal')
            except Exception:
                rec['optimal_performance'] = tpl
        else:
            rec['optimal_performance'] = cfg.get('high_action') or cfg.get('low_action')
    else:
        rec['detailed_analysis'] = rec.get('reason') or action
        rec['optimal_performance'] = None

    # Current performance: one-line summary of key metrics
    _parts = []
    for k in ('oil_volume_slope', 'oil_volume_roll_mean', 'gas_volume_slope'):
        v = metrics.get(k)
        if v is not None:
            try:
                _parts.append(f"{k}: {float(v):.2f}")
            except Exception:
                pass
    rec['current_performance'] = "; ".join(_parts) if _parts else "Metrics from sensor and production parquet"

    # Citations and references for audit/supporting analysis
    rec['citations'] = json.dumps({
        "source": "ML model + operational rule templates",
        "referenced_data": ["well_sensor_readings", "daily_production"],
        "supporting_analysis": "Recommendation from decision-tree classifier and sensor/production metrics.",
        "data_quality": "Parquet-based sensor and daily production data used for trends.",
    })
    rec['referenced_data'] = json.dumps(["well_sensor_readings", "daily_production"])
    rec['supporting_analysis'] = "Recommendation based on sensor trends, production metrics, and operational rules."
    rec['data_quality'] = "Sensor and production parquet used for metrics; ML confidence threshold applied."

    rec.setdefault('time_reduced_hours', None)
    rec.setdefault('conservative_approach', None)
    rec.setdefault('aggressive_optimization', None)
    rec.setdefault('hybrid_model', None)
    rec.setdefault('recommended_model', None)
    rec.setdefault('status_reason', None)
    rec.setdefault('mute_till', None)
    rec.setdefault('open_up', None)
    
    # Try category model
    try:
        cat_pred = predict_category(metrics)
        if cat_pred and cat_pred.get('category'):
            rec['category'] = cat_pred['category']
    except Exception:
        pass
    
    return rec


# =============================================================================
# DETECTOR: RECOMMEND FROM METRICS (NO DB – for SageMaker / stateless inference)
# =============================================================================

def recommend_from_metrics(
    well_id: str,
    lift_type: str,
    metrics: Dict[str, Any],
    confidence_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Generate operational recommendations from pre-computed metrics only (no Snowflake).
    Use this for stateless inference (e.g. SageMaker endpoint).

    Args:
        well_id: Well identifier
        lift_type: Lift type (Rod Pump, ESP, Gas Lift)
        metrics: Dict of computed metrics (slopes, roll_means, etc.)
        confidence_threshold: Minimum confidence for recommendations (0-1)

    Returns:
        List of recommendation dicts (no persistence)
    """
    if joblib is None:
        return []
    models_dir = str(get_models_dir())
    base_name = 'decision_tree_alldata'
    fallback_path = os.path.join(models_dir, f'{base_name}.joblib')
    try:
        mdl = get_model_for_lift_type(lift_type, models_dir, base_name, fallback_path)
    except FileNotFoundError:
        try:
            mdl = load_model(fallback_path)
        except Exception:
            return []
    clf = mdl['model'] if isinstance(mdl, dict) else mdl
    model_features = mdl.get('features') if isinstance(mdl, dict) else None
    feats, _ = _features_from_metrics(metrics, lift_type=lift_type, feature_list=model_features)
    candidates = _get_prediction_candidates(clf, feats, confidence_threshold)
    if not candidates or not _has_detection_signal(metrics):
        return []
    results = []
    for pred_action, proba in candidates:
        rec = _create_prediction_record(well_id, pred_action, proba, metrics, lift_type)
        results.append(rec)
    return results


# =============================================================================
# S3 OUTPUT (same pattern as anomaly review)
# =============================================================================

def write_operations_recommendations_to_s3(recommendations: List[Dict[str, Any]]) -> Optional[str]:
    """
    Write operations recommendation rows to S3 as a new run parquet.
    Path: s3://BUCKET/{ORG}/detections/operations/run_ts=YYYYMMDDTHHMMSSZ/operations_recommendations.parquet
    Returns the S3 key written, or None if bucket not set or recommendations empty.
    """
    if not recommendations:
        return None
    try:
        from operations.config import get_operations_s3_bucket, get_operations_output_prefix
    except Exception:
        return None
    bucket = get_operations_s3_bucket()
    output_prefix = get_operations_output_prefix()
    if not bucket:
        return None
    import io
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{output_prefix.rstrip('/')}/run_ts={ts}/operations_recommendations.parquet"
    # Flatten for parquet: scalar fields + metrics as JSON string.
    # Columns aligned with application schema (Prisma) where possible.
    flat_keys = (
        "id",
        "well_id",
        "category",
        "action",
        "status",
        "priority",
        "confidence_percent",
        "confidence",
        "probability",
        "reason",
        "expected_impact",
        "production_increase_bbl_day",
        "production_increase_percent",
        "daily_expense_benefit_usd",
        "implementation_cost_usd",
        "net_daily_benefit_usd",
        "asset_value_increase_usd",
        "time_reduced_hours",
        "detailed_analysis",
        "current_performance",
        "optimal_performance",
        "conservative_approach",
        "aggressive_optimization",
        "hybrid_model",
        "recommended_model",
        "citations",
        "referenced_data",
        "supporting_analysis",
        "data_quality",
        "status_reason",
        "prediction_source",
        "mute_till",
        "open_up",
        "metrics_json",
        "created_at",
        "updated_at",
        "organization_id",
    )
    rows = []
    for r in recommendations:
        row = {k: r.get(k) for k in flat_keys}
        # Backfill some derived/nullable fields when missing so parquet has consistent schema
        if row.get("confidence_percent") is None and r.get("confidence") is not None:
            try:
                row["confidence_percent"] = float(r["confidence"]) * 100.0
            except Exception:
                row["confidence_percent"] = None
        if "organization_id" not in r:
            try:
                from operations.config import get_organization_id
                row["organization_id"] = get_organization_id()
            except Exception:
                row["organization_id"] = None
        if row.get("updated_at") is None:
            row["updated_at"] = row.get("created_at")

        metrics_val = r.get("metrics")
        if metrics_val is not None:
            serialized = json.dumps(to_native_type(metrics_val), default=str)
            row["metrics_json"] = serialized
        elif row.get("metrics_json") is None:
            row["metrics_json"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    except Exception as e:
        logger.warning("Failed to write operations recommendations to S3: %s", e)
        return None
    logger.info("Wrote %s operations recommendations to s3://%s/%s", len(recommendations), bucket, key)
    return key


