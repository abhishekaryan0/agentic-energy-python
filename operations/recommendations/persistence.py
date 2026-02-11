"""
Persistence Module
==================

Provides functions for saving recommendations to Snowflake, loading ML models,
and making predictions across wells.

Usage:
    >>> from operations.recommendations.persistence import (
    ...     save_recommendation_to_snowflake,
    ...     predict_all_wells,
    ...     load_model
    ... )
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
)

logger = logging.getLogger(__name__)

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
# TABLE CREATION
# =============================================================================

def _create_table_sql(table_name: str = 'OPERATION_RECOMMENDATION') -> str:
    """
    Generate SQL to create the recommendation table.
    
    Args:
        table_name: Name of the table to create
        
    Returns:
        CREATE TABLE SQL statement
    """
    tn = table_name.upper()
    return f"""
    CREATE TABLE IF NOT EXISTS "{tn}" (
        id VARCHAR PRIMARY KEY,
        well_id VARCHAR NOT NULL,
        category VARCHAR,
        action VARCHAR,
        priority VARCHAR,
        status VARCHAR,
        reason VARCHAR,
        expected_impact VARCHAR,
        confidence FLOAT,
        prediction_source VARCHAR,
        probability FLOAT,
        oil_volume_slope FLOAT,
        oil_volume_roll_mean FLOAT,
        gas_volume_slope FLOAT,
        gas_volume_roll_mean FLOAT,
        motor_current_roll_mean FLOAT,
        motor_temp_roll_mean FLOAT,
        surface_pressure_roll_mean FLOAT,
        details VARCHAR,
        created_at TIMESTAMP_NTZ NOT NULL
    );
    """


# =============================================================================
# SNOWFLAKE PERSISTENCE
# =============================================================================

def save_recommendation_to_snowflake(
    conn, 
    rec: Dict[str, Any], 
    table_name: str = 'OPERATION_RECOMMENDATION', 
    metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a single recommendation dict into Snowflake.
    
    Args:
        conn: Snowflake connection with cursor/execute API
        rec: Recommendation dict to save
        table_name: Target table name
        metrics: Optional metrics dict to store alongside recommendation
        
    Returns:
        The recommendation ID (UUID string)
        
    Raises:
        Exception: If all insert attempts fail
    """
    cur = conn.cursor()
    try:
        cur.execute(_create_table_sql(table_name))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rec_id = rec.get('id') or str(uuid.uuid4())
        details = json.dumps(rec).replace("'", "''")
        metrics = metrics or rec.get('metrics', {})
        
        # Ensure category present
        if not rec.get('category'):
            rec['category'] = _infer_category(rec, metrics)
        
        tn = table_name.upper()

        # Column definitions
        orig_cols = [
            "ID", "WELL_ID", "CATEGORY", "ACTION", "PRIORITY", "STATUS", 
            "REASON", "EXPECTED_IMPACT", "CONFIDENCE", "PREDICTION_SOURCE", "PROBABILITY",
            "OIL_VOLUME_SLOPE", "OIL_VOLUME_ROLL_MEAN", "GAS_VOLUME_SLOPE", "GAS_VOLUME_ROLL_MEAN",
            "MOTOR_CURRENT_ROLL_MEAN", "MOTOR_TEMP_ROLL_MEAN", "SURFACE_PRESSURE_ROLL_MEAN", 
            "DETAILS", "CREATED_AT"
        ]

        # Build params tuple
        params = (
            rec_id,
            rec.get('well_id') or '',
            rec.get('category') or 'Production',
            rec.get('action') or '',
            rec.get('priority') or '',
            rec.get('status') or '',
            rec.get('reason') or '',
            rec.get('expected_impact') or '',
            float(rec.get('confidence') or 0.0),
            rec.get('prediction_source') or 'rule_based',
            float(rec.get('probability')) if rec.get('probability') is not None else None,
            metrics.get('oil_volume_slope'),
            metrics.get('oil_volume_roll_mean'),
            metrics.get('gas_volume_slope'),
            metrics.get('gas_volume_roll_mean'),
            metrics.get('motor_current_roll_mean'),
            metrics.get('motor_temp_roll_mean'),
            metrics.get('surface_pressure_roll_mean'),
            details,
            now
        )

        # Get existing columns
        try:
            cur.execute(f'DESCRIBE TABLE "{tn}"')
            existing_cols = [r[0].upper() for r in cur.fetchall()]
        except Exception:
            existing_cols = orig_cols

        # Add missing columns if needed
        _ensure_columns_exist(cur, conn, tn, orig_cols, existing_cols)

        # Refresh column list
        try:
            cur.execute(f'DESCRIBE TABLE "{tn}"')
            existing_cols = [r[0].upper() for r in cur.fetchall()]
        except Exception:
            pass

        insert_cols = [c for c in orig_cols if c in existing_cols]
        if not insert_cols:
            insert_cols = orig_cols

        insert_cols_sql = ', '.join([f'"{c}"' for c in insert_cols])
        placeholders = ', '.join(['%s'] * len(insert_cols))
        insert_sql = f'INSERT INTO "{tn}" ({insert_cols_sql}) VALUES ({placeholders})'

        safe_params = tuple(to_native_type(p) for p in params)
        
        def _execute_insert():
            # Log types of parameters to diagnose binding issues
            try:
                for i, p in enumerate(safe_params):
                    pass
            except Exception:
                pass
            cur.execute(insert_sql, safe_params)
            conn.commit()
            return rec_id
        
        try:
            result = _retry_db_operation(_execute_insert, max_retries=3, delay=0.5)
            return result
        except Exception:
            return _fallback_insert(cur, conn, tn, params, safe_params, orig_cols, rec_id)

    except Exception:
        raise
    finally:
        cur.close()


def _infer_category(rec: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Infer category using model if available."""
    if joblib is None:
        return rec.get('category', 'Production')
    
    try:
        model_path = os.path.join('models', 'category_classifier.joblib')
        if os.path.exists(model_path):
            mdl = joblib.load(model_path)
            features = mdl.get('features') if isinstance(mdl, dict) else None
            clf = mdl.get('model') if isinstance(mdl, dict) else mdl
            if features and clf is not None:
                fv = [metrics.get(f, 0.0) for f in features]
                try:
                    probas = clf.predict_proba([fv])[0]
                    return clf.classes_[probas.argmax()]
                except Exception:
                    return 'Production'
    except Exception:
        pass
    return rec.get('category', 'Production')


def _ensure_columns_exist(cur, conn, tn: str, orig_cols: List[str], existing_cols: List[str]) -> None:
    """Add missing columns to table."""
    missing_cols = [c for c in orig_cols if c not in existing_cols]
    if not missing_cols:
        return
    
    pass
    
    col_types = {
        "ID": "VARCHAR", "WELL_ID": "VARCHAR", "CATEGORY": "VARCHAR",
        "ACTION": "VARCHAR", "PRIORITY": "VARCHAR", "STATUS": "VARCHAR",
        "REASON": "VARCHAR", "EXPECTED_IMPACT": "VARCHAR", "CONFIDENCE": "FLOAT",
        "PREDICTION_SOURCE": "VARCHAR", "PROBABILITY": "FLOAT",
        "OIL_VOLUME_SLOPE": "FLOAT", "OIL_VOLUME_ROLL_MEAN": "FLOAT",
        "GAS_VOLUME_SLOPE": "FLOAT", "GAS_VOLUME_ROLL_MEAN": "FLOAT",
        "MOTOR_CURRENT_ROLL_MEAN": "FLOAT", "MOTOR_TEMP_ROLL_MEAN": "FLOAT",
        "SURFACE_PRESSURE_ROLL_MEAN": "FLOAT", "DETAILS": "VARCHAR",
        "CREATED_AT": "TIMESTAMP_NTZ"
    }
    
    for c in missing_cols:
        try:
            sql_type = col_types.get(c, 'VARCHAR')
            alter_sql = f'ALTER TABLE "{tn}" ADD COLUMN "{c}" {sql_type} NULL'
            cur.execute(alter_sql)
            conn.commit()
            pass
        except Exception as alter_ex:
            pass


def _fallback_insert(cur, conn, tn: str, params: tuple, safe_params: tuple, 
                     orig_cols: List[str], rec_id: str) -> str:
    """Attempt fallback insert strategies."""
    # Positional fallback
    try:
        placeholders = ','.join(['%s'] * len(params))
        alt_sql = f'INSERT INTO "{tn}" VALUES ({placeholders})'
        cur.execute(alt_sql, safe_params)
        conn.commit()
        return rec_id
    except Exception:
        pass
    
    # Adaptive fallback
    try:
        cur.execute(f'DESCRIBE TABLE "{tn}"')
        cols = [r[0].upper() for r in cur.fetchall()]
        
        param_map = {c: v for c, v in zip(orig_cols, safe_params)}
        insert_cols = [c for c in orig_cols if c in cols]
        
        if not insert_cols:
            raise RuntimeError(f"No matching columns found in table {tn}")
        
        insert_placeholders = ','.join(['%s'] * len(insert_cols))
        insert_cols_sql = ', '.join([f'"{c}"' for c in insert_cols])
        adaptive_sql = f'INSERT INTO "{tn}" ({insert_cols_sql}) VALUES ({insert_placeholders})'
        adaptive_params = tuple(param_map[c] for c in insert_cols)
        
        cur.execute(adaptive_sql, adaptive_params)
        conn.commit()
        return rec_id
    except Exception:
        raise


# =============================================================================
# MODEL LOADING
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


# =============================================================================
# BATCH PREDICTIONS
# =============================================================================

def predict_all_wells(
    snowflake_conn, 
    model_path: str = 'models/decision_tree.joblib', 
    table_name: str = 'OPERATION_RECOMMENDATION', 
    confidence_threshold: float = 0.6, 
    dedup_days: int = 7
) -> List[Dict[str, Any]]:
    """
    Run predictions for all wells and persist recommendations.
    
    Args:
        snowflake_conn: Snowflake connection
        model_path: Path to trained model
        table_name: Target table for recommendations
        confidence_threshold: Minimum confidence to persist
        dedup_days: Deduplication window in days
        
    Returns:
        List of recommendation dicts
    """
    cur = snowflake_conn.cursor()
    resolved_model_path = _get_model_path(model_path)
    models_dir = os.path.dirname(resolved_model_path)
    base_name = os.path.splitext(os.path.basename(resolved_model_path))[0]
    model_cache: Dict[str, Dict[str, Any]] = {}

    try:
        cur.execute("""
            SELECT DISTINCT wsr.well_id, COALESCE(wsr.lift_type, 'Unknown') as lift_type
            FROM well_sensor_readings wsr
            WHERE EXISTS (SELECT 1 FROM well_daily_production wdp WHERE wdp.well_id = wsr.well_id)
            ORDER BY wsr.well_id
        """)
        well_lift_pairs = [(r[0], str(r[1]).strip() if r[1] else 'Unknown') for r in cur.fetchall()]
        
        if not well_lift_pairs:
            cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
            well_lift_pairs = [(r[0], 'Unknown') for r in cur.fetchall()]
        
        pass

        results = []
        for well, lift_type in well_lift_pairs:
            try:
                result = _process_single_well(
                    cur, snowflake_conn, well, lift_type, model_cache,
                    models_dir, base_name, resolved_model_path,
                    table_name, confidence_threshold, dedup_days
                )
                results.extend(result)
            except Exception:
                continue

        pass
        return results
    finally:
        cur.close()


def _process_single_well(
    cur, conn, well: str, lift_type: str, model_cache: Dict,
    models_dir: str, base_name: str, resolved_model_path: str,
    table_name: str, confidence_threshold: float, dedup_days: int
) -> List[Dict[str, Any]]:
    """Process predictions for a single well."""
    results = []
    
    # Load model (cached)
    if lift_type not in model_cache:
        try:
            model_cache[lift_type] = get_model_for_lift_type(
                lift_type, models_dir, base_name, resolved_model_path
            )
        except FileNotFoundError:
            model_cache[lift_type] = load_model(resolved_model_path)
    
    mdl = model_cache[lift_type]
    clf = mdl['model'] if isinstance(mdl, dict) else mdl
    
    # Compute metrics
    metrics = _compute_well_metrics(cur, well)
    if metrics is None:
        return []
    
    # Make predictions
    model_features = mdl.get('features') if isinstance(mdl, dict) else None
    feats, _ = _features_from_metrics(metrics, lift_type=lift_type, feature_list=model_features)
    
    candidates = _get_prediction_candidates(clf, feats, confidence_threshold)
    if not candidates:
        return []

    if not _has_detection_signal(metrics):
        return []
    
    # Process candidates
    for pred_action, proba in candidates:
        if _is_duplicate(cur, table_name, well, pred_action, dedup_days):
            continue
        
        rec = _create_prediction_record(well, pred_action, proba, metrics, lift_type)
        results.append(rec)
        
        try:
            save_recommendation_to_snowflake(conn, rec, table_name=table_name, metrics=metrics)
        except Exception:
            pass
    
    return results


def _compute_well_metrics(cur, well: str) -> Optional[Dict[str, Any]]:
    """Compute metrics for a well."""
    cur.execute(
        "SELECT * FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 720", 
        (well,)
    )
    rows = cur.fetchall()
    if not rows:
        return None
    
    cols = [d[0].lower() for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    ta = TrendAnalyzer(df)
    metrics = ta.compute_metrics()
    
    # Add production metrics
    _add_production_metrics(cur, well, metrics)
    
    return metrics


def _add_production_metrics(cur, well: str, metrics: Dict[str, Any]) -> None:
    """Add production metrics to the metrics dict."""
    try:
        cur.execute(
            "SELECT date, oil_volume, gas_volume FROM well_daily_production "
            "WHERE well_id = %s ORDER BY date DESC LIMIT 60", 
            (well,)
        )
        prod_rows = cur.fetchall()
        if not prod_rows:
            return
        
        prod_cols = [d[0].lower() for d in cur.description]
        prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
        prod_df['date'] = pd.to_datetime(prod_df['date'])
        prod_df = prod_df.set_index('date')
        
        for field in ['oil_volume', 'gas_volume']:
            if field in prod_df.columns and len(prod_df) >= 3:
                try:
                    yv = prod_df[field].dropna().values[::-1]
                    x = np.arange(len(yv))
                    metrics[f'{field}_slope'] = float(np.polyfit(x, yv, 1)[0])
                    metrics[f'{field}_roll_mean'] = float(
                        prod_df[field].rolling(window=min(7, len(prod_df))).mean().iloc[-1]
                    )
                except Exception:
                    metrics[f'{field}_slope'] = None
                    metrics[f'{field}_roll_mean'] = None
    except Exception:
        pass


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


def _is_duplicate(cur, table_name: str, well: str, action: str, dedup_days: int) -> bool:
    """Check if recommendation is a duplicate."""
    try:
        cutoff_date = f"DATEADD(day, -{dedup_days}, CURRENT_TIMESTAMP())"
        cur.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE well_id = %s AND action = %s AND created_at > {cutoff_date}
        """, (str(well), str(action)))
        return cur.fetchone()[0] > 0
    except Exception as e:
        pass
        return False


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
        'prediction_source': 'ml_model',
        'probability': proba,
        'metrics': metrics,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'id': str(uuid.uuid4()),
        'category': ACTION_CATEGORY_MAP.get(action, 'Production')
    }
    
    # Try category model
    try:
        cat_pred = predict_category(metrics)
        if cat_pred and cat_pred.get('category'):
            rec['category'] = cat_pred['category']
    except Exception:
        pass
    
    return rec


# =============================================================================
# WELL-SPECIFIC RECOMMENDATIONS
# =============================================================================

def generate_recommendations_for_well(
    snowflake_conn, 
    well_id: str, 
    start_ts: Optional[str] = None, 
    end_ts: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate recommendations for a specific well.
    
    Args:
        snowflake_conn: Snowflake connection
        well_id: Well identifier
        start_ts: Optional start timestamp
        end_ts: Optional end timestamp
        
    Returns:
        List of saved recommendation dicts
    """
    cur = snowflake_conn.cursor()
    try:
        # Fetch sensor readings
        q = """SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, 
               motor_temp, wellhead_temp, motor_current 
               FROM well_sensor_readings WHERE well_id = %s"""
        params = [well_id]
        
        if start_ts:
            q += " AND timestamp >= %s"
            params.append(start_ts)
        if end_ts:
            q += " AND timestamp <= %s"
            params.append(end_ts)
        q += " ORDER BY timestamp"
        
        cur.execute(q, params)
        rows = cur.fetchall()
        if not rows:
            return []
        
        cols = [d[0].lower() for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        
        # Find timestamp column
        ts_col = None
        for c in df.columns:
            if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                ts_col = c
                break
        if ts_col is None:
            ts_col = df.columns[0]
        
        df['timestamp'] = pd.to_datetime(df[ts_col])
        df = df.set_index('timestamp')
        
        ta = TrendAnalyzer(df)
        metrics = ta.compute_metrics()
        
        # Add production metrics
        _add_production_metrics_for_well(cur, well_id, metrics, start_ts, end_ts)
        
        # Apply rules
        re = RuleEngine(snowflake_conn)
        raw_recs = re.apply(well_id, metrics, last_reading=df.iloc[-1].to_dict(), trend_analyzer=ta)
        
        # Estimate impact and save
        ie = ImpactEstimator()
        saved = []
        for r in raw_recs:
            impact = ie.estimate(r, metrics)
            r['impact'] = impact
            r['metrics'] = metrics
            r['created_at'] = datetime.now(timezone.utc).isoformat()
            r['id'] = str(uuid.uuid4())
            try:
                save_recommendation_to_snowflake(snowflake_conn, r)
            except Exception:
                pass
            saved.append(r)
        
        return saved
    finally:
        cur.close()


def _add_production_metrics_for_well(
    cur, well_id: str, metrics: Dict[str, Any], 
    start_ts: Optional[str], end_ts: Optional[str]
) -> None:
    """Add production metrics for a specific well with date range."""
    try:
        prod_q = "SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s"
        prod_params = [well_id]
        
        if start_ts:
            prod_q += " AND date >= %s"
            prod_params.append(start_ts.split('T')[0] if 'T' in start_ts else start_ts)
        if end_ts:
            prod_q += " AND date <= %s"
            prod_params.append(end_ts.split('T')[0] if 'T' in end_ts else end_ts)
        prod_q += " ORDER BY date"
        
        cur.execute(prod_q, prod_params)
        prod_rows = cur.fetchall()
        if not prod_rows:
            return
        
        prod_cols = [d[0] for d in cur.description]
        prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
        prod_df['date'] = pd.to_datetime(prod_df['date'])
        prod_df = prod_df.set_index('date')
        
        for field in ['oil_volume', 'gas_volume']:
            if field in prod_df.columns and len(prod_df) >= 3:
                try:
                    y = prod_df[field].dropna().values
                    x = np.arange(len(y))
                    metrics[f'{field}_slope'] = float(np.polyfit(x, y, 1)[0])
                    metrics[f'{field}_roll_mean'] = float(
                        prod_df[field].rolling(window=7).mean().iloc[-1]
                    )
                except Exception:
                    metrics[f'{field}_slope'] = None
                    metrics[f'{field}_roll_mean'] = None
    except Exception:
        pass


# =============================================================================
# DETECT AND PERSIST
# =============================================================================

def detect_and_persist(
    snowflake_conn, 
    pg_conn=None, 
    model_path: str = 'models/decision_tree.joblib',
    retrain: bool = False, 
    confidence_threshold: float = 0.6,
    snowflake_table: str = 'operational_recommendation'
) -> List[str]:
    """
    Run detection pipeline and persist recommendations.
    
    Args:
        snowflake_conn: Snowflake connection
        pg_conn: Optional PostgreSQL connection
        model_path: Path to model file
        retrain: Whether to retrain model first
        confidence_threshold: Minimum confidence to persist
        snowflake_table: Target Snowflake table
        
    Returns:
        List of persisted recommendation IDs
    """
    persisted_ids = []

    if retrain:
        try:
            from operations.recommendations.training import train_from_snowflake
            train_from_snowflake(snowflake_conn, model_path=model_path)
        except Exception:
            pass

    try:
        recs = predict_all_wells(
            snowflake_conn, 
            model_path=model_path, 
            table_name=snowflake_table, 
            confidence_threshold=confidence_threshold
        )
    except Exception:
        return persisted_ids

    # Handle PostgreSQL persistence
    local_conn_opened = False
    if pg_conn is None:
        pg_conn, local_conn_opened = _get_postgres_connection()

    for rec in recs:
        try:
            rec_id = _persist_recommendation(
                snowflake_conn, pg_conn, rec, snowflake_table, confidence_threshold
            )
            if rec_id:
                persisted_ids.append(rec_id)
        except Exception:
            pass

    if local_conn_opened and pg_conn is not None:
        try:
            pg_conn.close()
        except Exception:
            pass

    pass
    return persisted_ids


def _get_postgres_connection():
    """Get PostgreSQL connection from environment."""
    try:
        import psycopg2
        from urllib.parse import urlparse, urlunparse
        pg_url = os.getenv('DATABASE_URL')
        if pg_url:
            parsed = urlparse(pg_url)
            clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
            return psycopg2.connect(clean), True
    except Exception:
        pass
    return None, False


def _persist_recommendation(
    snowflake_conn, pg_conn, rec: Dict[str, Any], 
    snowflake_table: str, confidence_threshold: float
) -> Optional[str]:
    """Persist a single recommendation to both databases."""
    # Ensure category
    if not rec.get('category'):
        try:
            cat = predict_category(rec.get('metrics', {}))
            if cat:
                rec['category'] = cat.get('category')
                if cat.get('probability') is not None:
                    rec['probability'] = float(cat['probability'])
        except Exception:
            pass

    # Save to Snowflake
    try:
        sf_id = save_recommendation_to_snowflake(
            snowflake_conn, rec, table_name=snowflake_table, metrics=rec.get('metrics')
        )
    except Exception:
        return None

    # Persist to Postgres if available
    conf = rec.get('probability') or rec.get('confidence') or 0.0
    conf_pct = float(conf * 100.0) if conf <= 1.0 else float(conf)
    
    if pg_conn and conf_pct >= (confidence_threshold * 100.0):
        try:
            from operations.suggestions import save_operation_suggestion
            
            prod_metrics = {
                k: rec.get('metrics', {}).get(k) 
                for k in ('oil_volume_slope', 'oil_volume_roll_mean', 'gas_volume_slope', 'gas_volume_roll_mean')
            }
            sensor_metrics = {
                k: rec.get('metrics', {}).get(k) 
                for k in ('motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'missing_rate')
            }
            
            save_operation_suggestion(
                rec.get('well_id'),
                rec.get('action') or rec.get('recommendation') or 'Recommendation',
                category=rec.get('category') or 'Production',
                status=rec.get('status') or 'New',
                priority=rec.get('priority') or 'MEDIUM',
                confidence=conf_pct,
                production_data=prod_metrics,
                sensor_metrics=sensor_metrics,
                reason=rec.get('reason') or '',
                expected_impact=rec.get('expected_impact') or ''
            )
        except Exception:
            pass

    return sf_id


# =============================================================================
# TRAINING DELEGATION
# =============================================================================

def train_with_synthetic_labels(snowflake_conn, model_path: str = 'models/decision_tree_alldata.joblib'):
    """Delegate to operations.train_models. Kept for backward compatibility."""
    from operations.train_models import train_with_synthetic_labels as _train
    resolved_path = _get_model_path(model_path)
    return _train(snowflake_conn, model_path=resolved_path)


def train_from_snowflake(
    snowflake_conn, 
    lookback_days: int = 30, 
    model_path: str = 'models/decision_tree.joblib', 
    stricter: bool = False
):
    """Delegate to operations.train_models."""
    from operations.train_models import train_from_snowflake as _train
    resolved_path = _get_model_path(model_path)
    return _train(snowflake_conn, lookback_days=lookback_days, model_path=resolved_path, stricter=stricter)


def train_on_two_wells(
    snowflake_conn, 
    well_ids: List[str], 
    model_path: str = 'models/decision_tree_two_wells.joblib'
):
    """Delegate to operations.train_models."""
    from operations.train_models import train_on_two_wells as _train
    resolved_path = _get_model_path(model_path)
    return _train(snowflake_conn, well_ids, model_path=resolved_path)
