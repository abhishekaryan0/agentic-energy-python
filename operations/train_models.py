#!/usr/bin/env python3
"""
Training module for operational recommendation models.

Uses S3 parquet data only (no Snowflake). Data paths: well_sensor_readings.parquet and
daily_production.parquet under s3://bucket/<org_id>/sensor_data/ or local paths.

Trains action models (per lift-type + fallback) via train_operations_from_parquet.

Run from S3 parquet (recommended):
    python -m operations.train_models train_action_parquet \\
        --sensor s3://bucket/org_id/sensor_data/well_sensor_readings.parquet \\
        --production s3://bucket/org_id/sensor_data/daily_production.parquet \\
        --output-dir /opt/ml/model

Example:
    >>> from operations.train_models import train_operations_from_parquet
    >>> train_operations_from_parquet(sensor_path, production_path, output_dir)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import shared utilities
from operations.utils import (
    get_project_root,
    get_models_dir,
    get_model_path as _get_model_path,
    features_from_metrics,
)

# Import configuration
from operations.config import (
    CANONICAL_FEATURES,
    LIFT_TYPE_SENSORS,
    PRODUCTION_FIELDS,
    get_features_for_lift_type,
    get_sensors_for_lift_type,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
)

MODELS_DIR = get_models_dir()

# Import components from operations.recommendations subpackage
from operations.recommendations.analyzer import TrendAnalyzer
from operations.recommendations.rules import ACTION_MAPPING
from operations.recommendations.persistence import _features_from_metrics

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    DecisionTreeClassifier = None
    RandomForestClassifier = None
    CalibratedClassifierCV = None
    train_test_split = None
    classification_report = None
    joblib = None
    SKLEARN_AVAILABLE = False


def _load_parquet(path: str) -> pd.DataFrame:
    """Load parquet from local path or s3://."""
    if path.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        parsed = urlparse(path)
        s3 = boto3.client("s3")
        buf = __import__("io").BytesIO()
        s3.download_fileobj(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Fileobj=buf)
        buf.seek(0)
        return pd.read_parquet(buf)
    return pd.read_parquet(path)


def build_metrics_from_parquet(sensor_parquet_path: str, production_parquet_path: str) -> list:
    """
    Build (well, lift_type, metrics) list from sensor and production parquet files.
    For use with train_with_synthetic_labels(..., all_metrics=...) when running without Snowflake.
    """
    sensor_df = _load_parquet(sensor_parquet_path)
    prod_df = _load_parquet(production_parquet_path)
    well_col = "well_id" if "well_id" in sensor_df.columns else "asset_id"
    if well_col not in sensor_df.columns:
        well_col = sensor_df.columns[0]
    ts_col = next((c for c in sensor_df.columns if "timestamp" in c.lower() or c == "time"), None)
    if not ts_col:
        ts_col = sensor_df.columns[1] if len(sensor_df.columns) > 1 else sensor_df.columns[0]
    lift_col = "lift_type" if "lift_type" in sensor_df.columns else None
    prod_well_col = "well_id" if "well_id" in prod_df.columns else "asset_id"
    prod_date_col = "date" if "date" in prod_df.columns else None
    oil_col = "oil_volume" if "oil_volume" in prod_df.columns else "true_oil"
    gas_col = "gas_volume" if "gas_volume" in prod_df.columns else "true_gas"
    water_col = "water_volume" if "water_volume" in prod_df.columns else "true_water"
    all_metrics = []
    for well in sensor_df[well_col].dropna().astype(str).unique():
        well_sensor = sensor_df[sensor_df[well_col].astype(str) == well].copy()
        if len(well_sensor) < 24:
            continue
        well_sensor["timestamp"] = pd.to_datetime(well_sensor[ts_col])
        well_sensor = well_sensor.set_index("timestamp").sort_index()
        lift_type = str(well_sensor[lift_col].mode().iloc[0]).strip() if lift_col and lift_col in well_sensor.columns else "Unknown"
        ta = TrendAnalyzer(well_sensor, lift_type=lift_type)
        metrics = ta.compute_metrics(include_all_sensors=True)
        well_prod = prod_df[prod_df[prod_well_col].astype(str) == well].copy() if prod_well_col in prod_df.columns else pd.DataFrame()
        if len(well_prod) >= 3 and prod_date_col and prod_date_col in well_prod.columns:
            well_prod[prod_date_col] = pd.to_datetime(well_prod[prod_date_col])
            well_prod = well_prod.set_index(prod_date_col).sort_index()
            for f, col in [("oil_volume", oil_col), ("gas_volume", gas_col), ("water_volume", water_col)]:
                if col in well_prod.columns:
                    yv = well_prod[col].dropna().values[::-1]
                    if len(yv) >= 3:
                        metrics[f"{f}_slope"] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                        metrics[f"{f}_roll_mean"] = float(well_prod[col].rolling(window=min(7, len(well_prod))).mean().iloc[-1])
        all_metrics.append((well, lift_type, metrics))
    logger.info("Built metrics for %s wells from parquet", len(all_metrics))
    return all_metrics


def train_operations_from_parquet(
    sensor_parquet_path: str,
    production_parquet_path: str,
    output_dir: str,
    model_path: str = "models/decision_tree_alldata.joblib",
) -> dict:
    """
    Train operations action models from parquet (no Snowflake). Writes joblibs to output_dir.
    Sets OPERATIONS_MODELS_DIR so get_models_dir() returns output_dir. Used by SageMaker training.
    """
    os.environ["OPERATIONS_MODELS_DIR"] = output_dir
    all_metrics = build_metrics_from_parquet(sensor_parquet_path, production_parquet_path)
    if not all_metrics:
        raise ValueError(
            "No metrics from parquet. Ensure sensor parquet has well_id/asset_id, timestamp, and at least 24 rows per well. "
            "Production parquet should have date and well_id/asset_id."
        )
    if len(all_metrics) < 3:
        raise ValueError(
            f"Need at least 3 wells with sufficient data for training; got {len(all_metrics)}. "
            "Add more wells or ensure well_sensor_readings.parquet and daily_production.parquet have enough rows."
        )
    return train_with_synthetic_labels(snowflake_conn=None, model_path=model_path, all_metrics=all_metrics)


def train_with_synthetic_labels(
    snowflake_conn=None,
    model_path: str = 'models/decision_tree_alldata.joblib',
    all_metrics: list = None,
) -> dict:
    """Train action models (per lift-type + fallback) using data-driven synthetic labels.

    Uses S3/parquet only: call train_operations_from_parquet(sensor_path, production_path, output_dir)
    to build all_metrics from parquet. snowflake_conn is ignored (kept for API compatibility).
    """
    if not SKLEARN_AVAILABLE or DecisionTreeClassifier is None:
        raise RuntimeError('scikit-learn not available')
    if not all_metrics:
        raise ValueError(
            'Training uses S3 parquet data only. Use train_operations_from_parquet(sensor_parquet_path, '
            'production_parquet_path, output_dir) with paths to well_sensor_readings.parquet and '
            'daily_production.parquet (e.g. s3://bucket/org_id/sensor_data/...).'
        )

    resolved_path = _get_model_path(model_path)
    model_path = resolved_path

    logger.info("Using pre-built all_metrics (%s wells)", len(all_metrics))

    logger.info(f"Collected metrics for {len(all_metrics)} wells")

    # Compute field statistics across all wells for z-score normalization
    metric_keys = set()
    for _, _, m in all_metrics:
        metric_keys.update(m.keys())
    metric_keys = list(metric_keys)

    field_means = {}
    field_stds = {}
    for k in metric_keys:
        values = [m.get(k) for _, _, m in all_metrics if m.get(k) is not None]
        if values:
            field_means[k] = float(np.mean(values))
            field_stds[k] = float(np.std(values)) if len(values) > 1 else 1.0
        else:
            field_means[k] = 0.0
            field_stds[k] = 1.0

    # Determine consistent feature set for each lift type
    lift_type_feature_sets = {}
    for well, lift_type, metrics in all_metrics:
        lt_key = str(lift_type).strip() or 'Unknown'
        if lt_key not in lift_type_feature_sets:
            lift_type_feature_sets[lt_key] = get_features_for_lift_type(lt_key if lt_key != 'Unknown' else 'ESP')

    # Generate labels based on data-driven z-score deviations
    samples_by_lift = {}
    for well, lift_type, metrics in all_metrics:
        lt_key = str(lift_type).strip() or 'Unknown'
        feat_names = lift_type_feature_sets[lt_key]
        feats, _ = _features_from_metrics(metrics, feature_list=feat_names)
        candidates = []
        for sensor_field, action_config in ACTION_MAPPING.items():
            for suffix in ['_slope', '_roll_mean', '_latest']:
                metric_key = f'{sensor_field}{suffix}'
                val = metrics.get(metric_key)
                std = field_stds.get(metric_key, 1.0)
                mean = field_means.get(metric_key, 0.0)
                if val is not None and std and abs(std) > 1e-6:
                    z = (val - mean) / (std + 1e-6)
                    if abs(z) > 1.2:
                        action_class = action_config.get('class', f'Investigate {sensor_field}')
                        category = action_config.get('category', 'Optimization')
                        candidates.append((action_class, category, abs(z), sensor_field))
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            label = candidates[0][0]
        else:
            label = 'NoAction'
        samples_by_lift.setdefault(lt_key, []).append((feats, label, feat_names))

    # Common feature set for combined model
    common_features = [
        'oil_volume_slope', 'oil_volume_roll_mean',
        'gas_volume_slope', 'gas_volume_roll_mean',
        'water_volume_slope', 'water_volume_roll_mean',
        'surface_pressure_slope', 'surface_pressure_roll_mean',
        'tubing_pressure_slope', 'tubing_pressure_roll_mean',
        'casing_pressure_slope', 'casing_pressure_roll_mean',
        'wellhead_temp_slope', 'wellhead_temp_roll_mean',
        'missing_rate'
    ]
    X_all_list = []
    y_all_list = []
    sample_idx = {lt: 0 for lt in samples_by_lift}
    for well, lift_type, metrics in all_metrics:
        lt_key = str(lift_type).strip() or 'Unknown'
        feats, _ = _features_from_metrics(metrics, feature_list=common_features)
        X_all_list.append(feats)
        if lt_key in samples_by_lift and sample_idx[lt_key] < len(samples_by_lift[lt_key]):
            _, label, _ = samples_by_lift[lt_key][sample_idx[lt_key]]
            sample_idx[lt_key] += 1
        else:
            label = 'NoAction'
        y_all_list.append(label)
    X_all = np.vstack(X_all_list)
    y_all = np.array(y_all_list)
    label_dist = {k: int((y_all == k).sum()) for k in np.unique(y_all)}
    logger.info(f"Label distribution: {label_dist}")

    def _train_and_save(X, y, save_path, lift_label='combined', feature_names=None):
        if len(X) < 3:
            logger.warning(f"Insufficient samples ({len(X)}) for {lift_label}")
            return None
        stratify = y if min((y == c).sum() for c in np.unique(y)) >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
        if RandomForestClassifier and CalibratedClassifierCV:
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=3, min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1)
            try:
                cal = CalibratedClassifierCV(rf, method='sigmoid', cv=min(3, len(np.unique(y_tr))))
                cal.fit(X_tr, y_tr)
                clf = cal
            except Exception:
                rf.fit(X_tr, y_tr)
                clf = rf
        else:
            clf = DecisionTreeClassifier(max_depth=5, min_samples_split=3, min_samples_leaf=2, random_state=42, class_weight='balanced')
            clf.fit(X_tr, y_tr)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        prec, rec, _, _ = precision_recall_fscore_support(y_te, clf.predict(X_te), average='weighted', zero_division=0)
        logger.info(f"  {lift_label}: acc={te_acc:.3f}, prec={prec:.3f}, rec={rec:.3f}")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': clf,
            'features': feature_names or CANONICAL_FEATURES,
            'lift_type': lift_label,
            'classes': list(np.unique(y))
        }, save_path)
        return {'train_accuracy': tr_acc, 'test_accuracy': te_acc, 'precision': prec, 'recall': rec, 'trained_on': len(y)}

    results = {}
    models_dir = Path(model_path).parent
    base_name = Path(model_path).stem
    for lt_key, samples in samples_by_lift.items():
        if len(samples) < 5:
            logger.info(f"Skipping lift_type '{lt_key}': {len(samples)} samples (min 5)")
            continue
        X_lt = np.vstack([s[0] for s in samples])
        lt_feature_names = samples[0][2] if len(samples[0]) > 2 else None
        y_lt = np.array([s[1] for s in samples])
        lt_safe = lt_key.lower().replace(' ', '_').replace('-', '_')
        lt_path = str(models_dir / f"{base_name}_{lt_safe}.joblib")
        logger.info(f"Training model for lift_type: {lt_key} ({len(samples)} samples, {len(lt_feature_names) if lt_feature_names else 0} features)")
        res = _train_and_save(X_lt, y_lt, lt_path, lt_key, feature_names=lt_feature_names)
        if res:
            results[lt_key] = {**res, 'model_path': lt_path, 'features': lt_feature_names}
    if len(X_all) >= 3:
        logger.info(f"Training combined fallback model ({len(X_all)} samples, {len(common_features)} common features)")
        res = _train_and_save(X_all, y_all, model_path, 'combined', feature_names=common_features)
        if res:
            results['_fallback'] = {**res, 'model_path': model_path, 'features': common_features}
    fallback = results.get('_fallback') or (list(results.values())[0] if results else {})
    if not results and len(X_all) >= 3:
        logger.warning("No per-lift-type or fallback model was trained; check sample counts and labels.")
    return {
        'model_path': model_path,
        'per_lift_type': results,
        'label_distribution': label_dist,
        'train_accuracy': fallback.get('train_accuracy', 0),
        'test_accuracy': fallback.get('test_accuracy', 0),
        'trained_on': len(y_all),
    }


def main():
    parser = argparse.ArgumentParser(description='Train operational recommendation models (S3/parquet only)')
    sub = parser.add_subparsers(dest='cmd', help='Command')
    p_parquet = sub.add_parser('train_action_parquet', help='Train action models from S3/local parquet')
    p_parquet.add_argument('--sensor', required=True, help='Path to well_sensor_readings.parquet (local or s3://...)')
    p_parquet.add_argument('--production', required=True, help='Path to daily_production.parquet (local or s3://...)')
    p_parquet.add_argument('--output-dir', required=True, help='Directory to write joblib models (e.g. /opt/ml/model for SageMaker)')
    p_parquet.add_argument('--model-path', default='models/decision_tree_alldata.joblib', help='Base model path under output-dir')

    args = parser.parse_args()

    if args.cmd == 'train_action_parquet':
        model_path = os.path.join(args.output_dir, os.path.basename(args.model_path))
        train_operations_from_parquet(
            args.sensor,
            args.production,
            args.output_dir,
            model_path=model_path,
        )
        return

    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
