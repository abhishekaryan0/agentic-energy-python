#!/usr/bin/env python3
"""
Training module for operational recommendation models.

Uses S3 parquet data only. Data paths: well_sensor_readings.parquet and
daily_production.parquet under s3://bucket/<org_id>/sensor_data/ or local paths.

Trains action models (per lift-type + fallback) via train_operations_from_parquet.

Run from S3 partitioned data (recommended):
    python -m operation_recommendations.train_models train_action_s3 --output-dir /opt/ml/model

Or explicit s3:// parquet snapshots:
    python -m operation_recommendations.train_models train_action_parquet \\
        --sensor s3://bucket/org_id/sensor_data/well_sensor_readings.parquet \\
        --production s3://bucket/org_id/sensor_data/daily_production.parquet \\
        --output-dir /opt/ml/model

Example:
    >>> from operation_recommendations.train_models import train_operations_from_parquet
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
from operation_recommendations.utils import (
    get_project_root,
    get_models_dir,
    get_model_path as _get_model_path,
    features_from_metrics,
)

# Import configuration
from operation_recommendations.config import (
    CANONICAL_FEATURES,
    LIFT_TYPE_SENSORS,
    PRODUCTION_FIELDS,
    VALID_CATEGORIES,
    get_features_for_lift_type,
    get_sensors_for_lift_type,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
)

# Import components from operation_recommendations.recommendations subpackage
from operation_recommendations.recommendations.analyzer import TrendAnalyzer
from operation_recommendations.recommendations.rules import ACTION_MAPPING
from operation_recommendations.recommendations.persistence import _features_from_metrics

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    DecisionTreeClassifier = None
    RandomForestClassifier = None
    CalibratedClassifierCV = None
    accuracy_score = None
    joblib = None
    SKLEARN_AVAILABLE = False


def _load_parquet(path: str) -> pd.DataFrame:
    """Load parquet from local path or s3://."""
    try:
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
    except Exception as e:
        raise RuntimeError(f"Failed to load parquet: {path!r}") from e


def _fit_random_forest_classifier(X: np.ndarray, y: np.ndarray, lift_label: str):
    """
    Fit RandomForest, optionally with probability calibration when sample size allows.
    Calibration is skipped for small datasets to avoid sklearn split errors in production.
    """
    if not RandomForestClassifier:
        raise RuntimeError("sklearn RandomForestClassifier not available")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    n = len(y)
    n_classes = len(np.unique(y))
    if CalibratedClassifierCV and n >= 15:
        cv_folds = min(3, max(2, n // max(5, n_classes)))
        cv_folds = min(cv_folds, n - 1)
        if cv_folds >= 2 and n > cv_folds:
            try:
                cal = CalibratedClassifierCV(rf, method="sigmoid", cv=cv_folds)
                cal.fit(X, y)
                return cal
            except Exception as ex:
                logger.warning(
                    "CalibratedClassifierCV failed for %s (n=%s, cv=%s): %s; using uncalibrated RandomForest",
                    lift_label,
                    n,
                    cv_folds,
                    ex,
                )
    rf.fit(X, y)
    return rf


def _synthetic_action_and_category(
    metrics: Dict[str, Any],
    *,
    field_means: Dict[str, float],
    field_stds: Dict[str, float],
    zscore_threshold: float = 1.2,
) -> Tuple[str, str]:
    """Return (action_class, category) from metric z-scores vs ACTION_MAPPING."""
    candidates: List[Tuple[str, str, float, str]] = []
    for sensor_field, action_config in ACTION_MAPPING.items():
        for suffix in ("_slope", "_roll_mean", "_latest"):
            metric_key = f"{sensor_field}{suffix}"
            val = metrics.get(metric_key)
            std = field_stds.get(metric_key, 1.0)
            mean = field_means.get(metric_key, 0.0)
            if val is not None and std and abs(std) > 1e-6:
                z = (val - mean) / (std + 1e-6)
                if abs(z) > zscore_threshold:
                    action_class = action_config.get("class", f"Investigate {sensor_field}")
                    category = action_config.get("category", "Optimization")
                    candidates.append((action_class, category, abs(z), sensor_field))
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0], candidates[0][1]
    return "NoAction", "Optimization"


def _train_category_classifier(
    X: np.ndarray,
    y_categories: List[str],
    save_path: str,
    feature_names: List[str],
) -> dict:
    """Train recommendation category model (Production / Optimization / Maintenance)."""
    if len(X) < 3:
        raise RuntimeError(
            f"Need at least 3 wells to train category_classifier.joblib; got {len(X)}"
        )
    y = np.array(
        [c if c in VALID_CATEGORIES else "Optimization" for c in y_categories],
        dtype=object,
    )
    if RandomForestClassifier:
        clf = _fit_random_forest_classifier(X, y, "category")
    else:
        clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X, y)
    tr_acc = float(accuracy_score(y, clf.predict(X)))
    classes = [str(c) for c in np.unique(y)]
    logger.info(
        "Category classifier: train_accuracy=%.3f classes=%s (n=%s)",
        tr_acc,
        classes,
        len(y),
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "features": feature_names,
            "le": None,
            "classes": classes,
        },
        save_path,
    )
    if not Path(save_path).is_file():
        raise RuntimeError(f"Category model write failed: {save_path}")
    return {"train_accuracy": tr_acc, "trained_on": len(y), "classes": classes}


def _invoice_roll_features_from_history(
    inv_well: pd.DataFrame,
    *,
    as_of,
    windows: List[int],
) -> Dict[str, float]:
    """
    Sum invoice aggregates over calendar lookback windows ending at ``as_of``.

    Uses the latest production/checkpoint date as ``as_of``. When invoice parquet
    is current (``max invoice date >= as_of``), rolls are the trailing N-day sums
    ending on that date — same behavior as when financial data is live.

    When uploads lag (e.g. invoices only through 2024 while detection runs in 2026),
    the window end is clamped to ``max invoice date`` so LOE history still feeds
    implementation-cost features until fresher financial_data is uploaded.
    """
    from datetime import date, datetime, timedelta

    def _inv_keys(w: int):
        return (
            f"invoice_total_usd_roll{w}d",
            f"invoice_count_roll{w}d",
            f"workover_total_usd_roll{w}d",
            f"workover_count_roll{w}d",
        )

    out: Dict[str, float] = {}
    for w in windows:
        for key in _inv_keys(w):
            out[key] = 0.0

    if inv_well is None or inv_well.empty:
        return out

    idx = pd.to_datetime(inv_well.index, errors="coerce")
    inv_hist = inv_well.copy()
    inv_hist.index = idx
    inv_hist = inv_hist[~inv_hist.index.isna()].sort_index()
    if inv_hist.empty:
        return out

    if isinstance(as_of, datetime):
        as_of_date = as_of.date()
    elif isinstance(as_of, date):
        as_of_date = as_of
    else:
        as_of_date = pd.to_datetime(as_of).date()

    max_inv = inv_hist.index.max().date()
    effective_as_of = min(as_of_date, max_inv)
    if effective_as_of < as_of_date:
        logger.debug(
            "Invoice history ends %s before as_of %s; using trailing rolls through last invoice date",
            max_inv.isoformat(),
            as_of_date.isoformat(),
        )

    for w in windows:
        invoice_total_key, invoice_count_key, workover_total_key, workover_count_key = _inv_keys(w)
        start = effective_as_of - timedelta(days=max(0, w - 1))
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(effective_as_of)
        sl = inv_hist.loc[(inv_hist.index >= start_ts) & (inv_hist.index <= end_ts)]
        out[invoice_total_key] = float(pd.to_numeric(sl.get("invoice_total_usd"), errors="coerce").fillna(0.0).sum())
        out[invoice_count_key] = float(pd.to_numeric(sl.get("invoice_count"), errors="coerce").fillna(0.0).sum())
        out[workover_total_key] = float(pd.to_numeric(sl.get("workover_total_usd"), errors="coerce").fillna(0.0).sum())
        out[workover_count_key] = float(pd.to_numeric(sl.get("workover_count"), errors="coerce").fillna(0.0).sum())

    return out


def build_metrics_from_parquet(
    sensor_parquet_path: str,
    production_parquet_path: str,
    invoices_parquet_paths: Optional[List[str]] = None,
) -> list:
    """
    Build (well, lift_type, metrics) list from sensor and production parquet files.
    For use with train_with_synthetic_labels(..., all_metrics=...).

    If `invoices_parquet_paths` is provided, also computes invoice/workover roll features
    expected by the financial implementation-cost model:
    - invoice_total_usd_roll{window}d
    - invoice_count_roll{window}d
    - workover_total_usd_roll{window}d
    - workover_count_roll{window}d
    """
    from datetime import date, datetime, timedelta

    from operation_recommendations.well_id_utils import remap_invoice_rows_to_production_wells

    sensor_df = _load_parquet(sensor_parquet_path)
    prod_df = _load_parquet(production_parquet_path)
    min_rows = int(os.getenv("OPS_MIN_SENSOR_ROWS_PER_WELL", "24") or 24)
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

    # Invoice roll windows to compute. We compute multiple windows so whichever
    # latest financial model loads (e.g. 30d or 180d) has matching feature keys.
    windows_to_compute = [30, 180, 365]

    inv_day: Optional[pd.DataFrame] = None
    if invoices_parquet_paths:
        inv_frames: List[pd.DataFrame] = []
        for p in invoices_parquet_paths:
            try:
                inv_frames.append(_load_parquet(p))
            except Exception as e:
                logger.warning("Failed to load invoices parquet %s: %s", p, e)

        if inv_frames:
            inv_all = pd.concat(inv_frames, ignore_index=True)

            # Normalize column names and types to match financial/train_models.py expectations.
            if "asset_id" in inv_all.columns and "well_id" not in inv_all.columns:
                inv_all["well_id"] = inv_all["asset_id"].astype(str)
            elif "well_id" in inv_all.columns:
                inv_all["well_id"] = inv_all["well_id"].astype(str)
            else:
                inv_all["well_id"] = None

            if "invoice_date" in inv_all.columns:
                inv_all["invoice_date"] = pd.to_datetime(inv_all["invoice_date"], errors="coerce").dt.date
            else:
                inv_all["invoice_date"] = None

            if "total_usd" in inv_all.columns:
                inv_all["total_usd"] = pd.to_numeric(inv_all["total_usd"], errors="coerce")
            else:
                inv_all["total_usd"] = np.nan

            if "category" in inv_all.columns:
                inv_all["category"] = inv_all["category"].astype(str)
            else:
                inv_all["category"] = ""

            def _is_workover_like(category: Optional[str]) -> bool:
                if not category:
                    return False
                s = str(category).lower()
                return any(k in s for k in ("workover", "work over", "repair", "work_over"))

            inv_all["is_workover_like"] = inv_all["category"].apply(_is_workover_like)
            inv_all["total_usd_workover"] = inv_all["total_usd"].where(inv_all["is_workover_like"], 0.0)
            inv_all["workover_flag"] = inv_all["is_workover_like"].astype(int)

            # Invoice count: in our synthetic invoices, `invoice_id` is present and non-null,
            # so row-count equals invoice-count. This mirrors financial/train_models.py default behavior.
            inv_all["invoice_row_count"] = 1

            production_well_ids = set(sensor_df[well_col].dropna().astype(str))
            inv_remap = remap_invoice_rows_to_production_wells(
                inv_all["well_id"].astype(str).unique(),
                production_well_ids,
            )
            inv_all["well_id"] = inv_all["well_id"].astype(str).map(lambda w: inv_remap.get(w, w))

            inv_day = (
                inv_all.groupby(["well_id", "invoice_date"], as_index=False)
                .agg(
                    invoice_total_usd=("total_usd", "sum"),
                    invoice_count=("invoice_row_count", "sum"),
                    workover_total_usd=("total_usd_workover", "sum"),
                    workover_count=("workover_flag", "sum"),
                )
                .copy()
            )

            # Ensure numeric types.
            for c in (
                "invoice_total_usd",
                "invoice_count",
                "workover_total_usd",
                "workover_count",
            ):
                inv_day[c] = pd.to_numeric(inv_day[c], errors="coerce").fillna(0.0)
    def _inv_keys(w: int):
        return (
            f"invoice_total_usd_roll{w}d",
            f"invoice_count_roll{w}d",
            f"workover_total_usd_roll{w}d",
            f"workover_count_roll{w}d",
        )

    all_metrics = []
    for well in sensor_df[well_col].dropna().astype(str).unique():
        well_sensor = sensor_df[sensor_df[well_col].astype(str) == well].copy()
        if len(well_sensor) < min_rows:
            continue
        well_sensor["timestamp"] = pd.to_datetime(well_sensor[ts_col])
        well_sensor = well_sensor.set_index("timestamp").sort_index()
        lift_type = str(well_sensor[lift_col].mode().iloc[0]).strip() if lift_col and lift_col in well_sensor.columns else "Unknown"
        ta = TrendAnalyzer(well_sensor, lift_type=lift_type)
        metrics = ta.compute_metrics(include_all_sensors=True)
        well_prod = prod_df[prod_df[prod_well_col].astype(str) == well].copy() if prod_well_col in prod_df.columns else pd.DataFrame()

        # Convert production date to an index early so we can derive `last_date`
        # for invoice roll-window computation even when we have <3 production rows.
        if prod_date_col and prod_date_col in well_prod.columns and len(well_prod) > 0:
            well_prod[prod_date_col] = pd.to_datetime(well_prod[prod_date_col], errors="coerce")
            well_prod = well_prod.dropna(subset=[prod_date_col]).set_index(prod_date_col).sort_index()

        # Optionally compute invoice roll features for financial implementation-cost prediction.
        if invoices_parquet_paths:
            for w in windows_to_compute:
                invoice_total_key, invoice_count_key, workover_total_key, workover_count_key = _inv_keys(w)
                metrics[invoice_total_key] = 0.0
                metrics[invoice_count_key] = 0.0
                metrics[workover_total_key] = 0.0
                metrics[workover_count_key] = 0.0

            if inv_day is not None:
                as_of = date.today()
                if len(well_prod) > 0:
                    try:
                        as_of = pd.to_datetime(well_prod.index.max()).date()
                    except Exception:
                        pass
                inv_well = inv_day[inv_day["well_id"].astype(str) == str(well)].set_index("invoice_date")
                if len(inv_well) > 0:
                    roll_feats = _invoice_roll_features_from_history(
                        inv_well,
                        as_of=as_of,
                        windows=windows_to_compute,
                    )
                    metrics.update(roll_feats)

        # Production-derived signals for operational trends (slopes + short roll mean).
        if len(well_prod) >= 3 and len(well_prod.columns) > 0:
            for f, col in [
                ("oil_volume", oil_col),
                ("gas_volume", gas_col),
                ("water_volume", water_col),
            ]:
                if col in well_prod.columns:
                    yv = well_prod[col].dropna().values[::-1]
                    if len(yv) >= 3:
                        metrics[f"{f}_slope"] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                        metrics[f"{f}_roll_mean"] = float(
                            well_prod[col].rolling(window=min(7, len(well_prod))).mean().iloc[-1]
                        )

            # Latest 30 days production snapshot (mean + latest value).
            # This is used to render "Recent Production (Last 30 Days)" in the output parquet.
            try:
                recent_start = well_prod.index.max() - pd.Timedelta(days=29)
                recent_slice = well_prod.loc[well_prod.index >= recent_start]
            except Exception:
                recent_slice = None

            if recent_slice is not None and len(recent_slice) > 0:
                for f, col in [
                    ("oil_volume", oil_col),
                    ("gas_volume", gas_col),
                    ("water_volume", water_col),
                ]:
                    if col not in recent_slice.columns:
                        continue
                    s = pd.to_numeric(recent_slice[col], errors="coerce").dropna()
                    if s.empty:
                        continue
                    metrics[f"{f}_recent_30d_mean"] = float(s.mean())
                    metrics[f"{f}_recent_30d_latest"] = float(s.iloc[-1])

            # Baseline-vs-current deltas (data-driven uplift proxy).
            # These help avoid tiny “slope-only” uplifts when production is actually recoverable.
            #
            # current = last 7-day rolling mean
            # baseline = mean of prior 30 days excluding the most recent 7 days (when available)
            baseline_days = 30
            current_days = 7
            for f, col in [
                ("oil_volume", oil_col),
                ("gas_volume", gas_col),
                ("water_volume", water_col),
            ]:
                if col not in well_prod.columns:
                    continue
                s = pd.to_numeric(well_prod[col], errors="coerce").dropna()
                if s.empty:
                    continue
                cur = float(s.rolling(window=min(current_days, len(s))).mean().iloc[-1])
                base_slice = s.iloc[: max(0, len(s) - current_days)]
                if len(base_slice) >= 3:
                    base_tail = base_slice.iloc[-min(baseline_days, len(base_slice)) :]
                    baseline = float(base_tail.mean()) if len(base_tail) else float(base_slice.mean())
                else:
                    baseline = float(s.mean())
                metrics[f"{f}_baseline_mean"] = baseline
                metrics[f"{f}_current_mean"] = cur
                # Expected recoverable uplift (only positive deltas)
                metrics[f"{f}_recoverable_uplift"] = max(0.0, baseline - cur)
        all_metrics.append((well, lift_type, metrics))
    logger.info("Built metrics for %s wells from parquet", len(all_metrics))
    return all_metrics


def train_operations_from_s3(
    output_dir: str,
    *,
    bucket: Optional[str] = None,
    org: Optional[str] = None,
    max_partition_days: Optional[int] = None,
    model_path: str = "models/decision_tree_alldata.joblib",
) -> dict:
    """Train from partitioned sensor + production data on S3 (date= partitions or legacy parquet)."""
    from operation_recommendations.s3_inputs import materialize_training_parquets_from_s3

    sensor_path, prod_path = materialize_training_parquets_from_s3(
        bucket=bucket,
        org=org,
        max_partition_days=max_partition_days,
    )
    try:
        return train_operations_from_parquet(
            sensor_path,
            prod_path,
            output_dir,
            model_path=model_path,
        )
    finally:
        for p in (sensor_path, prod_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def train_operations_from_parquet(
    sensor_parquet_path: str,
    production_parquet_path: str,
    output_dir: str,
    model_path: str = "models/decision_tree_alldata.joblib",
) -> dict:
    """
    Train operations action models from parquet. Writes joblibs to output_dir.
    Sets OPERATIONS_MODELS_DIR so get_models_dir() returns output_dir. Used by SageMaker training.
    """
    os.environ["OPERATIONS_MODELS_DIR"] = output_dir
    all_metrics = build_metrics_from_parquet(sensor_parquet_path, production_parquet_path)
    if not all_metrics:
        raise ValueError(
            "No metrics from parquet. Ensure sensor parquet has well_id/asset_id, timestamp, and enough rows per well "
            "(see OPS_MIN_SENSOR_ROWS_PER_WELL, default 24). "
            "Production parquet should have date and well_id/asset_id."
        )
    if len(all_metrics) < 3:
        raise ValueError(
            f"Need at least 3 wells with sufficient data for training; got {len(all_metrics)}. "
            "Add more wells or ensure well_sensor_readings.parquet and daily_production.parquet have enough rows."
        )
    return train_with_synthetic_labels(model_path=model_path, all_metrics=all_metrics)


def train_with_synthetic_labels(
    model_path: str = 'models/decision_tree_alldata.joblib',
    all_metrics: list = None,
) -> dict:
    """Train action models (per lift-type + fallback) using data-driven synthetic labels from parquet metrics."""
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
    category_labels: List[str] = []
    for well, lift_type, metrics in all_metrics:
        lt_key = str(lift_type).strip() or 'Unknown'
        feat_names = lift_type_feature_sets[lt_key]
        feats, _ = _features_from_metrics(metrics, feature_list=feat_names)
        label, category = _synthetic_action_and_category(
            metrics, field_means=field_means, field_stds=field_stds
        )
        category_labels.append(category)
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
        X_tr, y_tr = X, y
        if RandomForestClassifier:
            clf = _fit_random_forest_classifier(X_tr, y_tr, lift_label)
        else:
            clf = DecisionTreeClassifier(max_depth=5, min_samples_split=3, min_samples_leaf=2, random_state=42, class_weight='balanced')
            clf.fit(X_tr, y_tr)
        tr_acc = float(accuracy_score(y_tr, clf.predict(X_tr)))
        logger.info(f"  {lift_label}: train_accuracy={tr_acc:.3f} (n={len(y_tr)})")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        classes_py = [str(c) for c in np.unique(y)]
        joblib.dump({
            'model': clf,
            'features': feature_names or CANONICAL_FEATURES,
            'lift_type': lift_label,
            'classes': classes_py,
        }, save_path)
        if not Path(save_path).is_file():
            raise RuntimeError(f"Model write failed: {save_path}")
        return {'train_accuracy': tr_acc, 'trained_on': len(y)}

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
    category_path = str(models_dir / "category_classifier.joblib")
    logger.info(
        "Training category classifier (%s samples, categories=%s)",
        len(X_all),
        {c: category_labels.count(c) for c in set(category_labels)},
    )
    cat_res = _train_category_classifier(
        X_all,
        category_labels,
        category_path,
        common_features,
    )
    results["category_classifier"] = {**cat_res, "model_path": category_path, "features": common_features}
    fallback = results.get('_fallback') or (list(results.values())[0] if results else {})
    if not results:
        raise RuntimeError(
            "Operations training produced no model artifacts. "
            "Need at least 3 wells with metrics, and enough samples per lift (>=5) or a successful combined model. "
            f"Wells in dataset: {len(all_metrics)}."
        )
    return {
        'model_path': model_path,
        'category_model_path': category_path,
        'per_lift_type': results,
        'label_distribution': label_dist,
        'category_distribution': {c: category_labels.count(c) for c in set(category_labels)},
        'train_accuracy': fallback.get('train_accuracy', 0),
        'category_train_accuracy': cat_res.get('train_accuracy', 0),
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

    p_s3 = sub.add_parser('train_action_s3', help='Train from S3 partitioned sensor_data (uses S3_BUCKET, ORGANIZATION_ID)')
    p_s3.add_argument('--output-dir', required=True, help='Directory to write joblib models (e.g. /opt/ml/model)')
    p_s3.add_argument('--model-path', default='models/decision_tree_alldata.joblib', help='Base model filename under output-dir')
    p_s3.add_argument('--max-partition-days', type=int, default=None, help='Limit training window (default OPS_TRAINING_PARTITION_DAYS)')

    args = parser.parse_args()

    if args.cmd == 'train_action_s3':
        model_path = os.path.join(args.output_dir, os.path.basename(args.model_path))
        train_operations_from_s3(
            args.output_dir,
            max_partition_days=args.max_partition_days,
            model_path=model_path,
        )
        return

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
