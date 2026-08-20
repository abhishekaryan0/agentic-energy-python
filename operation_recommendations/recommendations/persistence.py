"""
Persistence Module
==================

Provides loading of ML models from S3 model.tar.gz (or local OPERATIONS_MODELS_DIR)
and recommend_from_metrics for operational recommendations. Detection output is written to S3.

Usage:
    >>> from operation_recommendations.recommendations.persistence import recommend_from_metrics, load_model
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

from operation_recommendations.utils import (
    get_project_root,
    get_models_dir,
    get_model_path as _get_model_path,
    to_native_type,
)

from operation_recommendations.config import (
    CANONICAL_FEATURES,
    FEATURES,
    get_features_for_lift_type,
    get_organization_id,
)
from operation_recommendations.recommendations.analyzer import TrendAnalyzer
from operation_recommendations.recommendations.rules import (
    RuleEngine,
    ImpactEstimator,
    ACTION_CATEGORY_MAP,
    ACTION_MAPPING,
)
from operation_recommendations.oil_price import resolve_oil_price_usd_per_bbl

logger = logging.getLogger(__name__)

# Lazy-loaded financial model (implementation cost regressor)
_FINANCIAL_MODEL_OBJ: Optional[Dict[str, Any]] = None
_FINANCIAL_MODEL_FAILED: bool = False


def _get_financial_model_obj() -> Optional[Dict[str, Any]]:
    """Load financial_implementation_cost.joblib once; None if missing or error."""
    global _FINANCIAL_MODEL_OBJ, _FINANCIAL_MODEL_FAILED
    if _FINANCIAL_MODEL_FAILED:
        return None
    if _FINANCIAL_MODEL_OBJ is not None:
        return _FINANCIAL_MODEL_OBJ
    override = (os.getenv("FINANCIAL_MODEL_PATH") or "").strip()
    try:
        from financial.persistence import load_financial_model
        if override:
            # Support either a directory or a direct joblib filepath.
            p = Path(override)
            parent = str(p.parent if p.suffix else p)
            fname = p.name if p.suffix else "financial_implementation_cost.joblib"
            _FINANCIAL_MODEL_OBJ = load_financial_model(parent, filename=fname)
        else:
            from operation_recommendations.s3_models import load_financial_joblib

            s3_obj = load_financial_joblib()
            if s3_obj is not None:
                _FINANCIAL_MODEL_OBJ = s3_obj if isinstance(s3_obj, dict) else {"model": s3_obj}
            else:
                md = get_models_dir()
                joblib_path = md / "financial_implementation_cost.joblib"
                if not joblib_path.is_file():
                    return None
                _FINANCIAL_MODEL_OBJ = load_financial_model(str(md))
        return _FINANCIAL_MODEL_OBJ
    except FileNotFoundError as e:
        _FINANCIAL_MODEL_FAILED = True
        logger.info("Financial implementation model not found: %s", e)
        return None
    except Exception as e:
        logger.warning("Financial implementation model load failed (retry on next call): %s", e)
        return None


def _metrics_to_financial_feature_values(metrics: Dict[str, Any], model_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map TrendAnalyzer-style metrics to keys expected by the financial regressor.
    Invoice/workover roll features are taken from `metrics` when present
    (computed from invoice parquets upstream). Otherwise they default to 0.0.
    """
    window = int(model_obj.get("window_days") or 30)
    oil_rm = metrics.get("oil_volume_roll_mean")
    gas_rm = metrics.get("gas_volume_roll_mean")
    wat_rm = metrics.get("water_volume_roll_mean")
    try:
        o = float(oil_rm) if oil_rm is not None and not (isinstance(oil_rm, float) and np.isnan(oil_rm)) else 0.0
    except Exception:
        o = 0.0
    try:
        g = float(gas_rm) if gas_rm is not None and not (isinstance(gas_rm, float) and np.isnan(gas_rm)) else 0.0
    except Exception:
        g = 0.0
    try:
        w = float(wat_rm) if wat_rm is not None and not (isinstance(wat_rm, float) and np.isnan(wat_rm)) else 0.0
    except Exception:
        w = 0.0
    boe = o + (g / 6.0) if g else o

    st = metrics.get("status") or metrics.get("well_status")
    is_shut = 1.0 if (st and str(st).upper().replace(" ", "_") == "SHUT_IN") else 0.0

    def _f(v: Any) -> float:
        try:
            if v is None:
                return 0.0
            fv = float(v)
            # Guard against NaN.
            return 0.0 if np.isnan(fv) else fv
        except Exception:
            return 0.0

    base: Dict[str, Any] = {
        "true_oil": o,
        "true_gas": g,
        "true_water": w,
        "oil_volume": o,
        "gas_volume": g,
        "water_volume": w,
        "boe": boe,
        "is_shut_in": is_shut,
        f"invoice_total_usd_roll{window}d": _f(metrics.get(f"invoice_total_usd_roll{window}d")),
        f"invoice_count_roll{window}d": _f(metrics.get(f"invoice_count_roll{window}d")),
        f"workover_total_usd_roll{window}d": _f(metrics.get(f"workover_total_usd_roll{window}d")),
        f"workover_count_roll{window}d": _f(metrics.get(f"workover_count_roll{window}d")),
    }
    # Fill roll features from the model window, then fall back to other windows.
    for roll_w in (window, 365, 180, 30):
        for stem in (
            "invoice_total_usd",
            "invoice_count",
            "workover_total_usd",
            "workover_count",
        ):
            key = f"{stem}_roll{roll_w}d"
            if key not in base or base[key] == 0.0:
                alt = _f(metrics.get(key))
                if alt > 0.0:
                    base[key] = alt

    out: Dict[str, Any] = {}
    for k in model_obj.get("features") or []:
        if k in base:
            out[k] = base[k]
        else:
            out[k] = 0.0
    return out


def _invoice_roll_for_window(metrics: Dict[str, Any], window: int) -> Tuple[float, float, int]:
    """Return (invoice_total_roll, workover_total_roll, window_used)."""
    for w in (window, 365, 180, 30):
        inv = metrics.get(f"invoice_total_usd_roll{w}d")
        wo = metrics.get(f"workover_total_usd_roll{w}d")
        try:
            inv_f = float(inv) if inv is not None else 0.0
        except Exception:
            inv_f = 0.0
        try:
            wo_f = float(wo) if wo is not None else 0.0
        except Exception:
            wo_f = 0.0
        if inv_f > 0.0 or wo_f > 0.0:
            return inv_f, wo_f, w
    return 0.0, 0.0, window


def apply_time_reduced_hours(rec: Dict[str, Any]) -> None:
    """Hours to recover implementation cost from daily production benefit (ECS batch path)."""
    try:
        impl = float(rec.get("implementation_cost_usd") or 0.0)
        daily_benefit = float(rec.get("daily_expense_benefit_usd") or 0.0)
        amort_days = float(os.getenv("IMPLEMENTATION_AMORTIZATION_DAYS", "180") or 180)
        if impl > 0.0 and daily_benefit > 0.0:
            recovery_days = impl / daily_benefit
            max_hours = max(1.0, amort_days * 24.0)
            hours = min(recovery_days * 24.0, max_hours)
            rec["time_reduced_hours"] = float(round(hours, 2))
        else:
            rec["time_reduced_hours"] = 0.0
    except Exception:
        rec["time_reduced_hours"] = 0.0


# Keys written by ``apply_metric_descriptions`` (also used by S3 write backfill).
METRIC_DESCRIPTION_KEYS: tuple[str, ...] = (
    "production_increase_description",
    "expense_benefit_description",
    "implementation_cost_description",
    "time_reduced_description",
    "asset_value_increase_description",
    "net_daily_benefit_description",
    "confidence_description",
)


def _fmt_desc_number(val: Any, digits: int = 0) -> Optional[str]:
    """Format a metric number for short description text; None if missing/invalid."""
    try:
        if val is None:
            return None
        fv = float(val)
        if abs(fv) < 1e-9:
            fv = 0.0
        if digits <= 0:
            return f"{fv:,.0f}"
        return f"{fv:,.{digits}f}"
    except Exception:
        return None


def _oil_price_description_clause(rec: Dict[str, Any]) -> str:
    """Phrase oil $/bbl source from org_settings (Manual) or EIA Brent/WTI keys."""
    price = _fmt_desc_number(rec.get("eia_brent_usd_per_bbl"), 2)
    if price is None:
        return ""
    src = str(rec.get("eia_brent_source") or "").strip().lower()
    period = rec.get("eia_brent_period")
    if "manual" in src:
        clause = f" using org Manual oil price of ${price}/bbl"
    elif "wti" in src:
        clause = f" using EIA WTI spot of ${price}/bbl"
    elif "brent" in src:
        clause = f" using EIA Brent spot of ${price}/bbl"
    else:
        clause = f" using oil price of ${price}/bbl"
    if period:
        clause += f" ({period})"
    return clause


def build_metric_descriptions(rec: Dict[str, Any]) -> Dict[str, str]:
    """
    Short blurbs that include this recommendation's actual impact numbers.
    Example: \"About 25 bbl/day extra production if this action is taken.\"
    Revenue text includes org Manual vs EIA Brent/WTI from stored price keys.
    """
    prod = _fmt_desc_number(rec.get("production_increase_bbl_day"), 0)
    expense = _fmt_desc_number(rec.get("daily_expense_benefit_usd"), 0)
    impl = _fmt_desc_number(rec.get("implementation_cost_usd"), 0)
    hours = _fmt_desc_number(rec.get("time_reduced_hours"), 0)
    asset = _fmt_desc_number(rec.get("asset_value_increase_usd"), 0)
    net = _fmt_desc_number(rec.get("net_daily_benefit_usd"), 0)
    pct = _fmt_desc_number(rec.get("production_increase_percent"), 1)
    price_clause = _oil_price_description_clause(rec)

    if prod is not None:
        prod_desc = f"About {prod} bbl/day extra production if this action is taken."
        if pct is not None and float(rec.get("production_increase_percent") or 0) > 0:
            prod_desc = (
                f"About {prod} bbl/day extra production (~{pct}% uplift) "
                "if this action is taken."
            )
    else:
        prod_desc = "Estimated extra oil production per day if this action is taken."

    if expense is not None:
        expense_desc = (
            f"About ${expense}/day revenue gain from the expected production increase"
            f"{price_clause}."
        )
    else:
        expense_desc = "Estimated daily revenue gain from the expected production increase."

    if impl is not None:
        impl_desc = f"About ${impl} one-time cost to carry out this recommendation."
    else:
        impl_desc = "One-time cost to carry out this recommendation."

    if hours is not None:
        time_desc = f"About {hours} hrs of unproductive time this action can recover."
    else:
        time_desc = "Estimated hours of unproductive time this action can recover."

    if asset is not None:
        asset_desc = f"About ${asset} estimated yearly gain in asset value from the net daily benefit."
    else:
        asset_desc = "Estimated yearly gain in asset value from the net daily benefit."

    if net is not None:
        net_desc = (
            f"About ${net}/day net benefit after spreading the one-time implementation cost"
            f"{price_clause}."
        )
    else:
        net_desc = "Daily benefit after spreading the one-time implementation cost."

    conf_pct: Optional[float] = None
    try:
        if rec.get("confidence_percent") is not None:
            conf_pct = float(rec["confidence_percent"])
        elif rec.get("confidence") is not None:
            c = float(rec["confidence"])
            conf_pct = c * 100.0 if c <= 1.0 else c
    except Exception:
        conf_pct = None

    if conf_pct is not None and conf_pct > 0:
        if conf_pct >= 70:
            strength = "strong"
        elif conf_pct >= 50:
            strength = "moderate"
        else:
            strength = "lower"
        conf_desc = (
            f"Model confidence is about {_fmt_desc_number(conf_pct, 0)}%, "
            f"indicating a {strength} match between well data and this recommended action."
        )
    else:
        conf_desc = (
            "Model confidence reflects how closely well sensor and production data "
            "support this recommended action."
        )

    return {
        "production_increase_description": prod_desc,
        "expense_benefit_description": expense_desc,
        "implementation_cost_description": impl_desc,
        "time_reduced_description": time_desc,
        "asset_value_increase_description": asset_desc,
        "net_daily_benefit_description": net_desc,
        "confidence_description": conf_desc,
    }


def apply_metric_descriptions(rec: Dict[str, Any]) -> None:
    """Attach short number-aware descriptions for each impact metric field."""
    rec.update(build_metric_descriptions(rec))


def apply_analysis_description(rec: Dict[str, Any]) -> None:
    """Attach OpenRouter-generated action, detailed_analysis, and analysis text fields."""
    try:
        from operation_recommendations.suggestions import generate_analysis_fields_with_openai
        # Preserve ML class as hint only; OpenRouter may replace action.
        if rec.get("action") and not rec.get("ml_action_hint"):
            rec["ml_action_hint"] = rec.get("action")
        fields = generate_analysis_fields_with_openai(rec)
        for key, value in fields.items():
            if value:
                rec[key] = value
    except Exception as e:
        # No static fallback by request; keep explicit failure marker instead.
        msg = f"OpenRouter analysis generation failed: {e}"
        for key in (
            "analysis_description",
            "sensor_evidence_description",
            "operational_interpretation_description",
            "recommended_next_step_description",
            "detailed_analysis",
        ):
            rec[key] = msg


def apply_alternatives_and_citations(
    rec: Dict[str, Any],
    *,
    financial_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Fill model-alternation fields + SPE/research citations (OpenRouter / Perplexity).

    Soft-fails: logs and leaves prior/empty values so detection still completes.
    Does not emit financial_pricing or source labels in citations.
    """
    def _t(s: Any, n: int) -> Optional[str]:
        try:
            if s is None:
                return None
            s2 = str(s)
            return s2 if len(s2) <= n else s2[:n]
        except Exception:
            return None

    try:
        from operation_recommendations.suggestions import generate_alternatives_and_citations

        enriched = generate_alternatives_and_citations(rec)
        for key in (
            "conservative_approach",
            "aggressive_optimization",
            "hybrid_model",
            "recommended_model",
            "supporting_analysis",
            "data_quality",
        ):
            val = enriched.get(key)
            if val:
                limit = 500 if key == "recommended_model" else (100 if key == "data_quality" else 2000)
                rec[key] = _t(val, limit)

        if enriched.get("citations"):
            cite_obj: Dict[str, Any]
            try:
                parsed = json.loads(enriched["citations"])
                cite_obj = parsed if isinstance(parsed, dict) else {"referenced_data": parsed}
            except Exception:
                cite_obj = {
                    "referenced_data": json.loads(enriched.get("referenced_data") or "[]"),
                    "supporting_analysis": enriched.get("supporting_analysis") or "",
                    "data_quality": enriched.get("data_quality") or "",
                }
            # Drop UI-noise keys the product does not want displayed.
            cite_obj.pop("source", None)
            cite_obj.pop("financial_pricing", None)
            # Keep only citation payload fields.
            cleaned: Dict[str, Any] = {}
            if "referenced_data" in cite_obj:
                cleaned["referenced_data"] = cite_obj["referenced_data"]
            if cite_obj.get("supporting_analysis"):
                cleaned["supporting_analysis"] = cite_obj["supporting_analysis"]
            if cite_obj.get("data_quality"):
                cleaned["data_quality"] = cite_obj["data_quality"]
            rec["citations"] = json.dumps(cleaned)
        if enriched.get("referenced_data"):
            rec["referenced_data"] = enriched["referenced_data"]
    except Exception as e:
        logger.warning(
            "Alternatives/citations enrichment failed for %s: %s",
            rec.get("well_id"),
            e,
        )
        # Keep schema stable; do not overwrite analysis fields.
        rec.setdefault("conservative_approach", None)
        rec.setdefault("aggressive_optimization", None)
        rec.setdefault("hybrid_model", None)
        rec.setdefault("recommended_model", None)


def merge_metrics_for_financial_impact(
    production_data: Optional[Dict[str, Any]],
    sensor_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge API-style production + sensor dicts into TrendAnalyzer-like keys for financial features."""
    out: Dict[str, Any] = {}
    if sensor_metrics:
        out.update(sensor_metrics)
    pd = production_data or {}
    for k, v in pd.items():
        out.setdefault(k, v)
    if "oil_volume_roll_mean" not in out:
        for alt in ("oil_roll_mean", "oil_bbl_day", "oil_rate", "avg_oil"):
            if pd.get(alt) is not None:
                try:
                    out["oil_volume_roll_mean"] = float(pd[alt])
                    break
                except Exception:
                    pass
    if "gas_volume_roll_mean" not in out:
        for alt in ("gas_roll_mean", "gas_mcf_day"):
            if pd.get(alt) is not None:
                try:
                    out["gas_volume_roll_mean"] = float(pd[alt])
                    break
                except Exception:
                    pass
    if "water_volume_roll_mean" not in out:
        for alt in ("water_roll_mean", "water_bbl_day"):
            if pd.get(alt) is not None:
                try:
                    out["water_volume_roll_mean"] = float(pd[alt])
                    break
                except Exception:
                    pass
    return out


def compute_financial_impact_fields(
    metrics: Optional[Dict[str, Any]],
    production_increase_bbl_day: Optional[float],
    organization_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Revenue and cost fields aligned with org oil price settings and the financial ML model.

    Oil price: org_settings.manual_value when oil_price_type=Manual, else EIA Brent
    (or OPERATIONS_OIL_PRICE_USD fallback when EIA is unavailable).

    - daily_expense_benefit_usd: bbl/day × resolved oil spot price.
    - implementation_cost_usd: **financial model only** — one-time USD =
      ``predict_implementation_cost_usd_per_day × IMPLEMENTATION_LUMP_DAYS`` (no revenue heuristic,
      no EIA scaling on implementation; model output is USD).
    - net_daily_benefit_usd: daily benefit − (one-time cost / amortization days).
    - asset_value_increase_usd: net_daily × 365 (simple annualization).

    If the financial model is missing or prediction fails, implementation_cost_usd is 0.

    Env:
      IMPLEMENTATION_LUMP_DAYS: multiply modeled daily rate to one-time lump (default 30).
      IMPLEMENTATION_AMORTIZATION_DAYS: amortize one-time cost for net daily (default 180).
    """
    metrics = metrics or {}
    try:
        bbl = float(production_increase_bbl_day or 0.0)
    except Exception:
        bbl = 0.0

    org_id = organization_id or get_organization_id()
    price_per_bbl, price_src, price_period, oil_price_type = resolve_oil_price_usd_per_bbl(org_id)
    daily_benefit = bbl * price_per_bbl

    lump_days = float(os.getenv("IMPLEMENTATION_LUMP_DAYS", "30") or 30)
    amort_days = float(os.getenv("IMPLEMENTATION_AMORTIZATION_DAYS", "180") or 180)

    impl_one_time = 0.0
    impl_basis = "none"
    financial_model_mode: Optional[str] = None

    mdl = _get_financial_model_obj()
    if mdl is not None:
        financial_model_mode = mdl.get("implementation_invoice_mode")
        try:
            from financial.persistence import predict_implementation_cost_usd_per_day

            fv = _metrics_to_financial_feature_values(metrics, mdl)
            daily_impl = float(predict_implementation_cost_usd_per_day(mdl, fv))
            impl_one_time = max(0.0, daily_impl) * max(1.0, lump_days)
            impl_basis = "financial_model"
        except Exception as e:
            logger.warning("Financial model prediction failed: %s", e)
            impl_basis = "financial_model_error"

    # Workover-only model often predicts 0 when invoices are LOE-only (no workover category).
    if impl_one_time <= 0.0:
        win = int(mdl.get("window_days") or 365) if mdl is not None else 365
        inv_roll, wo_roll, win_used = _invoice_roll_for_window(metrics, win)
        if wo_roll > 0.0:
            daily_proxy = wo_roll / float(max(1, win_used))
            impl_one_time = daily_proxy * max(1.0, lump_days)
            impl_basis = "workover_roll_proxy"
        elif inv_roll > 0.0:
            daily_proxy = inv_roll / float(max(1, win_used))
            impl_one_time = daily_proxy * max(1.0, lump_days)
            impl_basis = "invoice_roll_proxy"

    impl_one_time = round(max(0.0, impl_one_time), 2)
    daily_benefit_r = round(daily_benefit, 2) if daily_benefit else 0.0

    amort = max(1.0, amort_days)
    net_daily = round(daily_benefit_r - (impl_one_time / amort), 2)
    asset_val = round(net_daily * 365.0, 2) if net_daily else 0.0

    spot_price = round(price_per_bbl, 4)
    return {
        "daily_expense_benefit_usd": daily_benefit_r,
        "implementation_cost_usd": impl_one_time,
        "net_daily_benefit_usd": net_daily,
        "asset_value_increase_usd": asset_val,
        "eia_brent_usd_per_bbl": spot_price,
        "eia_brent_source": price_src,
        "eia_brent_period": price_period,
        "_financial_meta": {
            "oil_price_type": oil_price_type,
            "oil_spot_usd_per_bbl": spot_price,
            "oil_spot_source": price_src,
            "oil_spot_period": price_period,
            "implementation_cost_basis": impl_basis,
            "financial_model_invoice_mode": financial_model_mode,
            "implementation_cost_is_one_time": True,
            "implementation_amortization_days": int(amort),
            "implementation_lump_days": int(lump_days),
        },
    }


# Map ML class (action) -> rule config for detailed_analysis, citations, etc.
CLASS_TO_ACTION_CONFIG: Dict[str, Dict[str, Any]] = {}
for _sensor, _cfg in (ACTION_MAPPING or {}).items():
    cls_name = _cfg.get("class")
    if cls_name and cls_name not in CLASS_TO_ACTION_CONFIG:
        CLASS_TO_ACTION_CONFIG[cls_name] = _cfg

# Project paths
_PROJECT_ROOT = get_project_root()


def _models_dir() -> Path:
    return get_models_dir()

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
# MODEL LOADING (S3 model.tar.gz first, then OPERATIONS_MODELS_DIR / repo models/)
# =============================================================================

def _joblib_basename(model_path: str) -> str:
    return Path(model_path).name


def _load_joblib_object(model_path: str) -> Any:
    """Load joblib from local path or operations model tarball on S3."""
    if joblib is None:
        raise RuntimeError("joblib not available")

    resolved_path = _get_model_path(model_path)
    if os.path.exists(resolved_path):
        return joblib.load(resolved_path)

    from operation_recommendations.s3_models import load_operations_joblib

    member = _joblib_basename(model_path)
    obj = load_operations_joblib(member)
    if obj is None:
        raise FileNotFoundError(
            f"Model {member!r} not found locally ({resolved_path}) "
            "and no operations model tarball on S3 (set OPERATIONS_MODEL_TARBALL_S3_URI or S3_BUCKET+ORGANIZATION_ID)."
        )
    return obj


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
    obj = _load_joblib_object(model_path)
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
        models_dir = str(_models_dir())
    elif not os.path.isabs(models_dir):
        models_dir = str(_PROJECT_ROOT / models_dir)
    
    from operation_recommendations.s3_models import load_operations_model_dict

    lt_safe = (lift_type or 'unknown').strip().lower().replace(' ', '_').replace('-', '_')
    lt_member = f'{base_name}_{lt_safe}.joblib'
    lt_path = os.path.join(models_dir, lt_member)

    if os.path.exists(lt_path):
        obj = joblib.load(lt_path)
    else:
        obj = load_operations_model_dict(lt_member)

    if obj is not None:
        return {
            'model': obj.get('model', obj) if isinstance(obj, dict) else obj,
            'features': (obj.get('features') or CANONICAL_FEATURES) if isinstance(obj, dict) else CANONICAL_FEATURES
        }

    fallback = fallback_path or os.path.join(models_dir, f'{base_name}.joblib')
    if os.path.exists(fallback):
        obj = joblib.load(fallback)
    else:
        obj = load_operations_model_dict(f'{base_name}.joblib')

    if obj is not None:
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
        mdl = _load_joblib_object(model_path)
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


def derive_production_impact_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Model-aligned production uplift fields from TrendAnalyzer-style metrics (same logic as ML recommendations).
    Returns keys: production_increase_bbl_day, production_increase_percent (may be None).
    """
    out: Dict[str, Any] = {
        "production_increase_bbl_day": None,
        "production_increase_percent": None,
    }
    try:
        # Prefer baseline-vs-current recoverable uplift if present (data-driven).
        # Fallback to slope-based proxy otherwise.
        oil_rec = metrics.get("oil_volume_recoverable_uplift")
        gas_rec = metrics.get("gas_volume_recoverable_uplift")
        oil_roll_mean = metrics.get("oil_volume_roll_mean")
        gas_roll_mean = metrics.get("gas_volume_roll_mean")

        boe_uplift = None
        try:
            o = float(oil_rec) if oil_rec is not None else None
        except Exception:
            o = None
        try:
            g = float(gas_rec) if gas_rec is not None else None
        except Exception:
            g = None

        if o is not None or g is not None:
            boe_uplift = float(o or 0.0) + (float(g or 0.0) / 6.0)
        else:
            # Convert slopes into a single "bbl/day" uplift proxy using BOE convention.
            oil_slope = metrics.get("oil_volume_slope")
            gas_slope = metrics.get("gas_volume_slope")

            oil_boeday = None
            gas_boeday = None
            if oil_slope is not None:
                oil_boeday = float(abs(oil_slope))
            if gas_slope is not None:
                gas_boeday = float(abs(gas_slope)) / 6.0

            if oil_boeday is None and gas_boeday is None:
                return out

            boe_uplift = 0.0
            if oil_boeday is not None:
                boe_uplift += oil_boeday
            if gas_boeday is not None:
                boe_uplift += gas_boeday

        out["production_increase_bbl_day"] = round(float(boe_uplift), 2)

        base_oil = float(oil_roll_mean) if oil_roll_mean is not None else 0.0
        base_gas = float(gas_roll_mean) / 6.0 if gas_roll_mean is not None else 0.0
        base = base_oil + base_gas
        if base > 0:
            uplift = float(out.get("production_increase_bbl_day") or 0.0)
            out["production_increase_percent"] = round((uplift / base) * 100.0, 2)
    except Exception:
        pass
    return out


def _fmt_metric(val: Any, digits: int = 2) -> str:
    """Format numeric metric for user-visible text."""
    if val is None:
        return "—"
    try:
        fv = float(val)
        if np.isnan(fv):
            return "—"
        # Avoid "-0.00"
        if abs(fv) < 10 ** (-digits):
            fv = 0.0
        return f"{fv:.{digits}f}"
    except Exception:
        return "—"


def _build_current_vs_optimal_performance(metrics: Dict[str, Any], uplift_bbl_day: Optional[float]) -> str:
    """
    Build a "Current Average Daily Performance vs Projected Improvement Estimate" text block.

    - Current production uses `*_roll_mean` values.
    - Optimal production uses uplift applied to BOE, split into oil/gas by current proportions.
    - Sensor snapshot uses `*_latest` values from TrendAnalyzer.
    """
    try:
        uplift = float(uplift_bbl_day or 0.0)
    except Exception:
        uplift = 0.0

    oil_cur = metrics.get("oil_volume_roll_mean")
    gas_cur = metrics.get("gas_volume_roll_mean")
    water_cur = metrics.get("water_volume_roll_mean")

    # BOE components (gas converted to oil-equivalent using /6).
    oil_boed = float(oil_cur or 0.0)
    gas_boed = float(gas_cur or 0.0) / 6.0
    cur_boe = oil_boed + gas_boed
    opt_boe = cur_boe + uplift

    if cur_boe > 0:
        oil_share = oil_boed / cur_boe
        gas_share = gas_boed / cur_boe
    else:
        oil_share = 1.0
        gas_share = 0.0

    oil_opt = opt_boe * oil_share
    gas_opt_boed = opt_boe * gas_share
    gas_opt = gas_opt_boed * 6.0
    water_opt = float(water_cur or 0.0)

    motor_current = metrics.get("motor_current_latest")
    motor_temp = metrics.get("motor_temp_latest")
    surface_pressure = metrics.get("surface_pressure_latest")

    oil_slope = metrics.get("oil_volume_slope")
    gas_slope = metrics.get("gas_volume_slope")
    water_slope = metrics.get("water_volume_slope")

    # Optional rule-template target hint will be appended by caller if available.
    return "\n".join(
        [
            "Current Average Daily Performance vs Projected Improvement Estimate",
            "Performance Snapshot",
            f"Oil: {_fmt_metric(oil_cur)} bbl/d  | Optimal: {_fmt_metric(oil_opt)} bbl/d",
            f"Gas: {_fmt_metric(gas_cur)} mcf/d  | Optimal: {_fmt_metric(gas_opt)} mcf/d",
            f"Water: {_fmt_metric(water_cur)} bbl/d  | Optimal: {_fmt_metric(water_opt)} bbl/d",
            f"Motor Current: {_fmt_metric(motor_current)} A · Temp: {_fmt_metric(motor_temp)} °F",
            f"Surface Pressure: {_fmt_metric(surface_pressure)} psi",
            "Historical Trend",
            f"Oil slope: {_fmt_metric(oil_slope)}  | Gas slope: {_fmt_metric(gas_slope)}",
            f"Water slope: {_fmt_metric(water_slope)}",
        ]
    )


def _build_recent_production_30d_block(metrics: Dict[str, Any]) -> str:
    """Build "Recent Production (Last 30 Days)" block from 30d mean/latest metrics."""
    oil_mean = metrics.get("oil_volume_recent_30d_mean")
    oil_latest = metrics.get("oil_volume_recent_30d_latest")
    gas_mean = metrics.get("gas_volume_recent_30d_mean")
    gas_latest = metrics.get("gas_volume_recent_30d_latest")
    water_mean = metrics.get("water_volume_recent_30d_mean")
    water_latest = metrics.get("water_volume_recent_30d_latest")

    return "\n".join(
        [
            "Recent Production (Last 30 Days)",
            f"Oil: mean {_fmt_metric(oil_mean)} bbl/d  | latest {_fmt_metric(oil_latest)} bbl/d",
            f"Gas: mean {_fmt_metric(gas_mean)} mcf/d  | latest {_fmt_metric(gas_latest)} mcf/d",
            f"Water: mean {_fmt_metric(water_mean)} bbl/d  | latest {_fmt_metric(water_latest)} bbl/d",
        ]
    )


def _chart_point(label: str, val: Any) -> Optional[Dict[str, Any]]:
    fv = _fmt_desc_number(val, 2) if val is not None else None
    if fv is None:
        return None
    try:
        return {"label": label, "value": float(str(fv).replace(",", ""))}
    except Exception:
        return None


def build_recommendation_chart_data(metrics: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    """Trend + projection payload for the dedicated chart_data column."""
    series: List[Dict[str, Any]] = []
    prod_specs = (
        ("Oil", "oil_volume", "bbl/d"),
        ("Water", "water_volume", "bbl/d"),
        ("Gas", "gas_volume", "Mcf/d"),
    )
    for name, prefix, unit in prod_specs:
        pts: List[Dict[str, Any]] = []
        for label, suffix in (
            ("baseline", "_baseline_mean"),
            ("recent_avg", "_roll_mean"),
            ("latest", "_latest"),
        ):
            p = _chart_point(label, metrics.get(f"{prefix}{suffix}") or metrics.get(f"{prefix}_current_mean"))
            if p:
                pts.append(p)
        if len(pts) >= 2:
            series.append({"name": name, "unit": unit, "points": pts})

    sensor_keys = [
        k
        for k in metrics
        if any(str(k).endswith(s) for s in ("_latest", "_roll_mean"))
        and not any(str(k).startswith(p) for p in ("oil_volume", "gas_volume", "water_volume", "invoice", "chart"))
    ][:6]
    seen: set[str] = set()
    for k in sensor_keys:
        sk = str(k)
        if sk.endswith("_roll_mean"):
            base = sk.rsplit("_", 2)[0]
        else:
            base = sk.replace("_latest", "")
        if base in seen:
            continue
        seen.add(base)
        pts = []
        for label, suffix in (("recent_avg", "_roll_mean"), ("latest", "_latest")):
            p = _chart_point(label, metrics.get(f"{base}{suffix}"))
            if p:
                pts.append(p)
        if len(pts) >= 2:
            label = base.replace("_", " ").title()
            series.append({"name": label, "unit": "", "points": pts})

    projection = None
    uplift = rec.get("production_increase_bbl_day")
    oil_latest = metrics.get("oil_volume_latest") or metrics.get("oil_volume_roll_mean")
    try:
        if uplift and oil_latest:
            projection = {
                "name": "Projected oil after action",
                "uplift_bbl_day": round(float(uplift), 2),
                "current_bbl_day": round(float(oil_latest), 2),
                "projected_bbl_day": round(float(oil_latest) + float(uplift), 2),
            }
    except Exception:
        projection = None

    return {
        "series": series,
        "projection": projection,
        "issue": rec.get("action"),
        "recommended_action": rec.get("analysis_description"),
    }


def apply_chart_data(rec: Dict[str, Any]) -> None:
    """Set top-level chart_data (JSON string). Does not modify metrics_json contents."""
    metrics = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
    # Never keep a nested copy inside metrics — chart_data is its own column.
    if isinstance(metrics, dict) and "chart_data" in metrics:
        metrics = {k: v for k, v in metrics.items() if k != "chart_data"}
        rec["metrics"] = metrics
    chart = build_recommendation_chart_data(metrics, rec)
    if chart.get("series"):
        rec["chart_data"] = json.dumps(to_native_type(chart), default=str)
    else:
        rec.setdefault("chart_data", None)


def finalize_analysis_fields(rec: Dict[str, Any]) -> None:
    """Attach chart_data after OpenRouter analysis. Leaves other existing fields alone."""
    apply_chart_data(rec)


def _financial_pricing_sentence(meta: Dict[str, Any]) -> str:
    """Create a compact, human-readable pricing blurb (keep <80 words)."""
    if not meta or not isinstance(meta, dict):
        return ""
    price = meta.get("oil_spot_usd_per_bbl")
    src = meta.get("oil_spot_source")
    period = meta.get("oil_spot_period")
    basis = meta.get("implementation_cost_basis")
    parts: List[str] = []
    try:
        parts.append(f"Oil spot: ${float(price):.2f}/bbl")
    except Exception:
        if price is not None:
            parts.append(f"Oil spot: {price}/bbl")
    if src:
        parts.append(f"source: {src}")
    if period:
        parts.append(f"period: {period}")
    if basis:
        parts.append(f"implementation cost basis: {basis}")
    s = "; ".join(parts).strip()
    # Hard-cap to keep logs/UI readable.
    words = s.split()
    if len(words) > 80:
        s = " ".join(words[:80]).rstrip()
    return s


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
    
    _ts = datetime.now(timezone.utc).isoformat()
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
        'created_at': _ts,
        'updated_at': _ts,
        'id': str(uuid.uuid4()),
        'category': ACTION_CATEGORY_MAP.get(action, 'Production'),
        'user_id': None,
        'user_name': None,
    }

    prod_imp = derive_production_impact_from_metrics(metrics)
    rec['production_increase_bbl_day'] = prod_imp.get('production_increase_bbl_day')
    rec['production_increase_percent'] = prod_imp.get('production_increase_percent')

    # Financial impact: org_settings Manual or EIA Brent/WTI; financial model for one-time implementation.
    meta: Dict[str, Any] = {}
    try:
        fin = compute_financial_impact_fields(metrics, rec.get('production_increase_bbl_day'))
        meta = fin.pop("_financial_meta", {}) or {}
        rec['daily_expense_benefit_usd'] = fin.get('daily_expense_benefit_usd', 0.0)
        rec['implementation_cost_usd'] = fin.get('implementation_cost_usd', 0.0)
        rec['net_daily_benefit_usd'] = fin.get('net_daily_benefit_usd', 0.0)
        rec['asset_value_increase_usd'] = fin.get('asset_value_increase_usd', 0.0)
        rec['eia_brent_usd_per_bbl'] = fin.get('eia_brent_usd_per_bbl')
        rec['eia_brent_source'] = fin.get('eia_brent_source')
        rec['eia_brent_period'] = fin.get('eia_brent_period')
    except Exception:
        rec.setdefault('daily_expense_benefit_usd', 0.0)
        rec.setdefault('implementation_cost_usd', 0.0)
        rec.setdefault('net_daily_benefit_usd', 0.0)
        rec.setdefault('asset_value_increase_usd', 0.0)
        rec.setdefault('eia_brent_usd_per_bbl', None)
        rec.setdefault('eia_brent_source', None)
        rec.setdefault('eia_brent_period', None)

    # Recompute net daily after invoice-roll proxy may have set implementation cost.
    try:
        impl = float(rec.get("implementation_cost_usd") or 0.0)
        daily_benefit = float(rec.get("daily_expense_benefit_usd") or 0.0)
        amort_days = float(os.getenv("IMPLEMENTATION_AMORTIZATION_DAYS", "180") or 180)
        rec["net_daily_benefit_usd"] = round(
            daily_benefit - (impl / max(1.0, amort_days)), 2
        )
        nd = float(rec["net_daily_benefit_usd"] or 0.0)
        rec["asset_value_increase_usd"] = round(nd * 365.0, 2) if nd else 0.0
    except Exception:
        pass

    apply_time_reduced_hours(rec)
    apply_metric_descriptions(rec)

    # Keep ML class as a hint only. OpenRouter generates smart action + detailed_analysis.
    rec["ml_action_hint"] = action
    cfg = CLASS_TO_ACTION_CONFIG.get(action)
    target_hint: Optional[str] = None
    if cfg:
        templates = cfg.get("templates") or []
        if templates:
            tpl = templates[0]
            try:
                raw_val = metrics.get("oil_volume_roll_mean") or "—"
                try:
                    cur_val = round(float(raw_val), 2)
                except Exception:
                    cur_val = raw_val
                target_hint = tpl.format(current=cur_val, target="optimal")
            except Exception:
                target_hint = tpl
        else:
            target_hint = cfg.get("high_action") or cfg.get("low_action")

    # OpenRouter fills action (= what), detailed_analysis (= why), and narrative fields.
    apply_analysis_description(rec)
    finalize_analysis_fields(rec)

    # Build user-visible performance blocks.
    current_vs_opt = _build_current_vs_optimal_performance(metrics, rec.get("production_increase_bbl_day"))
    if target_hint:
        current_vs_opt = current_vs_opt + f"\nOptimal Target: {target_hint}"
    rec["current_performance"] = current_vs_opt
    rec["historical_trend"] = _build_recent_production_30d_block(metrics)

    # Model alternation + SPE/research citations (replaces static placeholders).
    rec.setdefault("conservative_approach", None)
    rec.setdefault("aggressive_optimization", None)
    rec.setdefault("hybrid_model", None)
    rec.setdefault("recommended_model", None)
    rec.setdefault(
        "citations",
        json.dumps(
            {
                "referenced_data": ["well_sensor_readings", "daily_production"],
                "supporting_analysis": "Recommendation from decision-tree classifier and sensor/production metrics.",
                "data_quality": "Parquet-based sensor and daily production data used for trends.",
            }
        ),
    )
    rec.setdefault("referenced_data", json.dumps(["well_sensor_readings", "daily_production"]))
    rec.setdefault(
        "supporting_analysis",
        "Recommendation based on sensor trends, production metrics, and operational rules.",
    )
    rec.setdefault(
        "data_quality",
        "Sensor and production parquet used for metrics; ML confidence threshold applied.",
    )
    apply_alternatives_and_citations(rec)

    # Keep output schema stable: never emit null for numeric columns.
    # `time_reduced_hours` is deterministically computed later in the pipeline; if unavailable, default to 0.0.
    rec.setdefault('time_reduced_hours', 0.0)
    rec.setdefault('status_reason', None)
    rec.setdefault('mute_till', None)
    rec.setdefault('open_up', None)
    rec.setdefault('user_id', None)
    rec.setdefault('user_name', None)
    
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
    Generate operational recommendations from pre-computed metrics only (S3 / stateless).
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
        # Production-ready acceptance rules (model-driven economics):
        # - Don't recommend if predicted production uplift is tiny/zero.
        # - Don't recommend if net daily benefit is not positive.
        min_uplift = float(os.getenv("OPS_MIN_PRODUCTION_INCREASE_BBL_DAY", "0.5") or 0.5)
        try:
            uplift = float(rec.get("production_increase_bbl_day") or 0.0)
        except Exception:
            uplift = 0.0
        try:
            net = float(rec.get("net_daily_benefit_usd") or 0.0)
        except Exception:
            net = 0.0
        if uplift < min_uplift:
            continue
        if net <= 0:
            continue
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
        from operation_recommendations.config import get_operations_s3_bucket, get_operations_output_prefix
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
        "confidence_description",
        "reason",
        "expected_impact",
        "production_increase_bbl_day",
        "production_increase_percent",
        "production_increase_description",
        "daily_expense_benefit_usd",
        "expense_benefit_description",
        "implementation_cost_usd",
        "implementation_cost_description",
        "net_daily_benefit_usd",
        "net_daily_benefit_description",
        "asset_value_increase_usd",
        "asset_value_increase_description",
        "eia_brent_usd_per_bbl",
        "eia_brent_source",
        "eia_brent_period",
        "time_reduced_hours",
        "time_reduced_description",
        "analysis_description",
        "sensor_evidence_description",
        "operational_interpretation_description",
        "recommended_next_step_description",
        "detailed_analysis",
        "current_performance",
        "historical_trend",
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
        "chart_data",
        "user_id",
        "user_name",
        "created_at",
        "updated_at",
        "organization_id",
    )
    # Numeric fields to round to 2 decimal places
    _NUMERIC_FIELDS = {
        "confidence_percent",
        "confidence",
        "probability",
        "production_increase_bbl_day",
        "production_increase_percent",
        "daily_expense_benefit_usd",
        "implementation_cost_usd",
        "net_daily_benefit_usd",
        "asset_value_increase_usd",
        "eia_brent_usd_per_bbl",
        "time_reduced_hours",
    }

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
                from operation_recommendations.config import get_organization_id
                row["organization_id"] = get_organization_id()
            except Exception:
                row["organization_id"] = None
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(timezone.utc).isoformat()
        if row.get("updated_at") is None:
            row["updated_at"] = row.get("created_at")

        # Ensure metric description fields are always present in parquet output.
        apply_metric_descriptions(row)
        if any(
            not row.get(k)
            for k in (
                "analysis_description",
                "sensor_evidence_description",
                "operational_interpretation_description",
                "recommended_next_step_description",
            )
        ):
            apply_analysis_description(row)
        # Backfill model alternation + citations when missing (production UI fields).
        if any(
            not row.get(k)
            for k in (
                "conservative_approach",
                "aggressive_optimization",
                "hybrid_model",
                "recommended_model",
            )
        ):
            apply_alternatives_and_citations(row)

        metrics_val = r.get("metrics")
        if isinstance(metrics_val, dict) and "chart_data" in metrics_val:
            metrics_val = {k: v for k, v in metrics_val.items() if k != "chart_data"}
        if metrics_val is not None:
            serialized = json.dumps(to_native_type(metrics_val), default=str)
            row["metrics_json"] = serialized
        elif row.get("metrics_json") is None:
            row["metrics_json"] = None

        # Dedicated chart_data column (JSON string), separate from metrics_json.
        chart_val = r.get("chart_data")
        if chart_val is None and isinstance(r.get("metrics"), dict):
            apply_chart_data(r)
            chart_val = r.get("chart_data")
        if isinstance(chart_val, (dict, list)):
            row["chart_data"] = json.dumps(to_native_type(chart_val), default=str)
        elif chart_val is not None:
            row["chart_data"] = str(chart_val)
        else:
            row["chart_data"] = None

        # Round all numeric fields to 2 decimal places
        for field in _NUMERIC_FIELDS:
            if row.get(field) is not None:
                try:
                    row[field] = round(float(row[field]), 2)
                except Exception:
                    pass

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


