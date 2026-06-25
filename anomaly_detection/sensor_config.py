#!/usr/bin/env python3
"""
Sensor configuration for anomaly detection — tenant-aware band profiles.

Red Buff: full sensor set (red_buff_sensors).
Apex Permian: Apex parquet columns and bands (apex_permian_sensors).
"""

import logging
import os
from typing import Dict, List

from dotenv import load_dotenv

from anomaly_detection.tenant_sensors import (
    compute_band,
    get_all_sensor_columns as _tenant_all_columns,
    get_features_for_lift_type as _tenant_features,
    get_lift_types as _tenant_lift_types,
    get_sensor_bands,
    get_sensor_profiles,
    normalize_lift_type,
    resolve_tenant_profile,
    select_training_features as _tenant_select_training_features,
)

load_dotenv()

logger = logging.getLogger(__name__)

_config_cache: dict = {}


def default_deviation_pct() -> float:
    raw = os.getenv("ANOMALY_RULE_DEVIATION_PCT", "30").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 30.0


def get_db_connection():
    """Legacy stub — Snowflake rules are not used."""
    raise RuntimeError("Snowflake sensor rules are disabled; using tenant band profiles.")


def get_lift_types() -> Dict[str, int]:
    """Lift types for the active tenant profile."""
    if "lift_types" not in _config_cache:
        _config_cache["lift_types"] = _tenant_lift_types()
    return _config_cache["lift_types"]


def get_sensors_for_lift_type(lift_type_name: str) -> List[Dict]:
    """Sensor definitions derived from the active tenant profile."""
    cache_key = f"sensors_{lift_type_name}"
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    deviation = default_deviation_pct()
    sensors: List[Dict] = []
    for idx, profile in enumerate(get_sensor_profiles(lift_type_name)):
        band_min, band_max = compute_band(
            profile.mean, deviation_pct=deviation, span=profile.span
        )
        sensors.append(
            {
                "id": idx,
                "name": profile.canonical_field,
                "field_name": profile.canonical_field,
                "twin_field": profile.twin_field,
                "red_buff_field": profile.twin_field,
                "unit": profile.unit,
                "mean": profile.mean,
                "min": band_min,
                "max": band_max,
                "healthy_lo": profile.lo,
                "healthy_hi": profile.hi,
                "deviation_pct": deviation,
                "is_form_field": True,
                "display_order": idx,
            }
        )

    _config_cache[cache_key] = sensors
    return sensors


def get_form_sensors(lift_type_name: str) -> List[Dict]:
    return get_sensors_for_lift_type(lift_type_name)


def get_sensor_ranges(lift_type_name: str) -> Dict[str, Dict]:
    """
    Expected bands per sensor (mean ± deviation_pct%), not static min/max parquet.

    Returned shape is compatible with LLM prompts: mean, min, max, unit.
    """
    deviation = default_deviation_pct()
    bands = get_sensor_bands(lift_type_name, deviation_pct=deviation)
    return {
        field: {
            "mean": info["mean"],
            "min": info["min"],
            "max": info["max"],
            "unit": info.get("unit") or "",
            "deviation_pct": deviation,
            "twin_field": info.get("twin_field") or info.get("red_buff_field"),
            "red_buff_field": info.get("twin_field") or info.get("red_buff_field"),
            "healthy_lo": info.get("healthy_lo"),
            "healthy_hi": info.get("healthy_hi"),
        }
        for field, info in bands.items()
    }


def get_form_fields(lift_type_name: str) -> List[str]:
    return [s["field_name"] for s in get_form_sensors(lift_type_name)]


def get_features_for_lift_type(lift_type_name: str) -> List[str]:
    """All sensor columns for this lift type (tenant profile)."""
    return _tenant_features(lift_type_name)


def select_training_features(df, lift_type_name: str, *, min_non_null: int = 2) -> List[str]:
    """All lift-type sensors in df with enough non-null values for training."""
    return _tenant_select_training_features(df, lift_type_name, min_non_null=min_non_null)


def get_all_sensor_columns() -> List[str]:
    return _tenant_all_columns()


def get_anomaly_rules(lift_type_name: str) -> Dict:
    """Band profiles used for detection (no separate Snowflake rule table)."""
    cache_key = f"rules_{lift_type_name}"
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    deviation = default_deviation_pct()
    rules = {}
    for profile in get_sensor_profiles(lift_type_name):
        band_min, band_max = compute_band(
            profile.mean, deviation_pct=deviation, span=profile.span
        )
        rules[profile.canonical_field] = {
            "mean": profile.mean,
            "min": band_min,
            "max": band_max,
            "unit": profile.unit,
            "span": profile.span,
            "deviation_pct": deviation,
            "twin_field": profile.twin_field,
            "red_buff_field": profile.twin_field,
            "healthy_lo": profile.lo,
            "healthy_hi": profile.hi,
        }
    _config_cache[cache_key] = rules
    return rules


def clear_cache() -> None:
    global _config_cache
    _config_cache = {}


def get_all_sensor_info() -> Dict:
    lift_types = get_lift_types()
    return {
        lift_type_name: {
            "sensors": get_sensors_for_lift_type(lift_type_name),
            "form_fields": get_form_fields(lift_type_name),
            "ranges": get_sensor_ranges(lift_type_name),
            "rules": get_anomaly_rules(lift_type_name),
        }
        for lift_type_name in lift_types.keys()
        if lift_type_name != "Rod Pump"
    }


if __name__ == "__main__":
    print(f"=== Tenant profile: {resolve_tenant_profile()} ===")
    print("=== Lift Types ===")
    print(get_lift_types())

    for lift in get_lift_types():
        if lift == "Rod Pump" and resolve_tenant_profile() == "red_buff":
            continue
        print(f"\n=== {lift} — columns & bands (±{default_deviation_pct()}%) ===")
        for field, band in get_sensor_ranges(lift).items():
            print(
                f"  {field}: mean={band['mean']:.2f} "
                f"band=[{band['min']:.2f}, {band['max']:.2f}] {band['unit']}"
            )
