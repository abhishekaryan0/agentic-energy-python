#!/usr/bin/env python3
"""
Sensor configuration for anomaly detection — backed by Red Bluff sensor profiles.

Replaces the legacy sensor_ranges.parquet min/max rules with canonical column
mapping and healthy reference means from RED_BUFF (see red_buff_sensors.py).
"""

import logging
import os
from typing import Dict, List

from dotenv import load_dotenv

from anomaly_detection.red_buff_sensors import (
    compute_band,
    get_all_sensor_columns as _rb_all_columns,
    get_features_for_lift_type as _rb_features,
    get_lift_types as _rb_lift_types,
    get_sensor_bands,
    get_sensor_profiles,
    normalize_lift_type,
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
    raise RuntimeError("Snowflake sensor rules are disabled; using Red Bluff band profiles.")


def get_lift_types() -> Dict[str, int]:
    """Red Bluff lift types: Pumping Unit, ESP, Gas Lift (+ Rod Pump alias)."""
    if "lift_types" not in _config_cache:
        types = _rb_lift_types()
        types["Rod Pump"] = types["Pumping Unit"]
        _config_cache["lift_types"] = types
    return _config_cache["lift_types"]


def get_sensors_for_lift_type(lift_type_name: str) -> List[Dict]:
    """Sensor definitions derived from Red Bluff profiles."""
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
                "red_buff_field": profile.red_buff_field,
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
            "red_buff_field": info.get("red_buff_field"),
            "healthy_lo": info.get("healthy_lo"),
            "healthy_hi": info.get("healthy_hi"),
        }
        for field, info in bands.items()
    }


def get_form_fields(lift_type_name: str) -> List[str]:
    return [s["field_name"] for s in get_form_sensors(lift_type_name)]


def get_features_for_lift_type(lift_type_name: str) -> List[str]:
    """All canonical WELL_SENSOR_READINGS sensor columns for this lift type."""
    return _rb_features(lift_type_name)


def select_training_features(df, lift_type_name: str, *, min_non_null: int = 2) -> List[str]:
    """All lift-type sensors in df with enough non-null values for training."""
    from anomaly_detection.red_buff_sensors import select_training_features as _select

    return _select(df, lift_type_name, min_non_null=min_non_null)


def get_all_sensor_columns() -> List[str]:
    return _rb_all_columns()


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
            "red_buff_field": profile.red_buff_field,
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
    print("=== Lift Types (Red Bluff) ===")
    print(get_lift_types())

    for lift in ("Pumping Unit", "ESP", "Gas Lift"):
        print(f"\n=== {lift} — canonical columns & bands (±{default_deviation_pct()}%) ===")
        for field, band in get_sensor_ranges(lift).items():
            print(
                f"  {field}: mean={band['mean']:.2f} "
                f"band=[{band['min']:.2f}, {band['max']:.2f}] {band['unit']}"
            )
