"""
Apex Permian sensor profiles for anomaly detection.

Healthy operating ranges are aligned with APEX_PERMIAN/generate_digital_twin.py
physics (Permian-scale pressures, SPM, injection, ESP frequency). Separate from
Do not import band ranges from red_buff_sensors (Red Buff is a separate tenant).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

from anomaly_detection.red_buff_sensors import (
    IDENTITY_COLUMNS,
    SensorProfile,
    compute_band,
    is_sensor_column,
)

# Canonical columns present in Apex S3 sensor parquet (excluding identity).
APEX_SENSOR_FIELDS = frozenset(
    {
        "tubing_pressure",
        "casing_pressure",
        "strokes_per_minute",
        "pump_fillage",
        "motor_current",
        "pump_intake_pressure",
        "motor_temp",
        "freq_hz",
        "injection_rate",
        "surface_stroke_length",
        "downhole_net_stroke",
        "injection_pressure",
    }
)

LIFT_TYPES = ("Rod Pump", "ESP", "Gas Lift")

LIFT_TYPE_ALIASES = {
    "Pumping Unit": "Rod Pump",
    "Rod Pump": "Rod Pump",
    "rod_pump": "Rod Pump",
    "pumping unit": "Rod Pump",
    "esp": "ESP",
    "gas lift": "Gas Lift",
    "gas_lift": "Gas Lift",
}

# (canonical_field, twin_source_field, lo, hi, unit) — from Apex digital twin KPIs
_ROD_PUMP_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("strokes_per_minute", "spm", 6.0, 11.0, "SPM"),
    ("pump_fillage", "pump_fillage_pct", 65.0, 100.0, "%"),
    ("casing_pressure", "avg_casing_pressure", 25.0, 180.0, "psi"),
    ("tubing_pressure", "avg_tubing_pressure", 180.0, 420.0, "psi"),
    ("motor_current", "avg_motor_amps", 10.0, 180.0, "A"),
    ("pump_intake_pressure", "pump_intake_pressure", 200.0, 850.0, "psi"),
    ("surface_stroke_length", "gross_stroke_len", 80.0, 160.0, "in"),
    ("downhole_net_stroke", "net_stroke_len", 70.0, 145.0, "in"),
)

_ESP_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("tubing_pressure", "avg_tubing_pressure", 180.0, 420.0, "psi"),
    ("casing_pressure", "avg_casing_pressure", 25.0, 180.0, "psi"),
    ("motor_temp", "motor_temp_f", 160.0, 320.0, "°F"),
    ("motor_current", "avg_motor_amps", 25.0, 260.0, "A"),
    ("freq_hz", "freq_hz", 52.0, 63.0, "Hz"),
    ("pump_intake_pressure", "pump_intake_pressure", 350.0, 950.0, "psi"),
)

_GAS_LIFT_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("casing_pressure", "avg_casing_pressure", 820.0, 980.0, "psi"),
    ("injection_pressure", "injection_pressure", 820.0, 980.0, "psi"),
    ("tubing_pressure", "avg_tubing_pressure", 220.0, 320.0, "psi"),
    ("injection_rate", "injection_rate_mcf", 480.0, 720.0, "mscf/d"),
)

_PROFILES_BY_LIFT: Dict[str, Tuple[SensorProfile, ...]] = {
    "Rod Pump": tuple(
        SensorProfile(c, twin, lo, hi, u) for c, twin, lo, hi, u in _ROD_PUMP_PROFILES
    ),
    "ESP": tuple(SensorProfile(c, twin, lo, hi, u) for c, twin, lo, hi, u in _ESP_PROFILES),
    "Gas Lift": tuple(
        SensorProfile(c, twin, lo, hi, u) for c, twin, lo, hi, u in _GAS_LIFT_PROFILES
    ),
}


def normalize_lift_type(lift_type: Optional[str]) -> str:
    if not lift_type:
        return ""
    key = str(lift_type).strip()
    if key in _PROFILES_BY_LIFT:
        return key
    return LIFT_TYPE_ALIASES.get(key, LIFT_TYPE_ALIASES.get(key.lower(), key))


def get_sensor_profiles(lift_type: str) -> List[SensorProfile]:
    canonical = normalize_lift_type(lift_type)
    return list(_PROFILES_BY_LIFT.get(canonical, ()))


def get_features_for_lift_type(lift_type: str) -> List[str]:
    return [p.canonical_field for p in get_sensor_profiles(lift_type)]


def profile_by_field(lift_type: str) -> Dict[str, SensorProfile]:
    return {p.canonical_field: p for p in get_sensor_profiles(lift_type)}


def select_training_features(
    df: pd.DataFrame,
    lift_type: str,
    *,
    min_non_null: int = 2,
) -> List[str]:
    if df is None or df.empty:
        return []
    selected: List[str] = []
    seen: set[str] = set()
    min_count = max(int(min_non_null), 1)
    for field in get_features_for_lift_type(lift_type):
        if field in seen or field not in df.columns:
            continue
        if not is_numeric_dtype(df[field]):
            continue
        if int(df[field].notna().sum()) >= min_count:
            selected.append(field)
            seen.add(field)
    return selected


def detection_fields_for_lift(
    lift_type: str,
    readings: Optional[Dict] = None,
    trained_features: Optional[Sequence[str]] = None,
) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for field in get_features_for_lift_type(lift_type):
        if field not in seen:
            fields.append(field)
            seen.add(field)
    if trained_features:
        for field in trained_features:
            if field not in seen and is_sensor_column(str(field)):
                fields.append(str(field))
                seen.add(str(field))
    if readings:
        for field in readings:
            if field in seen or field in IDENTITY_COLUMNS:
                continue
            if readings.get(field) is not None and is_sensor_column(str(field)):
                fields.append(str(field))
                seen.add(str(field))
    return fields


def get_reference_means(lift_type: str) -> Dict[str, float]:
    return {p.canonical_field: p.mean for p in get_sensor_profiles(lift_type)}


def get_sensor_bands(
    lift_type: str,
    deviation_pct: float = 30.0,
) -> Dict[str, Dict[str, float]]:
    bands: Dict[str, Dict[str, float]] = {}
    for profile in get_sensor_profiles(lift_type):
        band_min, band_max = compute_band(
            profile.mean,
            deviation_pct=deviation_pct,
            span=profile.span,
        )
        bands[profile.canonical_field] = {
            "mean": profile.mean,
            "min": band_min,
            "max": band_max,
            "unit": profile.unit,
            "deviation_pct": deviation_pct,
            "twin_field": profile.twin_field,
            "red_buff_field": profile.twin_field,
            "healthy_lo": profile.lo,
            "healthy_hi": profile.hi,
        }
    return bands


def get_lift_types() -> Dict[str, int]:
    return {name: i + 1 for i, name in enumerate(LIFT_TYPES)}


def get_all_sensor_columns() -> List[str]:
    seen = set()
    for lift in LIFT_TYPES:
        for field in get_features_for_lift_type(lift):
            seen.add(field)
    base = ["well_id", "timestamp", "lift_type", "well_status"]
    return base + sorted(seen)


def features_fallback_dict() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for lift in LIFT_TYPES:
        out[lift] = get_features_for_lift_type(lift)
    return out


def all_sensors_by_lift() -> Dict[str, List[str]]:
    return features_fallback_dict()
