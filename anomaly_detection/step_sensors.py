"""
South Texas Energy Partners (STEP) sensor profiles for anomaly detection.

Healthy operating ranges are aligned with SOUTH_TEXAS_ENERGY_PARTNERS/generate_digital_twin.py
SENSOR_NORMALS and SOUTH_TEXAS_ENERGY_PARTNERS/column_maps.py canonical parquet columns.
Eagle Ford wells use higher ESP / Gas Lift pressures than Red Bluff.
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

LIFT_TYPES = ("Pumping Unit", "ESP", "Gas Lift")

LIFT_TYPE_ALIASES = {
    "Rod Pump": "Pumping Unit",
    "rod_pump": "Pumping Unit",
    "pumping unit": "Pumping Unit",
    "esp": "ESP",
    "gas lift": "Gas Lift",
    "gas_lift": "Gas Lift",
}

# (canonical_field, twin_source_field, lo, hi, unit)
_PUMPING_UNIT_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("strokes_per_minute", "pumping_speed_spm", 0.5, 12.0, "SPM"),
    ("pump_fillage", "volumetric_efficiency_pct", 0.0, 100.0, "%"),
    ("casing_pressure", "annular_casing_pressure_psi", 50.0, 3000.0, "psi"),
    ("tubing_pressure", "wellhead_tubing_pressure_psi", 50.0, 200.0, "psi"),
    ("motor_current", "surface_motor_amperage", 10.0, 150.0, "A"),
    ("drive_frequency", "motor_drive_frequency_hz", 57.0, 62.0, "Hz"),
    ("pump_intake_pressure", "downhole_pump_intake_psi", 0.0, 3000.0, "psi"),
    ("motor_temp", "gearbox_motor_temp_f", 100.0, 160.0, "°F"),
    ("surface_stroke_length", "polished_rod_stroke_length_in", 20.0, 600.0, "in"),
    ("downhole_gross_stroke", "polished_rod_stroke_length_in", 20.0, 600.0, "in"),
    ("downhole_net_stroke", "effective_plunger_stroke_in", 20.0, 600.0, "in"),
)

_ESP_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("tubing_pressure", "flowing_tubing_pressure_psi", 300.0, 800.0, "psi"),
    ("casing_pressure", "annular_casing_pressure_psi", 50.0, 400.0, "psi"),
    ("motor_temp", "motor_winding_temp_f", 180.0, 280.0, "°F"),
    ("motor_current", "motor_load_amps", 30.0, 180.0, "A"),
    ("drive_frequency", "vfd_output_frequency_hz", 45.0, 65.0, "Hz"),
    ("pump_intake_pressure", "pump_intake_pressure_psi", 500.0, 4000.0, "psi"),
    ("discharge_pressure", "pump_discharge_pressure_psi", 1500.0, 6000.0, "psi"),
    ("vibration_x", "lateral_vibration_hz", 0.0, 1.25, "in/s"),
)

_GAS_LIFT_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("casing_pressure", "annular_casing_pressure_psi", 800.0, 2000.0, "psi"),
    ("injection_pressure", "lift_gas_injection_pressure_psi", 1200.0, 2200.0, "psi"),
    ("tubing_pressure", "surface_tubing_pressure_psi", 100.0, 400.0, "psi"),
    ("injection_rate", "lift_gas_rate_mscfd", 500.0, 8000.0, "mscf/d"),
    ("wellhead_temp", "wellhead_flowing_temp_f", 80.0, 220.0, "°F"),
    ("motor_temp", "compressor_inlet_temp_f", 80.0, 130.0, "°F"),
)

_PROFILES_BY_LIFT: Dict[str, Tuple[SensorProfile, ...]] = {
    "Pumping Unit": tuple(
        SensorProfile(c, twin, lo, hi, u) for c, twin, lo, hi, u in _PUMPING_UNIT_PROFILES
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
    out["Rod Pump"] = list(out["Pumping Unit"])
    return out


def all_sensors_by_lift() -> Dict[str, List[str]]:
    return features_fallback_dict()
