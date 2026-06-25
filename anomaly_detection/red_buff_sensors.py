"""
Red Buff well sensor profiles for anomaly detection.

Canonical WELL_SENSOR_READINGS column names and healthy reference means are
aligned with RED_BUFF/column_maps.py and RED_BUFF/generate_digital_twin.py
SENSOR_NORMALS (midpoint of each healthy operating range).

Band checks use: reference_mean ± deviation_pct% (multiplicative when mean is
non-trivial; otherwise additive using half the Red Buff range span).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Lift labels as stored in Red Buff / Snowflake parquet
LIFT_TYPES = ("Pumping Unit", "ESP", "Gas Lift")

# Metadata columns — never treated as sensor features
IDENTITY_COLUMNS = frozenset({
    "organization_id",
    "well_id",
    "timestamp",
    "lift_type",
    "well_status",
})

# Aliases from legacy or UI naming
LIFT_TYPE_ALIASES = {
    "Rod Pump": "Pumping Unit",
    "rod_pump": "Pumping Unit",
    "pumping unit": "Pumping Unit",
    "esp": "ESP",
    "gas lift": "Gas Lift",
    "gas_lift": "Gas Lift",
}


@dataclass(frozen=True)
class SensorProfile:
    """One sensor: canonical parquet column + Red Buff twin field + healthy range."""

    canonical_field: str
    red_buff_field: str
    lo: float
    hi: float
    unit: str = ""

    @property
    def mean(self) -> float:
        return (self.lo + self.hi) / 2.0

    @property
    def span(self) -> float:
        return self.hi - self.lo

    @property
    def twin_field(self) -> str:
        """Digital-twin source column name (Red Buff or Apex)."""
        return self.red_buff_field


# (canonical_field, red_buff_field, lo, hi, unit)
_PUMPING_UNIT_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("strokes_per_minute", "strokes_per_min", 0.5, 12.0, "SPM"),
    ("pump_fillage", "pump_fillage_pct", 0.0, 100.0, "%"),
    ("casing_pressure", "casing_pressure_psi", 50.0, 5000.0, "psi"),
    ("tubing_pressure", "tubing_pressure_psi", 50.0, 150.0, "psi"),
    ("motor_current", "motor_current_amps", 10.0, 200.0, "A"),
    ("drive_frequency", "freq_hz", 57.0, 62.0, "Hz"),
    ("pump_intake_pressure", "pump_intake_pressure", 0.0, 5000.0, "psi"),
    ("motor_temp", "motor_temp_f", 100.0, 160.0, "°F"),
    ("surface_stroke_length", "gross_stroke_len", 20.0, 600.0, "in"),
    ("downhole_gross_stroke", "gross_stroke_len", 20.0, 600.0, "in"),
    ("downhole_net_stroke", "net_stroke_len", 20.0, 600.0, "in"),
)

_ESP_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("tubing_pressure", "avg_tubing_pressure", 200.0, 500.0, "psi"),
    ("casing_pressure", "avg_casing_pressure", 20.0, 200.0, "psi"),
    ("motor_temp", "motor_temp_f", 150.0, 250.0, "°F"),
    ("motor_current", "motor_current_amps", 10.0, 200.0, "A"),
    ("drive_frequency", "freq_hz", 0.0, 100.0, "Hz"),
    ("pump_intake_pressure", "pump_intake_pressure_psi", 0.0, 5000.0, "psi"),
    ("discharge_pressure", "pump_discharge_pressure_psi", 0.0, 5000.0, "psi"),
    ("vibration_x", "vibration_hz", 0.0, 1.25, "in/s"),
)

_GAS_LIFT_PROFILES: Tuple[Tuple[str, str, float, float, str], ...] = (
    ("casing_pressure", "avg_casing_pressure", 50.0, 5000.0, "psi"),
    ("injection_pressure", "casing_injection_pressure_psi", 800.0, 1200.0, "psi"),
    ("tubing_pressure", "tubing_pressure_psi", 50.0, 150.0, "psi"),
    ("injection_rate", "injection_rate_mscfd", 0.0, 50000.0, "mscf/d"),
    ("wellhead_temp", "wellhead_temp_f", 10.0, 400.0, "°F"),
    ("motor_temp", "motor_temp_f", 80.0, 130.0, "°F"),
)

_PROFILES_BY_LIFT: Dict[str, Tuple[SensorProfile, ...]] = {
    "Pumping Unit": tuple(
        SensorProfile(c, r, lo, hi, u) for c, r, lo, hi, u in _PUMPING_UNIT_PROFILES
    ),
    "ESP": tuple(SensorProfile(c, r, lo, hi, u) for c, r, lo, hi, u in _ESP_PROFILES),
    "Gas Lift": tuple(
        SensorProfile(c, r, lo, hi, u) for c, r, lo, hi, u in _GAS_LIFT_PROFILES
    ),
}


def normalize_lift_type(lift_type: Optional[str]) -> str:
    """Map lift_type labels to Red Buff canonical names."""
    if not lift_type:
        return ""
    key = str(lift_type).strip()
    if key in _PROFILES_BY_LIFT:
        return key
    return LIFT_TYPE_ALIASES.get(key, LIFT_TYPE_ALIASES.get(key.lower(), key))


def get_sensor_profiles(lift_type: str) -> List[SensorProfile]:
    """Sensor profiles for a lift type (canonical columns + Red Buff healthy ranges)."""
    canonical = normalize_lift_type(lift_type)
    return list(_PROFILES_BY_LIFT.get(canonical, ()))


def get_features_for_lift_type(lift_type: str) -> List[str]:
    """All canonical sensor columns for this lift type (full Red Buff set)."""
    return [p.canonical_field for p in get_sensor_profiles(lift_type)]


def profile_by_field(lift_type: str) -> Dict[str, SensorProfile]:
    return {p.canonical_field: p for p in get_sensor_profiles(lift_type)}


def is_sensor_column(column: str) -> bool:
    return column not in IDENTITY_COLUMNS


def select_training_features(
    df: pd.DataFrame,
    lift_type: str,
    *,
    min_non_null: int = 2,
) -> List[str]:
    """
    All lift-type sensors present in df with enough non-null readings.

    Uses the full Red Buff profile per lift (not a critical subset).
    """
    if df is None or df.empty:
        return []

    selected: List[str] = []
    seen: set = set()
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
    """
    Sensor fields to evaluate at detection time: full lift profile plus any
    extra columns the per-well model was trained on.
    """
    fields: List[str] = []
    seen: set = set()
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
    """Red Buff healthy midpoint per canonical sensor field."""
    return {p.canonical_field: p.mean for p in get_sensor_profiles(lift_type)}


def compute_band(
    mean: float,
    *,
    deviation_pct: float,
    span: float,
) -> Tuple[float, float]:
    """
    Band around mean using deviation_pct.

    Uses multiplicative band when |mean| is meaningful; otherwise additive
    using half the Red Buff healthy span.
    """
    fraction = max(float(deviation_pct), 0.0) / 100.0
    if abs(mean) > 1e-6:
        low = mean * (1.0 - fraction)
        high = mean * (1.0 + fraction)
    else:
        half = span / 2.0
        low = mean - half * fraction
        high = mean + half * fraction
    return min(low, high), max(low, high)


def get_sensor_bands(
    lift_type: str,
    deviation_pct: float = 30.0,
) -> Dict[str, Dict[str, float]]:
    """
    Expected bands per sensor: mean ± deviation_pct%.

    Returns dict[field] -> {mean, min, max, unit, deviation_pct, twin_field}.
    """
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
    """Lift type registry keyed by Red Buff / parquet lift_type values."""
    return {name: i + 1 for i, name in enumerate(LIFT_TYPES)}


def get_all_sensor_columns() -> List[str]:
    """Union of identity columns and all Red Buff sensor fields."""
    seen = set()
    for lift in LIFT_TYPES:
        for field in get_features_for_lift_type(lift):
            seen.add(field)
    base = ["well_id", "timestamp", "lift_type", "well_status"]
    return base + sorted(seen)


def features_fallback_dict() -> Dict[str, List[str]]:
    """Per-lift feature lists — full sensor set for train_models / detector."""
    out: Dict[str, List[str]] = {}
    for lift in LIFT_TYPES:
        out[lift] = get_features_for_lift_type(lift)
    out["Rod Pump"] = list(out["Pumping Unit"])
    return out


def all_sensors_by_lift() -> Dict[str, List[str]]:
    """Every Red Buff sensor per lift (alias for CRITICAL_SENSORS replacement)."""
    return features_fallback_dict()
