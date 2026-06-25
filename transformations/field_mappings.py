"""
Per-organization Snowflake → S3 parquet field mappings (export direction).

Ingest mappings live in APEX_PERMIAN/column_maps.py and RED_BUFF/column_maps.py
(twin engine field names → canonical Snowflake columns). This module applies the
reverse normalization when exporting so each tenant's parquet matches the shape
produced by its digital-twin script, under the same S3 prefix:

  s3://{bucket}/{ORGANIZATION_ID}/sensor_data/...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import pandas as pd

RED_BUFF_ORGANIZATION_ID = "3e0b5b18-bec1-488a-82c1-a47881c69058"
APEX_PERMIAN_ORGANIZATION_ID = "147a92b9-f90a-473d-b443-e6e668c4fbb7"

# Apex twin engine → canonical sensor names (see APEX_PERMIAN/generate_digital_twin.py).
APEX_TWIN_TO_CANONICAL_SENSOR: Dict[str, str] = {
    "avg_tubing_pressure": "tubing_pressure",
    "avg_casing_pressure": "casing_pressure",
    "spm": "strokes_per_minute",
    "pump_fillage_pct": "pump_fillage",
    "avg_motor_amps": "motor_current",
    "motor_temp_f": "motor_temp",
    "freq_hz": "drive_frequency",
    "injection_rate_mcf": "injection_rate",
    "gross_stroke_len": "surface_stroke_length",
    "net_stroke_len": "downhole_net_stroke",
}

# Parquet columns written by APEX_PERMIAN.generate_digital_twin._build_well_sensor_readings_df
APEX_SENSOR_PARQUET_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "well_id",
    "timestamp",
    "lift_type",
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
    "well_status",
)

# Red Bluff lift labels in Snowflake / parquet (anomaly_detection aliases Rod Pump → Pumping Unit).
RED_BUFF_LIFT_TYPES: FrozenSet[str] = frozenset({"Pumping Unit", "ESP", "Gas Lift"})


@dataclass(frozen=True)
class ExportFieldProfile:
    """How to shape parquet for one tenant on export."""

    name: str
    sensor_parquet_columns: Optional[Tuple[str, ...]] = None
    sensor_coalesce: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    drop_sensor_columns: Tuple[str, ...] = ()
    lift_type_values: Optional[FrozenSet[str]] = None


def _profile_from_env(org_id: str) -> Optional[str]:
    """Optional override: TRANSFORM_ORG_PROFILE_{UUID_NO_DASHES}=apex_permian."""
    key = f"TRANSFORM_ORG_PROFILE_{org_id.replace('-', '').upper()}"
    return (os.getenv(key) or "").strip().lower() or None


_EXPORT_PROFILES: Dict[str, ExportFieldProfile] = {
    "apex_permian": ExportFieldProfile(
        name="apex_permian",
        sensor_parquet_columns=APEX_SENSOR_PARQUET_COLUMNS,
        sensor_coalesce=(
            ("freq_hz", ("drive_frequency", "freq_hz")),
        ),
        drop_sensor_columns=("drive_frequency",),
        lift_type_values=frozenset({"Rod Pump", "ESP", "Gas Lift"}),
    ),
    "red_buff": ExportFieldProfile(
        name="red_buff",
        sensor_parquet_columns=None,
        sensor_coalesce=(
            ("drive_frequency", ("drive_frequency", "freq_hz")),
            ("freq_hz", ("freq_hz", "drive_frequency")),
        ),
        drop_sensor_columns=(),
        lift_type_values=RED_BUFF_LIFT_TYPES,
    ),
}

_ORG_ID_TO_PROFILE: Dict[str, str] = {
    APEX_PERMIAN_ORGANIZATION_ID: "apex_permian",
    RED_BUFF_ORGANIZATION_ID: "red_buff",
}


def resolve_export_profile(organization_id: Optional[str]) -> Optional[ExportFieldProfile]:
    """Return export field profile for a tenant, or None for generic canonical export."""
    org = (organization_id or "").strip()
    if not org:
        return None
    override = _profile_from_env(org)
    profile_key = override or _ORG_ID_TO_PROFILE.get(org)
    if not profile_key:
        apex_env = (os.getenv("APEX_PERMIAN_ORGANIZATION_ID") or "").strip()
        red_env = (os.getenv("RED_BUFF_ORGANIZATION_ID") or "").strip()
        if org == apex_env:
            profile_key = "apex_permian"
        elif org == red_env:
            profile_key = "red_buff"
    if not profile_key:
        return None
    return _EXPORT_PROFILES.get(profile_key)


def _coalesce_column(df: pd.DataFrame, target: str, sources: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if target not in out.columns:
        out[target] = None
    series = out[target]
    for source in sources:
        if source == target or source not in out.columns:
            continue
        series = series.fillna(out[source])
    out[target] = series
    return out


def apply_export_field_mapping(
    df: pd.DataFrame,
    organization_id: Optional[str],
    source_table: str,
) -> pd.DataFrame:
    """
    Apply tenant-specific column coalesce / projection after canonical coercion.

    Apex: Snowflake stores drive_frequency; twin script parquet uses freq_hz.
    Red Bluff: keep full canonical sensor schema with both drive_frequency and freq_hz.
    """
    if df is None or df.empty:
        return df

    profile = resolve_export_profile(organization_id)
    if profile is None:
        return df

    out = df.copy()

    if source_table == "WELL_SENSOR_READINGS":
        for target, sources in profile.sensor_coalesce:
            out = _coalesce_column(out, target, sources)

        # Gas lift: mirror twin fallback when injection_pressure missing
        if "injection_pressure" in out.columns and "casing_pressure" in out.columns:
            mask = (
                out["lift_type"].astype(str).eq("Gas Lift")
                & out["injection_pressure"].isna()
                & out["casing_pressure"].notna()
            )
            if mask.any():
                out.loc[mask, "injection_pressure"] = out.loc[mask, "casing_pressure"]

        for col in profile.drop_sensor_columns:
            if col in out.columns:
                out = out.drop(columns=[col])

        if profile.sensor_parquet_columns:
            out = out.reindex(columns=list(profile.sensor_parquet_columns))

    return out


def resolve_parquet_columns(
    organization_id: Optional[str],
    source_table: str,
    default_columns: Sequence[str],
) -> Tuple[str, ...]:
    """Final parquet column order for empty files and schema checks."""
    profile = resolve_export_profile(organization_id)
    if (
        profile
        and source_table == "WELL_SENSOR_READINGS"
        and profile.sensor_parquet_columns
    ):
        return profile.sensor_parquet_columns
    return tuple(default_columns)
