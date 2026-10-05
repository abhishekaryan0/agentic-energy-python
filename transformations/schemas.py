"""Canonical table specs for Snowflake exports consumed by ML pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from transformations.columns import WELL_DAILY_PRODUCTION_COLUMNS, WELL_SENSOR_READINGS_COLUMNS
from transformations.config import (
    RED_BUFF_ORGANIZATION_ID,
    STEP_ORGANIZATION_ID,
    normalize_data_version,
)


def _sensor_parquet_columns() -> Tuple[str, ...]:
    """
    Canonical sensor columns plus script-compatible alias.

    APEX_PERMIAN/generate_digital_twin.py writes `freq_hz`; Snowflake stores the same
    signal as DRIVE_FREQUENCY (see RED_BUFF/column_maps). We select both so
    parquet matches twin script field names without breaking consumers of
    `drive_frequency`.
    """
    base = tuple(column for column, _dtype in WELL_SENSOR_READINGS_COLUMNS)
    if "freq_hz" in base:
        return base
    out: List[str] = []
    inserted = False
    for column in base:
        out.append(column)
        if column == "drive_frequency" and not inserted:
            out.append("freq_hz")
            inserted = True
    return tuple(out)


@dataclass(frozen=True)
class TableExportSpec:
    """Snowflake source table and destination parquet shape."""

    source_table: str
    output_filename: str
    columns: Tuple[str, ...]
    order_by: Tuple[str, ...] = ()
    organization_column: str = "organization_id"
    partition_subdir: str = ""
    partition_column: str = ""  # date | timestamp_date
    watermark_column: str = ""  # date | timestamp
    dedupe_keys: Tuple[str, ...] = ()


WELL_SENSOR_READINGS_SPEC = TableExportSpec(
    source_table="WELL_SENSOR_READINGS",
    output_filename="well_sensor_readings.parquet",
    columns=_sensor_parquet_columns(),
    order_by=("timestamp", "well_id"),
    partition_subdir="well_sensor_readings",
    partition_column="timestamp_date",
    watermark_column="timestamp",
    dedupe_keys=("well_id", "timestamp"),
)

WELL_DAILY_PRODUCTION_SPEC = TableExportSpec(
    source_table="WELL_DAILY_PRODUCTION",
    output_filename="daily_production.parquet",
    columns=tuple(column for column, _dtype in WELL_DAILY_PRODUCTION_COLUMNS),
    order_by=("date", "well_id"),
    partition_subdir="daily_production",
    partition_column="date",
    watermark_column="date",
    dedupe_keys=("well_id", "date"),
)

EXPORT_SPECS: Dict[str, TableExportSpec] = {
    "well_sensor_readings": WELL_SENSOR_READINGS_SPEC,
    "sensor_readings": WELL_SENSOR_READINGS_SPEC,
    "sensors": WELL_SENSOR_READINGS_SPEC,
    "well_daily_production": WELL_DAILY_PRODUCTION_SPEC,
    "daily_production": WELL_DAILY_PRODUCTION_SPEC,
    "production": WELL_DAILY_PRODUCTION_SPEC,
}

# --- V2 native columns (STEP vs Red Buff differ; same parquet filenames/subdirs) ---

STEP_V2_DAILY_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "date",
    "timestamp",
    "lift_type",
    "run_status",
    "gross_oil_production_bbl",
    "casinghead_gas_mcf",
    "produced_water_volume_bbl",
    "well_test_rate_bopd",
)

RED_BUFF_V2_DAILY_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "date",
    "timestamp",
    "lift_type",
    "run_status",
    "allocated_oil_bbl",
    "allocated_gas_mcf",
    "allocated_water_bbl",
    "test_oil_bopd",
)

STEP_V2_SENSOR_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "date",
    "timestamp",
    "lift_type",
    "run_status",
    "flowing_tubing_pressure_psi",
    "annular_casing_pressure_psi",
    "motor_winding_temp_f",
    "motor_load_amps",
    "vfd_output_frequency_hz",
    "pump_intake_pressure_psi",
    "pump_discharge_pressure_psi",
    "lateral_vibration_hz",
    "lift_gas_injection_pressure_psi",
    "surface_tubing_pressure_psi",
    "lift_gas_rate_mscfd",
    "wellhead_flowing_temp_f",
    "production_choke_opening_pct",
    "compressor_inlet_temp_f",
    "casing_tubing_pressure_gap_psi",
    "gas_oil_ratio_scf_bbl",
    "polished_rod_stroke_length_in",
    "effective_plunger_stroke_in",
    "pumping_speed_spm",
    "volumetric_efficiency_pct",
    "wellhead_tubing_pressure_psi",
    "surface_motor_amperage",
    "motor_drive_frequency_hz",
    "downhole_pump_intake_psi",
    "gearbox_motor_temp_f",
    "daily_runtime_hrs",
)

RED_BUFF_V2_SENSOR_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "date",
    "timestamp",
    "lift_type",
    "run_status",
    "gross_stroke_length_in",
    "net_stroke_length_in",
    "strokes_per_min",
    "pump_fillage_pct",
    "pump_fillage_trend_7day",
    "casing_pressure_psi",
    "tubing_pressure_psi",
    "motor_current_amps",
    "freq_hz",
    "pump_intake_pressure_psi",
    "motor_temp_f",
    "daily_runtime_hrs",
    "rod_load_ratio",
    "casing_annulus_gas_rate_mscfd",
    "counterbalance_efficiency_pct",
    "avg_tubing_pressure_psi",
    "avg_casing_pressure_psi",
    "intake_temp_f",
    "pump_discharge_pressure_psi",
    "vibration_hz",
    "casing_injection_pressure_psi",
    "injection_rate_mscfd",
    "wellhead_temp_f",
    "choke_position_pct",
    "casing_tubing_gap_psi",
    "gor_scf_bbl",
)


def _v2_org_profile(organization_id: Optional[str]) -> str:
    org = (organization_id or "").strip()
    step_id = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    red_id = (os.getenv("RED_BUFF_ORGANIZATION_ID") or RED_BUFF_ORGANIZATION_ID).strip()
    if org == step_id:
        return "step"
    if org == red_id:
        return "red_buff"
    # Default to Red Buff native columns if unknown (safer than Apex).
    return "red_buff"


def _v2_specs_for_org(organization_id: Optional[str]) -> Dict[str, TableExportSpec]:
    profile = _v2_org_profile(organization_id)
    if profile == "step":
        daily_cols = STEP_V2_DAILY_COLUMNS
        sensor_cols = STEP_V2_SENSOR_COLUMNS
    else:
        daily_cols = RED_BUFF_V2_DAILY_COLUMNS
        sensor_cols = RED_BUFF_V2_SENSOR_COLUMNS

    sensor = TableExportSpec(
        source_table="WELL_SENSOR_READINGS_V2",
        output_filename="well_sensor_readings.parquet",
        columns=sensor_cols,
        order_by=("date", "asset_id"),
        partition_subdir="well_sensor_readings",
        partition_column="date",
        watermark_column="date",
        dedupe_keys=("asset_id", "date"),
    )
    daily = TableExportSpec(
        source_table="WELL_DAILY_PRODUCTION_V2",
        output_filename="daily_production.parquet",
        columns=daily_cols,
        order_by=("date", "asset_id"),
        partition_subdir="daily_production",
        partition_column="date",
        watermark_column="date",
        dedupe_keys=("asset_id", "date"),
    )
    return {
        "well_sensor_readings": sensor,
        "sensor_readings": sensor,
        "sensors": sensor,
        "well_daily_production": daily,
        "daily_production": daily,
        "production": daily,
    }


def get_export_specs(
    table_names: Iterable[str],
    *,
    data_version: str = "v1",
    organization_id: Optional[str] = None,
) -> List[TableExportSpec]:
    """Resolve requested table names or aliases to de-duplicated export specs."""
    version = normalize_data_version(data_version)
    catalog = (
        _v2_specs_for_org(organization_id) if version == "v2" else EXPORT_SPECS
    )
    specs: List[TableExportSpec] = []
    seen = set()
    unknown = []

    for raw_name in table_names:
        name = (raw_name or "").strip().lower()
        if not name:
            continue
        spec = catalog.get(name)
        if spec is None:
            unknown.append(raw_name)
            continue
        if spec.source_table not in seen:
            specs.append(spec)
            seen.add(spec.source_table)

    if unknown:
        known = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown transform table(s): {unknown}. Known values: {known}")
    if not specs:
        raise ValueError("No transform tables selected")
    return specs
