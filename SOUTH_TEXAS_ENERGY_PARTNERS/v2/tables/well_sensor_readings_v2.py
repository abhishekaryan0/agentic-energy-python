"""WELL_SENSOR_READINGS_V2 — native STEP V2 well sensor fields (union across lifts)."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

TABLE_NAME = "WELL_SENSOR_READINGS_V2"

# Shared identity + production-adjacent fields, then lift-specific sensors.
COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("date", "DATE"),
    ("timestamp", "VARCHAR(32)"),
    ("lift_type", "VARCHAR(50)"),
    ("run_status", "VARCHAR(50)"),
    # ESP
    ("flowing_tubing_pressure_psi", "FLOAT"),
    ("annular_casing_pressure_psi", "FLOAT"),
    ("motor_winding_temp_f", "FLOAT"),
    ("motor_load_amps", "FLOAT"),
    ("vfd_output_frequency_hz", "FLOAT"),
    ("pump_intake_pressure_psi", "FLOAT"),
    ("pump_discharge_pressure_psi", "FLOAT"),
    ("lateral_vibration_hz", "FLOAT"),
    # Gas Lift
    ("lift_gas_injection_pressure_psi", "FLOAT"),
    ("surface_tubing_pressure_psi", "FLOAT"),
    ("lift_gas_rate_mscfd", "FLOAT"),
    ("wellhead_flowing_temp_f", "FLOAT"),
    ("production_choke_opening_pct", "FLOAT"),
    ("compressor_inlet_temp_f", "FLOAT"),
    ("casing_tubing_pressure_gap_psi", "FLOAT"),
    ("gas_oil_ratio_scf_bbl", "FLOAT"),
    # Pumping Unit
    ("polished_rod_stroke_length_in", "FLOAT"),
    ("effective_plunger_stroke_in", "FLOAT"),
    ("pumping_speed_spm", "FLOAT"),
    ("volumetric_efficiency_pct", "FLOAT"),
    ("wellhead_tubing_pressure_psi", "FLOAT"),
    ("surface_motor_amperage", "FLOAT"),
    ("motor_drive_frequency_hz", "FLOAT"),
    ("downhole_pump_intake_psi", "FLOAT"),
    ("gearbox_motor_temp_f", "FLOAT"),
    ("daily_runtime_hrs", "FLOAT"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]
_SENSOR_FIELDS = [c for c, _ in COLUMNS if c not in {
    "organization_id", "asset_id", "date", "timestamp", "lift_type", "run_status",
}]


def ddl(qualified: str) -> str:
    cols = ",\n            ".join(f"{name.upper()} {ctype}" for name, ctype in COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            ID INT AUTOINCREMENT PRIMARY KEY,
            {cols},
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID, DATE)
        )
        """


def history_to_dataframe(
    history: List[dict],
    *,
    asset_id: str,
    lift_type: Optional[str],
    organization_id: str,
) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for day in history:
        rec: dict[str, Any] = {c: None for c in COLUMN_NAMES}
        rec["organization_id"] = organization_id
        rec["asset_id"] = asset_id
        rec["date"] = pd.to_datetime(day["date"]).date()
        rec["timestamp"] = day.get("timestamp") or f"{day['date']} 00:00:00"
        rec["lift_type"] = lift_type
        rec["run_status"] = day.get("run_status")
        for field in _SENSOR_FIELDS:
            if field in day:
                rec[field] = day.get(field)
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


class WellSensorReadingsV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    history_to_dataframe = staticmethod(history_to_dataframe)
