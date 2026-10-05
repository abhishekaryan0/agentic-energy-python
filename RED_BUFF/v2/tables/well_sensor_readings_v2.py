"""WELL_SENSOR_READINGS_V2 — native RBNR V2 sensor fields (union across lifts)."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

TABLE_NAME = "WELL_SENSOR_READINGS_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("date", "DATE"),
    ("timestamp", "VARCHAR(32)"),
    ("lift_type", "VARCHAR(50)"),
    ("run_status", "VARCHAR(50)"),
    # Pumping Unit
    ("gross_stroke_length_in", "FLOAT"),
    ("net_stroke_length_in", "FLOAT"),
    ("strokes_per_min", "FLOAT"),
    ("pump_fillage_pct", "FLOAT"),
    ("pump_fillage_trend_7day", "FLOAT"),
    ("casing_pressure_psi", "FLOAT"),
    ("tubing_pressure_psi", "FLOAT"),
    ("motor_current_amps", "FLOAT"),
    ("freq_hz", "FLOAT"),
    ("pump_intake_pressure_psi", "FLOAT"),
    ("motor_temp_f", "FLOAT"),
    ("daily_runtime_hrs", "FLOAT"),
    ("rod_load_ratio", "FLOAT"),
    ("casing_annulus_gas_rate_mscfd", "FLOAT"),
    ("counterbalance_efficiency_pct", "FLOAT"),
    # ESP
    ("avg_tubing_pressure_psi", "FLOAT"),
    ("avg_casing_pressure_psi", "FLOAT"),
    ("intake_temp_f", "FLOAT"),
    ("pump_discharge_pressure_psi", "FLOAT"),
    ("vibration_hz", "FLOAT"),
    # Gas Lift
    ("casing_injection_pressure_psi", "FLOAT"),
    ("injection_rate_mscfd", "FLOAT"),
    ("wellhead_temp_f", "FLOAT"),
    ("choke_position_pct", "FLOAT"),
    ("casing_tubing_gap_psi", "FLOAT"),
    ("gor_scf_bbl", "FLOAT"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]
_SENSOR_FIELDS = [
    c
    for c, _ in COLUMNS
    if c not in {"organization_id", "asset_id", "date", "timestamp", "lift_type", "run_status"}
]


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
