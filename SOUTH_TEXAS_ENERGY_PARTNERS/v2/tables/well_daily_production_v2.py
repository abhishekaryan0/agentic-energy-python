"""WELL_DAILY_PRODUCTION_V2 — daily volumes using native STEP V2 field names."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

TABLE_NAME = "WELL_DAILY_PRODUCTION_V2"

# (column_name, snowflake_type)
COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("date", "DATE"),
    ("timestamp", "VARCHAR(32)"),
    ("lift_type", "VARCHAR(50)"),
    ("run_status", "VARCHAR(50)"),
    ("gross_oil_production_bbl", "FLOAT"),
    ("casinghead_gas_mcf", "FLOAT"),
    ("produced_water_volume_bbl", "FLOAT"),
    ("well_test_rate_bopd", "FLOAT"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


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
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "date": pd.to_datetime(day["date"]).date(),
                "timestamp": day.get("timestamp") or f"{day['date']} 00:00:00",
                "lift_type": lift_type,
                "run_status": day.get("run_status"),
                "gross_oil_production_bbl": day.get("gross_oil_production_bbl"),
                "casinghead_gas_mcf": day.get("casinghead_gas_mcf"),
                "produced_water_volume_bbl": day.get("produced_water_volume_bbl"),
                "well_test_rate_bopd": day.get("well_test_rate_bopd"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


class WellDailyProductionV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    history_to_dataframe = staticmethod(history_to_dataframe)
