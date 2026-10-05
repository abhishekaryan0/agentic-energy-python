"""SYNTH_ASSET_DAILY_HISTORY_V2 — full daily history row as JSON (all asset types)."""

from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

TABLE_NAME = "SYNTH_ASSET_DAILY_HISTORY_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("date", "DATE"),
    ("timestamp", "VARCHAR(32)"),
    ("asset_type", "VARCHAR(50)"),
    ("lift_type", "VARCHAR(50)"),
    ("run_status", "VARCHAR(50)"),
    ("history_json", "VARCHAR"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


def ddl(qualified: str) -> str:
    cols = ",\n            ".join(f"{name.upper()} {ctype}" for name, ctype in COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            {cols},
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID, DATE)
        )
        """


def history_to_dataframe(
    history: List[dict],
    *,
    asset_id: str,
    asset_type: Optional[str],
    lift_type: Optional[str],
    organization_id: str,
) -> pd.DataFrame:
    rows = []
    for day in history:
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "date": pd.to_datetime(day["date"]).date(),
                "timestamp": day.get("timestamp") or f"{day['date']} 00:00:00",
                "asset_type": asset_type,
                "lift_type": lift_type,
                "run_status": day.get("run_status"),
                "history_json": json.dumps(day, default=str),
            }
        )
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


class SynthAssetDailyHistoryV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    history_to_dataframe = staticmethod(history_to_dataframe)
