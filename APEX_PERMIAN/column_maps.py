"""Map Apex Permian digital-twin DataFrames to canonical Snowflake tables (scripts shape)."""

from __future__ import annotations

from typing import List

import pandas as pd

from transformations.columns import WELL_DAILY_PRODUCTION_COLUMNS, WELL_SENSOR_READINGS_COLUMNS

from APEX_PERMIAN.generate_digital_twin import (
    _build_well_sensor_readings_df,
    _uniform_production_df,
)

_DAILY_COLS = [c for c, _ in WELL_DAILY_PRODUCTION_COLUMNS]
_SENSOR_COLS = [c for c, _ in WELL_SENSOR_READINGS_COLUMNS]


def master_df_to_daily_production(
    master_df: pd.DataFrame,
    well_meta: List[dict],
    organization_id: str,
) -> pd.DataFrame:
    """Same production shape as generate_digital_twin upload_to_s3 (scripts)."""
    if master_df.empty:
        return pd.DataFrame(columns=_DAILY_COLS)

    lift_by_id = {w["asset_id"]: w.get("lift_type") for w in well_meta}
    rows = []
    for _, row in master_df.iterrows():
        oil = float(row.get("true_oil") or 0)
        gas = float(row.get("true_gas") or 0)
        water = float(row.get("true_water") or 0)
        boe = round(oil + gas / 6.0, 2) if (oil or gas) else 0.0
        notes = row.get("notes")
        rows.append(
            {
                "organization_id": organization_id,
                "well_id": row["asset_id"],
                "date": pd.to_datetime(row["date"]).date(),
                "oil_volume": round(oil, 2),
                "gas_volume": round(gas, 2),
                "water_volume": round(water, 2),
                "boe": boe,
                "lift_type": lift_by_id.get(row["asset_id"]),
                "status": row.get("status"),
                "notes": notes if notes else None,
                "test_oil": row.get("test_oil"),
                "test_water": row.get("test_water"),
                "test_gas": row.get("test_gas"),
                "facility_id": row.get("facility_id"),
            }
        )

    df = pd.DataFrame(rows)
    return _uniform_production_df(df)[_DAILY_COLS]


def master_df_to_sensor_readings(
    master_df: pd.DataFrame,
    well_meta: List[dict],
    organization_id: str,
) -> pd.DataFrame:
    """Same sensor shape as generate_digital_twin._build_well_sensor_readings_df."""
    if master_df.empty:
        return pd.DataFrame(columns=_SENSOR_COLS)
    df = _build_well_sensor_readings_df(master_df, well_meta, organization_id)
    if "freq_hz" in df.columns:
        df["drive_frequency"] = df["freq_hz"]
    for col in _SENSOR_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_SENSOR_COLS]
