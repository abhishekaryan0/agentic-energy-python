"""STEP V2 Snowflake pipeline configuration (mirrors v1 config)."""

from __future__ import annotations

import os
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parent
_STEP_DIR = _V2_DIR.parent
_PROJECT_ROOT = _STEP_DIR.parent

STEP_ORGANIZATION_ID = (
    os.getenv("STEP_ORGANIZATION_ID")
    or os.getenv("ORGANIZATION_ID")
    or "d270f89a-aada-4ff2-adb0-68b3e8087565"
)

LEGACY_ORGANIZATION_LABEL = "SOUTH_TEXAS_ENERGY_PARTNERS"

STEP_V2_SNOWFLAKE_SCHEMA = os.getenv(
    "STEP_V2_SNOWFLAKE_SCHEMA",
    os.getenv("STEP_SNOWFLAKE_SCHEMA", "SOUTH_TEXAS_SENSOR_DATA"),
)

NUM_WELLS = 100

TABLE_NAMES = (
    "WELL_DAILY_PRODUCTION_V2",
    "WELL_SENSOR_READINGS_V2",
    "SYNTH_ASSETS_V2",
    "SYNTH_PERSONNEL_V2",
    "SYNTH_ASSET_DAILY_HISTORY_V2",
    "SYNTH_INVOICES_V2",
    "SYNTH_REVENUE_V2",
    "SYNTH_PIPELINE_STATE_V2",
)
