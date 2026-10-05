"""RED_BUFF V2 Snowflake pipeline configuration."""

from __future__ import annotations

import os
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parent
_RED_BUFF_DIR = _V2_DIR.parent
_PROJECT_ROOT = _RED_BUFF_DIR.parent

RED_BUFF_ORGANIZATION_ID = (
    os.getenv("RED_BUFF_ORGANIZATION_ID")
    or os.getenv("ORGANIZATION_ID")
    or "3e0b5b18-bec1-488a-82c1-a47881c69058"
)

LEGACY_ORGANIZATION_LABEL = "RED_BLUFF_RESOURCES"

RED_BUFF_V2_SNOWFLAKE_SCHEMA = os.getenv(
    "RED_BUFF_V2_SNOWFLAKE_SCHEMA",
    os.getenv("SNOWFLAKE_SCHEMA", "SENSOR_DATA"),
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
