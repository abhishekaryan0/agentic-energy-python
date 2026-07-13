"""South Texas Energy Partners pipeline configuration."""

from __future__ import annotations

import os

# Tenant UUID — same style as RED_BUFF / APEX (used on all Snowflake rows).
STEP_ORGANIZATION_ID = (
    os.getenv("STEP_ORGANIZATION_ID")
    or os.getenv("ORGANIZATION_ID")
    or "d270f89a-aada-4ff2-adb0-68b3e8087565"
)

# Human-readable label in legacy master JSON (remapped on Snowflake ingest).
LEGACY_ORGANIZATION_LABEL = "SOUTH_TEXAS_ENERGY_PARTNERS"

STEP_SNOWFLAKE_SCHEMA = os.getenv("STEP_SNOWFLAKE_SCHEMA", "SOUTH_TEXAS_SENSOR_DATA")
NUM_WELLS = 100
