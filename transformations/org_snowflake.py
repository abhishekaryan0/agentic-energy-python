"""Per-organization Snowflake schema routing for multi-tenant exports."""

from __future__ import annotations

import os
from dataclasses import replace

from transformations.config import SnowflakeConnectionConfig

STEP_ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"
DEFAULT_SENSOR_SCHEMA = "SENSOR_DATA"
STEP_SENSOR_SCHEMA = "SOUTH_TEXAS_SENSOR_DATA"


def resolve_org_snowflake_schema(organization_id: str) -> str:
    """Return Snowflake schema for a tenant's WELL_* / SYNTH_* tables."""
    org = (organization_id or "").strip()
    step_id = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    if org == step_id:
        return (os.getenv("STEP_SNOWFLAKE_SCHEMA") or STEP_SENSOR_SCHEMA).strip()
    return (os.getenv("SNOWFLAKE_SCHEMA") or DEFAULT_SENSOR_SCHEMA).strip()


def snowflake_connection_for_org(
    base: SnowflakeConnectionConfig,
    organization_id: str,
) -> SnowflakeConnectionConfig:
    schema = resolve_org_snowflake_schema(organization_id)
    if schema == (base.schema or "").strip():
        return base
    return replace(base, schema=schema)
