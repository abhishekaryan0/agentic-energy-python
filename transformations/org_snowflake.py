"""Per-organization Snowflake schema routing for multi-tenant exports."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from transformations.config import SnowflakeConnectionConfig, normalize_data_version

STEP_ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"
DEFAULT_SENSOR_SCHEMA = "SENSOR_DATA"
STEP_SENSOR_SCHEMA = "SOUTH_TEXAS_SENSOR_DATA"


def resolve_org_snowflake_schema(
    organization_id: str,
    *,
    data_version: Optional[str] = None,
) -> str:
    """Return Snowflake schema for a tenant's WELL_* / SYNTH_* (or *_V2) tables."""
    org = (organization_id or "").strip()
    step_id = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    version = normalize_data_version(data_version)
    if org == step_id:
        if version == "v2":
            return (
                os.getenv("STEP_V2_SNOWFLAKE_SCHEMA")
                or os.getenv("STEP_SNOWFLAKE_SCHEMA")
                or STEP_SENSOR_SCHEMA
            ).strip()
        return (os.getenv("STEP_SNOWFLAKE_SCHEMA") or STEP_SENSOR_SCHEMA).strip()
    if version == "v2":
        return (
            os.getenv("RED_BUFF_V2_SNOWFLAKE_SCHEMA")
            or os.getenv("SNOWFLAKE_SCHEMA")
            or DEFAULT_SENSOR_SCHEMA
        ).strip()
    return (os.getenv("SNOWFLAKE_SCHEMA") or DEFAULT_SENSOR_SCHEMA).strip()


def snowflake_connection_for_org(
    base: SnowflakeConnectionConfig,
    organization_id: str,
    *,
    data_version: Optional[str] = None,
) -> SnowflakeConnectionConfig:
    schema = resolve_org_snowflake_schema(organization_id, data_version=data_version)
    if schema == (base.schema or "").strip():
        return base
    return replace(base, schema=schema)
