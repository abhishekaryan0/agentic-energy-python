"""
Tenant-aware sensor lists for operation recommendations.

Red Buff / South Texas Energy Partners: full WELL_SENSOR_READINGS column set (config.LIFT_TYPE_SENSORS).
Apex Permian: subset aligned with Apex parquet (transformations/field_mappings.py).
"""

from __future__ import annotations

import os
from typing import Dict, List

RED_BUFF_ORGANIZATION_ID = "3e0b5b18-bec1-488a-82c1-a47881c69058"
APEX_PERMIAN_ORGANIZATION_ID = "147a92b9-f90a-473d-b443-e6e668c4fbb7"
STEP_ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"

# Apex twin parquet sensor columns (see APEX_PERMIAN/generate_digital_twin.py).
APEX_LIFT_TYPE_SENSORS: Dict[str, List[str]] = {
    "Rod Pump": [
        "strokes_per_minute",
        "pump_fillage",
        "tubing_pressure",
        "casing_pressure",
        "motor_current",
        "pump_intake_pressure",
        "surface_stroke_length",
        "downhole_net_stroke",
    ],
    "ESP": [
        "tubing_pressure",
        "casing_pressure",
        "motor_temp",
        "motor_current",
        "freq_hz",
        "pump_intake_pressure",
    ],
    "Gas Lift": [
        "injection_rate",
        "injection_pressure",
        "tubing_pressure",
        "casing_pressure",
    ],
}

APEX_COMMON_SENSORS: List[str] = ["tubing_pressure", "casing_pressure"]

LIFT_TYPE_ALIASES = {
    "Pumping Unit": "Rod Pump",
    "pumping unit": "Rod Pump",
    "rod_pump": "Rod Pump",
}


def normalize_lift_type(lift_type: str | None) -> str:
    if not lift_type:
        return ""
    key = str(lift_type).strip()
    return LIFT_TYPE_ALIASES.get(key, LIFT_TYPE_ALIASES.get(key.lower(), key))


def resolve_tenant_profile(organization_id: str | None = None) -> str:
    org = (organization_id or os.getenv("ORGANIZATION_ID") or "").strip()
    override = (os.getenv("OPERATIONS_TENANT_PROFILE") or os.getenv("ANOMALY_TENANT_PROFILE") or "").strip().lower()
    if override in ("apex_permian", "apex", "apex-permian"):
        return "apex_permian"
    if override in ("step", "south_texas", "south-texas", "step_org"):
        return "red_buff"
    if override in ("red_buff", "red-buff", "redbuff"):
        return "red_buff"
    step_env = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    if org == APEX_PERMIAN_ORGANIZATION_ID:
        return "apex_permian"
    if org == step_env:
        return "red_buff"
    return "red_buff"


def tenant_profile_label(organization_id: str | None = None) -> str:
    org = (organization_id or os.getenv("ORGANIZATION_ID") or "").strip()
    step_env = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    if resolve_tenant_profile(organization_id) == "apex_permian":
        return "Apex Permian"
    if org == step_env:
        return "South Texas Energy Partners"
    return "Red Buff"


def get_tenant_sensors_for_lift_type(lift_type: str, organization_id: str | None = None) -> List[str]:
    from operation_recommendations.config import COMMON_SENSORS, LIFT_TYPE_SENSORS

    canonical = normalize_lift_type(lift_type)
    if resolve_tenant_profile(organization_id) == "apex_permian":
        return list(APEX_LIFT_TYPE_SENSORS.get(canonical, APEX_COMMON_SENSORS))
    return list(LIFT_TYPE_SENSORS.get(canonical, COMMON_SENSORS))


def get_tenant_common_sensors(organization_id: str | None = None) -> List[str]:
    from operation_recommendations.config import COMMON_SENSORS

    if resolve_tenant_profile(organization_id) == "apex_permian":
        return list(APEX_COMMON_SENSORS)
    return list(COMMON_SENSORS)
