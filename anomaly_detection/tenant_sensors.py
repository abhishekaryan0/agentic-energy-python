"""
Resolve anomaly-detection sensor profiles per organization.

Default (Red Buff and unknown orgs): red_buff_sensors — unchanged behavior.
Apex Permian org: apex_permian_sensors — Apex parquet columns and bands.
South Texas Energy Partners: step_sensors — Eagle Ford bands on full canonical parquet.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

RED_BUFF_ORGANIZATION_ID = "3e0b5b18-bec1-488a-82c1-a47881c69058"
APEX_PERMIAN_ORGANIZATION_ID = "147a92b9-f90a-473d-b443-e6e668c4fbb7"
STEP_ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"


def tenant_profile_label(organization_id: Optional[str] = None) -> str:
    """Human-readable tenant name for logs and LLM prompts."""
    profile = resolve_tenant_profile(organization_id)
    if profile == "apex_permian":
        return "Apex Permian"
    if profile == "step":
        return "South Texas Energy Partners"
    return "Red Buff"


def resolve_tenant_profile(organization_id: Optional[str] = None) -> str:
    org = (organization_id or os.getenv("ORGANIZATION_ID") or "").strip()
    override = (os.getenv("ANOMALY_TENANT_PROFILE") or "").strip().lower()
    if override in ("apex_permian", "apex", "apex-permian"):
        return "apex_permian"
    if override in ("step", "south_texas", "south-texas", "step_org"):
        return "step"
    if override in ("red_buff", "red-buff", "redbuff"):
        return "red_buff"
    step_env = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    if org == APEX_PERMIAN_ORGANIZATION_ID:
        return "apex_permian"
    if org == step_env:
        return "step"
    return "red_buff"


def _backend():
    profile = resolve_tenant_profile()
    if profile == "apex_permian":
        from anomaly_detection import apex_permian_sensors as backend

        return backend
    if profile == "step":
        from anomaly_detection import step_sensors as backend

        return backend
    from anomaly_detection import red_buff_sensors as backend

    return backend


def normalize_lift_type(lift_type: Optional[str]) -> str:
    return _backend().normalize_lift_type(lift_type)


def get_sensor_profiles(lift_type: str):
    return _backend().get_sensor_profiles(lift_type)


def get_features_for_lift_type(lift_type: str) -> List[str]:
    return _backend().get_features_for_lift_type(lift_type)


def profile_by_field(lift_type: str) -> Dict[str, Any]:
    return _backend().profile_by_field(lift_type)


def select_training_features(df, lift_type: str, *, min_non_null: int = 2) -> List[str]:
    return _backend().select_training_features(df, lift_type, min_non_null=min_non_null)


def detection_fields_for_lift(
    lift_type: str,
    readings: Optional[Dict] = None,
    trained_features: Optional[Sequence[str]] = None,
) -> List[str]:
    return _backend().detection_fields_for_lift(lift_type, readings, trained_features)


def get_reference_means(lift_type: str) -> Dict[str, float]:
    return _backend().get_reference_means(lift_type)


def compute_band(mean: float, *, deviation_pct: float, span: float):
    from anomaly_detection.red_buff_sensors import compute_band as _compute_band

    return _compute_band(mean, deviation_pct=deviation_pct, span=span)


def get_sensor_bands(lift_type: str, deviation_pct: float = 30.0) -> Dict[str, Dict[str, float]]:
    return _backend().get_sensor_bands(lift_type, deviation_pct=deviation_pct)


def get_lift_types() -> Dict[str, int]:
    types = _backend().get_lift_types()
    profile = resolve_tenant_profile()
    if profile in ("red_buff", "step"):
        types = dict(types)
        if "Pumping Unit" in types:
            types["Rod Pump"] = types["Pumping Unit"]
    return types


def get_all_sensor_columns() -> List[str]:
    return _backend().get_all_sensor_columns()


def features_fallback_dict() -> Dict[str, List[str]]:
    return _backend().features_fallback_dict()


def all_sensors_by_lift() -> Dict[str, List[str]]:
    return _backend().all_sensors_by_lift()


def is_sensor_column(column: str) -> bool:
    from anomaly_detection.red_buff_sensors import is_sensor_column as _is_sensor_column

    return _is_sensor_column(column)
