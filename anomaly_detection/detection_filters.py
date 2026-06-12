"""Row-level filters before running anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from anomaly_detection.red_buff_sensors import get_features_for_lift_type


def should_skip_detection_row(row: Dict[str, Any], lift_type: str) -> Optional[str]:
    """
    Return a skip reason when the row should not be scored (SHUT-IN / all-zero sensors).
    """
    status = str(row.get("well_status") or "").upper().strip()
    if status and status != "PRODUCING":
        return f"well_status={status}"

    fields = [f for f in get_features_for_lift_type(lift_type) if f in row]
    if not fields:
        return None

    nonzero = False
    for field in fields:
        val = row.get(field)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            if abs(float(val)) > 1e-9:
                nonzero = True
                break
        except (TypeError, ValueError):
            continue

    if not nonzero:
        return "all_lift_sensors_zero_or_null"
    return None
