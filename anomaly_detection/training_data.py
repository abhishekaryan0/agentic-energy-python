"""
Healthy training row selection for per-well anomaly model training.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.red_buff_sensors import get_features_for_lift_type

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = int(os.getenv("ANOMALY_MIN_TRAINING_ROWS", "50") or 50)
EXCLUDE_RECENT_PCT = float(os.getenv("ANOMALY_TRAIN_EXCLUDE_RECENT_PCT", "15") or 15)


def _parse_clean_well_ids() -> set:
    raw = (os.getenv("ANOMALY_CLEAN_WELL_IDS") or "").strip()
    if not raw:
        return set()
    return {w.strip() for w in raw.split(",") if w.strip()}


def filter_healthy_training_rows(
    df: pd.DataFrame,
    lift_type: str,
    *,
    well_id: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Keep PRODUCING rows with at least one non-zero sensor reading.

    Optionally drops the most recent EXCLUDE_RECENT_PCT of rows (pre-failure buffer)
    unless the well is listed in ANOMALY_CLEAN_WELL_IDS.
    """
    stats = {
        "input_rows": len(df),
        "producing_rows": 0,
        "nonzero_sensor_rows": 0,
        "after_recent_exclude": 0,
        "output_rows": 0,
    }
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame(), stats

    out = df.copy()
    clean_ids = _parse_clean_well_ids()
    is_clean_well = well_id is not None and str(well_id) in clean_ids

    if "well_status" in out.columns:
        status = out["well_status"].astype(str).str.upper().str.strip()
        out = out[status == "PRODUCING"]
    stats["producing_rows"] = len(out)

    profile_fields = [
        f for f in get_features_for_lift_type(lift_type) if f in out.columns
    ]
    if profile_fields:
        sensor_block = out[profile_fields]
        nonzero_mask = (sensor_block.fillna(0).abs() > 1e-9).any(axis=1)
        out = out[nonzero_mask]
    stats["nonzero_sensor_rows"] = len(out)

    if (
        not is_clean_well
        and EXCLUDE_RECENT_PCT > 0
        and "timestamp" in out.columns
        and len(out) > MIN_TRAINING_ROWS
    ):
        out = out.sort_values("timestamp")
        keep = max(int(len(out) * (1.0 - EXCLUDE_RECENT_PCT / 100.0)), MIN_TRAINING_ROWS)
        keep = min(keep, len(out))
        out = out.iloc[:keep]
    stats["after_recent_exclude"] = len(out)

    out = out.reset_index(drop=True)
    stats["output_rows"] = len(out)
    return out, stats
