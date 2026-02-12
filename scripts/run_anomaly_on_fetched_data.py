#!/usr/bin/env python3
"""
Run anomaly review on REAL well data with the last few days modified in-memory
to inject a chosen anomaly. No DB writes — fetch, modify in memory, run detection.

Usage:
  python scripts/run_anomaly_on_fetched_data.py --well Well_002_RodPump --anomaly ENERGY_DIV
  python scripts/run_anomaly_on_fetched_data.py --well Well_001_RodPump --anomaly MODEL_DIVERGENCE --days 4

Then inspect the printed JSON (or set --save to write results to DB).

Why this works: we use REAL data (real production levels, real sensor ranges). We only
change the last 3–4 days so baselines (rolling 30d/90d) stay realistic and the checker
sees "current period bad, baseline normal" and fires. No DB changes.
"""

import os
import sys
import argparse
import json
from datetime import timedelta

# Project root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

# After path is set
import numpy as np
import pandas as pd

from anomaly_review.data import fetch_well_data
from anomaly_review.service import detect_anomalies
from anomaly_review.config import ANOMALY_REVIEW_FETCH_DAYS


# How many trailing days to modify (inject anomaly)
DEFAULT_ANOMALY_DAYS = 4

# Anomaly codes we can inject
ANOMALY_CODES = [
    "DATA_MISMATCH",      # Zero production, sensors active
    "MODEL_DIVERGENCE",   # Production drop vs model
    "ENERGY_DIV",         # High amps, low oil
    "FLUID_POUND",        # Low pump fillage (Rod Pump)
    "PRESSURE_INSTABILITY",  # High pressure variance
]


def _last_n_days_mask(df: pd.DataFrame, n_days: int) -> pd.Series:
    """Boolean mask True for rows in the last n calendar days of df (by timestamp)."""
    if df.empty or "timestamp" not in df.columns:
        return pd.Series(False, index=df.index)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    try:
        dates = ts.dt.tz_localize(None).dt.date
    except Exception:
        dates = ts.dt.date
    max_date = dates.max()
    min_date = max_date - timedelta(days=n_days - 1)
    return (dates >= min_date) & (dates <= max_date)


def inject_anomaly(df: pd.DataFrame, anomaly_code: str, n_days: int) -> pd.DataFrame:
    """
    Modify the last n_days of df in-place to mimic the chosen anomaly.
    Uses real baselines from the rest of the data so checkers fire reliably.
    """
    if df.empty or n_days < 1:
        return df
    mask = _last_n_days_mask(df, n_days)
    if not mask.any():
        return df

    # Precompute baselines from the *non*-modified period (so they stay realistic)
    non_modified = df.loc[~mask]
    oil_baseline = non_modified["oil_volume"].mean() if "oil_volume" in df.columns and not non_modified.empty else 200
    amps_baseline = non_modified["motor_current"].mean() if "motor_current" in df.columns and not non_modified.empty else 50
    fillage_baseline = non_modified["pump_fillage"].mean() if "pump_fillage" in df.columns and not non_modified.empty else 90
    tp_baseline = non_modified["tubing_pressure"].mean() if "tubing_pressure" in df.columns and not non_modified.empty else 1200

    if anomaly_code == "DATA_MISMATCH":
        # Zero production, sensors stay active (real values)
        for col in ["oil_volume", "gas_volume", "water_volume"]:
            if col in df.columns:
                df.loc[mask, col] = 0.0

    elif anomaly_code == "MODEL_DIVERGENCE":
        # Strong production drop so actual << predicted (model sees history, predicts higher)
        for col in ["oil_volume", "gas_volume", "water_volume"]:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] * 0.2

    elif anomaly_code == "ENERGY_DIV":
        # High amps + low oil (baselines from non-modified; we make current period bad)
        if "motor_current" in df.columns:
            df.loc[mask, "motor_current"] = df.loc[mask, "motor_current"] * 1.5
        if "oil_volume" in df.columns:
            df.loc[mask, "oil_volume"] = np.maximum(0, oil_baseline * 0.2)
        if "gas_volume" in df.columns:
            df.loc[mask, "gas_volume"] = df.loc[mask, "gas_volume"] * 0.2
        if "water_volume" in df.columns:
            df.loc[mask, "water_volume"] = df.loc[mask, "water_volume"] * 0.2

    elif anomaly_code == "FLUID_POUND":
        if "pump_fillage" in df.columns:
            df.loc[mask, "pump_fillage"] = 65.0

    elif anomaly_code == "PRESSURE_INSTABILITY":
        if "tubing_pressure" in df.columns:
            # Add variance: multiply by random factor around 1.0
            n = mask.sum()
            df.loc[mask, "tubing_pressure"] = df.loc[mask, "tubing_pressure"] * np.random.uniform(0.6, 1.4, n)
        if "casing_pressure" in df.columns:
            n = mask.sum()
            df.loc[mask, "casing_pressure"] = df.loc[mask, "casing_pressure"] * np.random.uniform(0.6, 1.4, n)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch real well data, inject an anomaly in the last N days in-memory, run anomaly review (no DB write by default)."
    )
    parser.add_argument("--well", "-w", required=True, help="Well ID (e.g. Well_002_RodPump)")
    parser.add_argument("--anomaly", "-a", required=True, choices=ANOMALY_CODES,
                        help="Anomaly type to inject in the last N days")
    parser.add_argument("--days", "-d", type=int, default=DEFAULT_ANOMALY_DAYS,
                        help=f"Number of trailing days to modify (default {DEFAULT_ANOMALY_DAYS})")
    parser.add_argument("--save", action="store_true", help="If set, save detection results to DB")
    parser.add_argument("--fetch-days", type=int, default=ANOMALY_REVIEW_FETCH_DAYS,
                        help=f"Days of history to fetch (default {ANOMALY_REVIEW_FETCH_DAYS})")
    args = parser.parse_args()

    well_id = args.well
    anomaly_code = args.anomaly
    n_days = max(1, min(args.days, 14))
    save_to_db = args.save

    print(f"Fetching up to {args.fetch_days} days of data for {well_id}...")
    df = fetch_well_data(well_id, days=args.fetch_days)
    if df is None or df.empty or len(df) < 50:
        print("ERROR: Not enough data. Need at least 50 rows.")
        return 1

    print(f"Injecting anomaly '{anomaly_code}' in the last {n_days} days (in-memory only)...")
    inject_anomaly(df, anomaly_code, n_days)

    print("Running anomaly review on modified data (save_to_db={})...".format(save_to_db))
    results = detect_anomalies(well_id, df=df, save_to_db=save_to_db)

    print(json.dumps(results, indent=2, default=str))
    if results:
        codes = [r.get("anomaly_code") for r in results]
        print("\nDetected:", codes)
        if anomaly_code in codes:
            print("SUCCESS: Requested anomaly was detected.")
        else:
            print("NOTE: Requested anomaly was not in the detected list. Check thresholds or try --days 5.")
    else:
        print("\nNo anomalies reported. Try different --days or check checker thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
