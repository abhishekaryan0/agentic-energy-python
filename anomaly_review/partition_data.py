"""
Load partitioned sensor + production data from S3 for anomaly review batch runs.

Layout:
  s3://{bucket}/{org}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet
  s3://{bucket}/{org}/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def list_partition_keys_in_range(
    s3,
    *,
    bucket: str,
    org: str,
    subdir: str,
    start: date,
    end: date,
) -> List[str]:
    prefix = f"{org}/sensor_data/{subdir}/date="
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key") or ""
            if not key.endswith("/part-0000.parquet"):
                continue
            try:
                part = key.split("date=", 1)[1].split("/", 1)[0]
                d = pd.to_datetime(part).date()
            except Exception:
                continue
            if start <= d <= end:
                keys.append(key)
    keys.sort()
    return keys


def read_partition_df(s3, *, bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as exc:
        logger.debug("Skip partition %s: %s", key, exc)
        return None


def load_partitions(
    s3,
    *,
    bucket: str,
    org: str,
    subdir: str,
    start: date,
    end: date,
) -> Tuple[pd.DataFrame, List[str]]:
    keys = list_partition_keys_in_range(
        s3, bucket=bucket, org=org, subdir=subdir, start=start, end=end
    )
    frames: List[pd.DataFrame] = []
    for key in keys:
        df = read_partition_df(s3, bucket=bucket, key=key)
        if df is not None and not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, keys


def _well_column(df: pd.DataFrame) -> Optional[str]:
    if "well_id" in df.columns:
        return "well_id"
    if "asset_id" in df.columns:
        return "asset_id"
    return None


def wells_with_data_in_range(
    sensor_df: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> List[str]:
    if sensor_df.empty:
        return []
    well_col = _well_column(sensor_df)
    if not well_col:
        return []
    df = sensor_df.copy()
    ts_col = "timestamp" if "timestamp" in df.columns else "event_ts"
    if ts_col not in df.columns:
        ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    if not ts_col:
        return sorted(df[well_col].dropna().astype(str).unique().tolist())
    df["_ts"] = pd.to_datetime(df[ts_col], utc=True)
    mask = (df["_ts"].dt.date >= start) & (df["_ts"].dt.date <= end)
    return sorted(df.loc[mask, well_col].dropna().astype(str).unique().tolist())


def build_well_dataframe(
    sensor_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    well_id: str,
    *,
    fetch_days: int,
) -> pd.DataFrame:
    """
    Build per-well time series (sensor + daily production join) like data.fetch_well_data.
    """
    if sensor_df.empty:
        return pd.DataFrame()

    well_col = _well_column(sensor_df)
    if not well_col:
        return pd.DataFrame()

    sensor = sensor_df[sensor_df[well_col].astype(str) == str(well_id)].copy()
    if sensor.empty:
        return pd.DataFrame()

    ts_col = "timestamp" if "timestamp" in sensor.columns else "event_ts"
    if ts_col not in sensor.columns:
        ts_col = next((c for c in sensor.columns if "time" in c.lower()), sensor.columns[0])
    sensor["timestamp"] = pd.to_datetime(sensor[ts_col], utc=True)

    if fetch_days and fetch_days > 0:
        max_ts = sensor["timestamp"].max()
        cutoff = max_ts - timedelta(days=int(fetch_days))
        sensor = sensor[sensor["timestamp"] >= cutoff].copy()

    if not prod_df.empty:
        prod_well = _well_column(prod_df)
        date_col = next((c for c in prod_df.columns if c in ("date", "event_date")), None)
        if prod_well and date_col:
            prod_sub = prod_df[prod_df[prod_well].astype(str) == str(well_id)].copy()
            if not prod_sub.empty:
                prod_sub.columns = [c.lower() for c in prod_sub.columns]
                sensor.columns = [c.lower() for c in sensor.columns]
                prod_sub[date_col] = pd.to_datetime(prod_sub[date_col])
                sensor["_date"] = sensor["timestamp"].dt.date
                prod_sub["_date"] = pd.to_datetime(prod_sub[date_col]).dt.date
                for canonical, twin in [
                    ("oil_volume", "true_oil"),
                    ("water_volume", "true_water"),
                    ("gas_volume", "true_gas"),
                ]:
                    if canonical not in prod_sub.columns and twin in prod_sub.columns:
                        prod_sub[canonical] = prod_sub[twin]
                merge_cols = [
                    c for c in ["oil_volume", "water_volume", "gas_volume"] if c in prod_sub.columns
                ]
                if merge_cols:
                    sensor = sensor.merge(
                        prod_sub[["_date"] + merge_cols],
                        on="_date",
                        how="left",
                    )
                    sensor.drop(columns=["_date"], inplace=True, errors="ignore")

    for c in ["oil_volume", "water_volume", "gas_volume"]:
        if c in sensor.columns:
            sensor[c] = sensor[c].fillna(0)

    operational_cols = [
        "motor_current",
        "tubing_pressure",
        "casing_pressure",
        "strokes_per_minute",
        "discharge_pressure",
        "pump_intake_pressure",
    ]
    for c in operational_cols:
        if c in sensor.columns:
            sensor[c] = sensor[c].ffill()

    return sensor.sort_values("timestamp").reset_index(drop=True)


def max_partition_date(keys: List[str]) -> Optional[date]:
    dates: List[date] = []
    for key in keys:
        try:
            part = key.split("date=", 1)[1].split("/", 1)[0]
            dates.append(pd.to_datetime(part).date())
        except Exception:
            pass
    return max(dates) if dates else None
