"""
Data fetching and processing functions for anomaly review.

Uses S3 only (no Snowflake). Input: well_sensor_readings.parquet and
daily_production.parquet under s3://BUCKET/{ORG}/sensor_data/.
Output/history: s3://BUCKET/{ORG}/detections/anomaly_review/run_ts=.../anomaly_reviews.parquet.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger('AnomalyReviewData')

# =============================================================================
# S3 HELPERS
# =============================================================================

def _get_s3_client():
    import boto3
    return boto3.client("s3")


def _parse_s3_uri(uri: str):
    from urllib.parse import urlparse
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return p.netloc, p.path.lstrip("/")


def _load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Load a single parquet file from S3."""
    import io
    s3 = _get_s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def _load_parquet(path_or_uri: str) -> pd.DataFrame:
    """Load parquet from local path or s3:// URI."""
    if (path_or_uri or "").startswith("s3://"):
        bucket, key = _parse_s3_uri(path_or_uri)
        return _load_parquet_from_s3(bucket, key)
    return pd.read_parquet(path_or_uri)


def _get_s3_prefixes():
    from anomaly_review.config import (
        get_anomaly_review_s3_bucket,
        get_anomaly_review_input_prefix,
        get_anomaly_review_output_prefix,
    )
    bucket = get_anomaly_review_s3_bucket()
    input_prefix = get_anomaly_review_input_prefix()
    output_prefix = get_anomaly_review_output_prefix()
    return bucket, input_prefix, output_prefix


def _list_review_parquet_keys(bucket: str, output_prefix: str, max_runs: int = 20) -> List[str]:
    """List S3 keys for anomaly_reviews.parquet under run_ts=* (newest first)."""
    s3 = _get_s3_client()
    prefix = output_prefix.rstrip("/") + "/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("anomaly_reviews.parquet"):
                keys.append(k)
    keys.sort(reverse=True)  # run_ts=20250115... > run_ts=20250114...
    return keys[:max_runs]


def get_openai_client():
    """Create and return an OpenAI client. Returns None if openai package is not installed."""
    try:
        import openai
    except ImportError:
        logger.warning(
            "OpenAI client unavailable: 'openai' package not installed. "
            "Install with: pip install openai."
        )
        return None
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )


def write_anomaly_reviews_to_s3(reviews: List[dict]) -> Optional[str]:
    """
    Write anomaly review rows to S3 as a new run parquet.
    s3://BUCKET/{output_prefix}/run_ts=YYYYMMDDTHHMMSSZ/anomaly_reviews.parquet
    Returns the S3 key written, or None if bucket not set or reviews empty.
    """
    if not reviews:
        return None
    bucket, _, output_prefix = _get_s3_prefixes()
    if not bucket:
        return None
    import io
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{output_prefix.rstrip('/')}/run_ts={ts}/anomaly_reviews.parquet"
    s3 = _get_s3_client()
    # Flatten for parquet: include id, organization_id plus ui_text, impact_metrics, chart_data as JSON strings
    rows = []
    from anomaly_review.config import get_organization_id
    org_id = get_organization_id()
    for r in reviews:
        row = {k: v for k, v in r.items() if k in (
            "id", "well_id", "event_date", "detected_at", "anomaly_code", "category",
            "severity", "title", "impact_value", "status", "organization_id", "reason",
            "mute_till", "open_up", "user_id", "user_name",
        )}
        # Ensure organization_id is present to match Prisma model
        if not row.get("organization_id"):
            row["organization_id"] = org_id
        # Preserve NULL when user_id isn't available.
        # Only normalize empty string -> NULL.
        if row.get("user_id") == "":
            row["user_id"] = None
        if row.get("user_name") == "":
            row["user_name"] = None
        # Ensure reason column exists even if always null
        if "reason" not in row:
            row["reason"] = None
        for json_key in ("ui_text", "impact_metrics", "chart_data"):
            if json_key in r and r[json_key] is not None:
                import json as _json
                row[json_key] = _json.dumps(r[json_key], default=str) if isinstance(r[json_key], (dict, list)) else str(r[json_key])
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Wrote %s anomaly reviews to s3://%s/%s", len(reviews), bucket, key)
    return key


def _load_review_history_from_s3(
    bucket: str, output_prefix: str, well_id: str, anomaly_code: str, limit_runs: int = 10
) -> pd.DataFrame:
    """Load anomaly review rows from S3 run parquets, filtered by well_id and anomaly_code."""
    keys = _list_review_parquet_keys(bucket, output_prefix, max_runs=limit_runs)
    if not keys:
        return pd.DataFrame()
    rows = []
    for key in keys:
        try:
            df = _load_parquet_from_s3(bucket, key)
            if df.empty:
                continue
            df.columns = [c.lower() for c in df.columns]
            need = ["well_id", "anomaly_code", "event_date", "severity", "title"]
            if "well_id" not in df.columns or "anomaly_code" not in df.columns:
                continue
            mask = (df["well_id"].astype(str) == str(well_id)) & (
                df["anomaly_code"].astype(str) == str(anomaly_code)
            )
            sub = df.loc[mask]
            for _, r in sub.iterrows():
                rows.append({
                    "event_date": r.get("event_date"),
                    "severity": r.get("severity"),
                    "title": r.get("title"),
                })
        except Exception as e:
            logger.debug("Skip S3 key %s: %s", key, e)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# =============================================================================
# DATA FETCHING (S3 only)
# =============================================================================

def fetch_well_data(well_id: str, days: int = None) -> pd.DataFrame:
    """
    Fetch well sensor and production data from S3 parquet.

    Reads well_sensor_readings.parquet and daily_production.parquet from
    s3://BUCKET/{input_prefix}/, filters by well_id, joins on date.
    When days is set, uses this well's last N days (relative to well's max timestamp).

    Args:
        well_id: Well identifier
        days: Number of days of history (None = all). Cutoff = (well's max timestamp) - days.

    Returns:
        DataFrame with sensor and production columns (timestamp, oil_volume, etc.)
    """
    bucket, input_prefix, _ = _get_s3_prefixes()
    if not bucket:
        logger.error("S3_BUCKET not set; cannot fetch well data from S3.")
        return pd.DataFrame()

    base = input_prefix.rstrip("/")
    sensor_key = f"{base}/well_sensor_readings.parquet"
    prod_key = f"{base}/daily_production.parquet"

    try:
        sensor_df = _load_parquet_from_s3(bucket, sensor_key)
        prod_df = _load_parquet_from_s3(bucket, prod_key)
    except Exception as e:
        logger.error("Fetch from S3 failed for %s: %s", well_id, e)
        return pd.DataFrame()

    sensor_df.columns = [c.lower() for c in sensor_df.columns]
    prod_df.columns = [c.lower() for c in prod_df.columns]

    well_col = "well_id" if "well_id" in sensor_df.columns else "asset_id"
    if well_col not in sensor_df.columns:
        logger.error("No well_id/asset_id in sensor parquet")
        return pd.DataFrame()

    sensor_df = sensor_df[sensor_df[well_col].astype(str) == str(well_id)].copy()
    if sensor_df.empty:
        return pd.DataFrame()

    ts_col = "timestamp" if "timestamp" in sensor_df.columns else "event_ts"
    if ts_col not in sensor_df.columns:
        ts_col = next((c for c in sensor_df.columns if "time" in c.lower()), sensor_df.columns[0])
    sensor_df["timestamp"] = pd.to_datetime(sensor_df[ts_col], utc=True)

    if days:
        max_ts = sensor_df["timestamp"].max()
        cutoff = max_ts - timedelta(days=int(days))
        sensor_df = sensor_df[sensor_df["timestamp"] >= cutoff].copy()

    # Digital twin uses asset_id + date; production columns may be true_oil/true_water/true_gas or oil_volume/water_volume/gas_volume
    prod_well = "well_id" if "well_id" in prod_df.columns else "asset_id"
    date_col = next((c for c in prod_df.columns if c in ("date", "event_date")), None)
    if date_col and prod_well in prod_df.columns:
        prod_sub = prod_df[prod_df[prod_well].astype(str) == str(well_id)].copy()
        if not prod_sub.empty:
            prod_sub[date_col] = pd.to_datetime(prod_sub[date_col])
            sensor_df["_date"] = sensor_df["timestamp"].dt.date
            prod_sub["_date"] = pd.to_datetime(prod_sub[date_col]).dt.date
            # Support both digital twin (true_oil, true_water, true_gas) and canonical (oil_volume, water_volume, gas_volume)
            for canonical, twin in [("oil_volume", "true_oil"), ("water_volume", "true_water"), ("gas_volume", "true_gas")]:
                if canonical not in prod_sub.columns and twin in prod_sub.columns:
                    prod_sub[canonical] = prod_sub[twin]
            merge_cols = [c for c in ["oil_volume", "water_volume", "gas_volume"] if c in prod_sub.columns]
            if merge_cols:
                sensor_df = sensor_df.merge(
                    prod_sub[["_date"] + merge_cols],
                    on="_date",
                    how="left",
                )
                sensor_df.drop(columns=["_date"], inplace=True, errors="ignore")

    for c in ["oil_volume", "water_volume", "gas_volume"]:
        if c in sensor_df.columns:
            sensor_df[c] = sensor_df[c].fillna(0)
    operational_cols = [
        "motor_current", "tubing_pressure", "casing_pressure",
        "strokes_per_minute", "discharge_pressure", "pump_intake_pressure",
    ]
    for c in operational_cols:
        if c in sensor_df.columns:
            sensor_df[c] = sensor_df[c].ffill()
    return sensor_df.sort_values("timestamp").reset_index(drop=True)


def fetch_all_active_wells() -> List[str]:
    """Fetch list of distinct well_ids from S3 sensor or production parquet."""
    bucket, input_prefix, _ = _get_s3_prefixes()
    if not bucket:
        logger.error("S3_BUCKET not set.")
        return []

    base = input_prefix.rstrip("/")
    key = f"{base}/well_sensor_readings.parquet"
    try:
        df = _load_parquet_from_s3(bucket, key)
    except Exception as e:
        logger.error("Failed to fetch wells from S3: %s", e)
        return []
    well_col = "well_id" if "well_id" in df.columns else "asset_id"
    if well_col not in df.columns:
        well_col = df.columns[0]
    wells = df[well_col].dropna().astype(str).unique().tolist()
    return sorted(w for w in wells if w)


# =============================================================================
# ANOMALY HISTORY (from S3 output parquets)
# =============================================================================

def _format_history_rows(rows, exclude_date: str = None):
    """Build list of history lines from (event_date, severity, title) rows (dict or tuple)."""
    out = []
    for r in rows:
        if isinstance(r, dict):
            r_date = str(r.get("event_date", "")) if r else ""
            sev = r.get("severity", "N/A")
            title = r.get("title") or "Anomaly"
        else:
            r_date = str(r[0]) if r and len(r) > 0 else ""
            sev = r[1] if len(r) > 1 else "N/A"
            title = (r[2] if len(r) > 2 else "Anomaly") or "Anomaly"
        if exclude_date and r_date == str(exclude_date):
            continue
        out.append(f"- Date: {r_date}, Severity: {sev}, Title: {title}")
    return out


def get_recent_anomaly_history(
    well_id: str,
    anomaly_code: str,
    exclude_date: str = None,
) -> str:
    """
    Fetch recent anomaly history for a well+code for historical_context.
    S3-only: reads anomaly_review parquet outputs from S3.
    """
    bucket, _, output_prefix = _get_s3_prefixes()
    if bucket:
        try:
            df = _load_review_history_from_s3(bucket, output_prefix, well_id, anomaly_code, limit_runs=10)
            if not df.empty:
                df = df.drop_duplicates(subset=["event_date"], keep="first")
                df = df.sort_values("event_date", ascending=False).head(10)
                rows = df.to_dict("records")
                history_list = _format_history_rows(rows, exclude_date)
                if history_list:
                    return "\n".join(history_list)
        except Exception as e:
            logger.warning("get_recent_anomaly_history (S3) failed: %s", e)

    return "No prior history found."


def has_recent_anomaly(well_id: str, anomaly_code: str, within_days: int) -> bool:
    """Return True if this well has an anomaly_review row for the code within the last within_days (S3-only)."""
    from datetime import date
    cutoff = (date.today() - timedelta(days=within_days)).isoformat()
    bucket, _, output_prefix = _get_s3_prefixes()
    if bucket:
        try:
            df = _load_review_history_from_s3(bucket, output_prefix, well_id, anomaly_code, limit_runs=20)
            if df.empty:
                return False
            if "event_date" not in df.columns:
                return False
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            return (df["event_date"] >= cutoff).any()
        except Exception as e:
            logger.warning("has_recent_anomaly (S3) failed: %s", e)
    return False


def get_existing_anomaly_event_date(well_id: str, anomaly_code: str, within_days: int):
    """Return the most recent event_date for this well+code within within_days, or None. S3-only."""
    from datetime import date
    cutoff = (date.today() - timedelta(days=within_days)).isoformat()
    bucket, _, output_prefix = _get_s3_prefixes()
    if bucket:
        try:
            df = _load_review_history_from_s3(bucket, output_prefix, well_id, anomaly_code, limit_runs=20)
            if df.empty or "event_date" not in df.columns:
                return None
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            df = df[df["event_date"] >= pd.Timestamp(cutoff)]
            if df.empty:
                return None
            return str(df["event_date"].max().date())
        except Exception as e:
            logger.warning("get_existing_anomaly_event_date (S3) failed: %s", e)
    return None


# =============================================================================
# SENSOR METADATA (optional S3 or empty)
# =============================================================================

_SENSOR_METADATA_CACHE = None


def fetch_sensor_metadata() -> pd.DataFrame:
    """
    Fetch sensor definitions (Min/Max) from S3 or return empty DataFrame.
    Optional: set ANOMALY_REVIEW_SENSOR_DEFINITIONS_S3_URI (s3://bucket/key) to a parquet or CSV.
    Columns expected: field_name, min_value, max_value (case-insensitive).
    """
    global _SENSOR_METADATA_CACHE
    if _SENSOR_METADATA_CACHE is not None:
        return _SENSOR_METADATA_CACHE

    uri = os.getenv("ANOMALY_REVIEW_SENSOR_DEFINITIONS_S3_URI", "").strip()
    if not uri:
        _SENSOR_METADATA_CACHE = pd.DataFrame(columns=["field_name", "min_value", "max_value"])
        return _SENSOR_METADATA_CACHE

    try:
        if uri.startswith("s3://"):
            import io
            bucket, key = _parse_s3_uri(uri)
            s3 = _get_s3_client()
            buf = io.BytesIO()
            s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=buf)
            buf.seek(0)
            if key.lower().endswith(".parquet"):
                df = pd.read_parquet(buf)
            else:
                df = pd.read_csv(buf)
        else:
            df = pd.read_parquet(uri) if uri.endswith(".parquet") else pd.read_csv(uri)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if "field_name" in df.columns:
                df["field_name"] = df["field_name"].astype(str).str.lower()
        _SENSOR_METADATA_CACHE = df
        return df
    except Exception as e:
        logger.warning("Failed to fetch sensor metadata from %s: %s", uri, e)
    _SENSOR_METADATA_CACHE = pd.DataFrame(columns=["field_name", "min_value", "max_value"])
    return _SENSOR_METADATA_CACHE


# =============================================================================
# DATA PROCESSING
# =============================================================================

def resample_to_daily_if_subdaily(df: pd.DataFrame) -> pd.DataFrame:
    """If multiple rows per day, aggregate to one row per day for baseline comparison."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
        return df
    n_rows = len(df)
    days = df.index.normalize().unique()
    n_days = len(days)
    if n_rows <= n_days * 1.5:
        return df
    by_day = df.groupby(df.index.normalize())
    agg_dict = {}
    for c in df.columns:
        if df[c].dtype.kind in 'iufc':
            agg_dict[c] = 'mean'
        else:
            agg_dict[c] = 'first'
    resampled = by_day.agg(agg_dict)
    resampled.index = resampled.index + pd.Timedelta(hours=23, minutes=59, seconds=59)
    resampled.index.name = df.index.name
    return resampled.sort_index()


def calculate_rolling_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling baselines for various metrics."""
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp').sort_index()
    df = resample_to_daily_if_subdaily(df)
    MIN_PERIODS = 1

    if 'oil_volume' in df.columns:
        df['oil_baseline_90d'] = df['oil_volume'].rolling('90D', min_periods=MIN_PERIODS).mean()
        df['oil_avg_7d'] = df['oil_volume'].rolling('7D', min_periods=3).mean()
        df['oil_avg_30d'] = df['oil_volume'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['oil_std_90d'] = df['oil_volume'].rolling('90D', min_periods=MIN_PERIODS).std()
    if 'gas_volume' in df.columns:
        df['gas_baseline_90d'] = df['gas_volume'].rolling('90D', min_periods=MIN_PERIODS).mean()
        df['gas_avg_7d'] = df['gas_volume'].rolling('7D', min_periods=3).mean()
        df['gas_avg_30d'] = df['gas_volume'].rolling('30D', min_periods=MIN_PERIODS).mean()
    if 'water_volume' in df.columns:
        df['water_avg_30d'] = df['water_volume'].rolling('30D', min_periods=MIN_PERIODS).mean()
    if 'pump_fillage' in df.columns:
        df['fillage_avg_24h'] = df['pump_fillage'].rolling('24h', min_periods=1).mean()
        df['fillage_avg_3d'] = df['pump_fillage'].rolling('3D', min_periods=3).mean()
    if 'injection_rate' in df.columns:
        df['injection_avg_7d'] = df['injection_rate'].rolling('7D', min_periods=MIN_PERIODS).mean()
    if 'tubing_pressure' in df.columns:
        df['tp_avg'] = df['tubing_pressure'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['tp_std'] = df['tubing_pressure'].rolling('3D', min_periods=3).std()
    if 'discharge_pressure' in df.columns:
        df['dp_avg'] = df['discharge_pressure'].rolling('30D', min_periods=MIN_PERIODS).mean()
    if 'motor_current' in df.columns:
        df['amps_avg'] = df['motor_current'].rolling('30D', min_periods=MIN_PERIODS).mean()
    if 'oil_volume' in df.columns:
        roll_oil = df['oil_volume'].rolling('90D', min_periods=MIN_PERIODS).mean()
        df['oil_baseline_90d_1d_ago'] = roll_oil.reindex(roll_oil.index - pd.Timedelta(days=1)).set_axis(roll_oil.index)
    if 'motor_current' in df.columns:
        roll_amps = df['motor_current'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['amps_avg_1d_ago'] = roll_amps.reindex(roll_amps.index - pd.Timedelta(days=1)).set_axis(roll_amps.index)
    if 'strokes_per_minute' in df.columns:
        df['spm_avg'] = df['strokes_per_minute'].rolling('30D', min_periods=MIN_PERIODS).mean()

    df = df.reset_index()
    return df
