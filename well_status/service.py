#!/usr/bin/env python3
"""
Well Status Service
Manages well operational status (RUNNING, IDLE, SHUTDOWN) based on sensor readings activity.

Status Logic:
- SHUTDOWN: No sensor readings for last 24 hours (1 day)
- IDLE: Recent readings but low/zero production metrics based on lift type
- RUNNING: Normal operation with recent readings

Run this service periodically (e.g., every hour via cron) to update well statuses.
"""

import os
import logging
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import io

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Set to logging.WARNING to suppress INFO logs in production/cron
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SHUTDOWN_THRESHOLD_HOURS = 24  # No readings for 24 hours = SHUTDOWN
PARTITIONED_SENSOR_DIR = "well_sensor_readings"
DEFAULT_PARTITION_START_DATE = "2026-01-01"
PARTITION_DATE_RE = re.compile(r"/date=(\d{4}-\d{2}-\d{2})/")
DEFAULT_CHECKPOINT_FILENAME = "well_status_checkpoint.json"

LIFT_TYPE_ALIASES = {
    "rod pump": "Rod Pump",
    "pumping unit": "Rod Pump",
    "esp": "ESP",
    "gas lift": "Gas Lift",
}

# IDLE detection thresholds - multiple metrics per lift type
# Each metric has: field_name, threshold, operator ('lt' = less than, 'gt' = greater than), weight
IDLE_METRICS = {
    'Rod Pump': [
        # Primary indicators
        {'field': 'strokes_per_minute', 'threshold': 2.0, 'operator': 'lt', 'weight': 3, 'description': 'SPM very low'},
        {'field': 'pump_fillage', 'threshold': 10.0, 'operator': 'lt', 'weight': 2, 'description': 'Pump fillage very low'},
        {'field': 'torque', 'threshold': 50000, 'operator': 'lt', 'weight': 2, 'description': 'Torque minimal'},
        # Secondary indicators
        {'field': 'polish_rod_load', 'threshold': 5000, 'operator': 'lt', 'weight': 1, 'description': 'Rod load minimal'},
        {'field': 'tubing_pressure', 'threshold': 50, 'operator': 'lt', 'weight': 1, 'description': 'Tubing pressure very low'},
    ],
    'ESP': [
        # Primary indicators
        {'field': 'motor_current', 'threshold': 5.0, 'operator': 'lt', 'weight': 3, 'description': 'Motor current very low'},
        {'field': 'motor_temp', 'threshold': 80.0, 'operator': 'lt', 'weight': 2, 'description': 'Motor temp at ambient'},
        {'field': 'discharge_pressure', 'threshold': 100, 'operator': 'lt', 'weight': 2, 'description': 'Discharge pressure minimal'},
        # Secondary indicators
        {'field': 'motor_voltage', 'threshold': 100, 'operator': 'lt', 'weight': 1, 'description': 'Motor voltage very low'},
        {'field': 'pump_intake_pressure', 'threshold': 50, 'operator': 'lt', 'weight': 1, 'description': 'Intake pressure very low'},
    ],
    'Gas Lift': [
        # Primary indicators
        {'field': 'injection_rate', 'threshold': 5.0, 'operator': 'lt', 'weight': 3, 'description': 'Injection rate very low'},
        {'field': 'injection_pressure', 'threshold': 200, 'operator': 'lt', 'weight': 2, 'description': 'Injection pressure low'},
        {'field': 'bottomhole_pressure', 'threshold': 100, 'operator': 'lt', 'weight': 2, 'description': 'BHP very low'},
        # Secondary indicators
        {'field': 'injection_temperature', 'threshold': 50, 'operator': 'lt', 'weight': 1, 'description': 'Injection temp at ambient'},
        {'field': 'cycle_time', 'threshold': 5, 'operator': 'lt', 'weight': 1, 'description': 'Cycle time very short'},
    ]
}

IDLE_SCORE_THRESHOLD = 4  # Need at least 4 weighted points to be IDLE


def _default_sensor_parquet_s3() -> str:
    """
    Build default S3 path for well_sensor_readings.parquet from env:
      - S3_BUCKET (e.g. agentic-energy-ml-models-200937443240)
      - ORGANIZATION_ID (tenant id)
    """
    bucket = os.getenv("S3_BUCKET", "agentic-energy-ml-models-200937443240")
    org_id = os.getenv("ORGANIZATION_ID")
    if not org_id:
        raise ValueError("ORGANIZATION_ID env var is required to build default sensor parquet path")
    return f"s3://{bucket}/{org_id}/sensor_data/well_sensor_readings.parquet"


DEFAULT_SENSOR_PARQUET_S3 = _default_sensor_parquet_s3()


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _normalize_lift_type(lift_type: Optional[str]) -> Optional[str]:
    if lift_type is None:
        return None
    text = str(lift_type).strip()
    if not text:
        return None
    return LIFT_TYPE_ALIASES.get(text.lower(), text)


def _get_source_well_status(well: Dict) -> Optional[str]:
    for key in ("well_status", "run_status", "status"):
        value = well.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _partition_prefix_from_sensor_uri(uri: str) -> str:
    _bucket, key = _parse_s3_uri(uri)
    clean = key.rstrip("/")
    if clean.endswith(".parquet"):
        base = clean.rsplit("/", 1)[0]
        return f"{base}/{PARTITIONED_SENSOR_DIR}/"
    return clean + "/"


def _base_partition_start_date() -> Optional[pd.Timestamp]:
    raw = (os.getenv("WELL_STATUS_PARTITION_START_DATE") or DEFAULT_PARTITION_START_DATE).strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        logger.warning("Invalid WELL_STATUS_PARTITION_START_DATE=%r; scanning all partitions", raw)
        return None
    return ts.normalize()


def _checkpoint_s3_uri(sensor_uri: str) -> str:
    bucket, key = _parse_s3_uri(sensor_uri)
    clean = key.rstrip("/")
    base = clean.rsplit("/", 1)[0] if clean.endswith(".parquet") else clean
    return f"s3://{bucket}/{base}/_state/{DEFAULT_CHECKPOINT_FILENAME}"


def _load_checkpoint(checkpoint_uri: str) -> Optional[Dict]:
    bucket, key = _parse_s3_uri(checkpoint_uri)
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    return json.loads(obj["Body"].read().decode("utf-8"))


def _save_checkpoint(checkpoint_uri: str, payload: Dict) -> None:
    bucket, key = _parse_s3_uri(checkpoint_uri)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload, indent=2).encode("utf-8"))


def _checkpoint_timestamp(checkpoint: Optional[Dict]) -> Optional[pd.Timestamp]:
    if not checkpoint:
        return None
    raw = checkpoint.get("sensor_high_watermark")
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _effective_partition_start_date(checkpoint: Optional[Dict]) -> Optional[pd.Timestamp]:
    start_date = _base_partition_start_date()
    checkpoint_ts = _checkpoint_timestamp(checkpoint)
    if checkpoint_ts is None:
        return start_date
    checkpoint_date = checkpoint_ts.normalize()
    if getattr(checkpoint_date, "tzinfo", None) is not None:
        checkpoint_date = checkpoint_date.tz_localize(None)
    if start_date is None:
        return checkpoint_date
    return max(start_date, checkpoint_date)


def _extract_partition_date(key: str) -> Optional[pd.Timestamp]:
    match = PARTITION_DATE_RE.search(f"/{key}")
    if not match:
        return None
    ts = pd.to_datetime(match.group(1), errors="coerce")
    if pd.isna(ts):
        return None
    return ts.normalize()


def _list_partition_parquet_keys(
    bucket: str,
    prefix: str,
    *,
    start_date: Optional[pd.Timestamp] = None,
) -> List[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    if start_date is not None:
        logger.info("Well status partition scan start date: %s", start_date.date().isoformat())
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key.endswith(".parquet"):
                continue
            partition_date = _extract_partition_date(key)
            if start_date is not None and partition_date is not None and partition_date < start_date:
                logger.info(
                    "Skipping old partition file before start date: s3://%s/%s (partition=%s)",
                    bucket,
                    key,
                    partition_date.date().isoformat(),
                )
                continue
            logger.info("Including partition file for well status: s3://%s/%s", bucket, key)
            keys.append(key)
    keys.sort()
    logger.info("Well status partition scan selected %s parquet files", len(keys))
    return keys


def _load_partitioned_sensor_parquet_from_s3(bucket: str, prefix: str) -> pd.DataFrame:
    logger.info("Loading partitioned sensor parquet from s3://%s/%s", bucket, prefix)
    keys = _list_partition_parquet_keys(bucket, prefix)
    if not keys:
        raise FileNotFoundError(f"No parquet files found under s3://{bucket}/{prefix}")

    s3 = boto3.client("s3")
    frames: List[pd.DataFrame] = []
    for key in keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _latest_rows_from_partitioned_s3(
    bucket: str,
    prefix: str,
    *,
    start_date: Optional[pd.Timestamp] = None,
    exclusive_start_ts: Optional[pd.Timestamp] = None,
) -> Tuple[List[Dict], Optional[pd.Timestamp]]:
    """
    Read partition parquet files one-by-one and keep only the newest row per well.

    This avoids loading the full transformed dataset into memory for well status.
    """
    logger.info("Computing latest readings from partitioned sensor parquet at s3://%s/%s", bucket, prefix)
    keys = _list_partition_parquet_keys(bucket, prefix, start_date=start_date)
    if not keys:
        return [], None

    s3 = boto3.client("s3")
    latest_by_well: Dict[str, Dict] = {}
    latest_ts_by_well: Dict[str, pd.Timestamp] = {}
    max_seen_ts: Optional[pd.Timestamp] = None

    for key in keys:
        logger.info("Checking partition file: s3://%s/%s", bucket, key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        batch = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        batch.columns = [c.lower() for c in batch.columns]
        ts_col = None
        for cand in ("timestamp", "time", "datetime"):
            if cand in batch.columns:
                ts_col = cand
                break
        if ts_col is None:
            logger.warning("Skipping partition without timestamp column: s3://%s/%s", bucket, key)
            continue
        batch[ts_col] = pd.to_datetime(batch[ts_col], errors="coerce", utc=True)
        batch = batch.dropna(subset=[ts_col])
        if exclusive_start_ts is not None:
            batch = batch[batch[ts_col] > exclusive_start_ts]
            if batch.empty:
                logger.info(
                    "No rows after checkpoint in partition file: s3://%s/%s (checkpoint=%s)",
                    bucket,
                    key,
                    exclusive_start_ts.isoformat(),
                )
                continue
        batch_max = batch[ts_col].max()
        if batch_max is not None and not pd.isna(batch_max):
            max_seen_ts = batch_max if max_seen_ts is None else max(max_seen_ts, batch_max)
        rows = _compute_latest_readings_from_df(batch)
        for rec in rows:
            well_id = str(rec.get("well_id"))
            ts = pd.to_datetime(rec.get("last_reading_timestamp"), errors="coerce", utc=True)
            if not well_id or pd.isna(ts):
                continue
            current = latest_ts_by_well.get(well_id)
            if current is None or ts > current:
                latest_ts_by_well[well_id] = ts
                latest_by_well[well_id] = rec

    wells = list(latest_by_well.values())
    logger.info("Computed latest readings for %s wells from partitioned sensor parquet", len(wells))
    return wells, max_seen_ts


def _load_sensor_parquet_from_s3(uri: str) -> pd.DataFrame:
    bucket, key = _parse_s3_uri(uri)
    if not key.endswith(".parquet"):
        return _load_partitioned_sensor_parquet_from_s3(bucket, key.rstrip("/") + "/")

    logger.info("Loading sensor parquet from s3://%s/%s", bucket, key)
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        buf = io.BytesIO(data)
        return pd.read_parquet(buf)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code not in ("NoSuchKey", "404", "NotFound"):
            raise
        partition_prefix = _partition_prefix_from_sensor_uri(uri)
        logger.info(
            "Root sensor parquet not found; falling back to partitioned dataset at s3://%s/%s",
            bucket,
            partition_prefix,
        )
        return _load_partitioned_sensor_parquet_from_s3(bucket, partition_prefix)


def _compute_latest_readings_from_df(df: pd.DataFrame) -> List[Dict]:
    """
    From full well_sensor_readings dataframe, compute one latest row per well_id.
    Returns list of dicts with keys used by determine_well_status.
    """
    if df.empty:
        return []

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    well_col = "well_id" if "well_id" in df.columns else "asset_id"
    if well_col not in df.columns:
        raise ValueError(f"Expected well_id or asset_id column in parquet; got columns: {df.columns.tolist()}")

    ts_col = None
    for cand in ("timestamp", "time", "datetime"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        ts_col = df.columns[1]

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(by=[well_col, ts_col])
    latest = df.groupby(well_col).tail(1).copy()

    wells: List[Dict] = []
    for _, row in latest.iterrows():
        rec = row.to_dict()
        rec["well_id"] = str(rec.get(well_col))
        rec["last_reading_timestamp"] = rec.get(ts_col)
        wells.append(rec)

    logger.info("Computed latest readings for %s wells from sensor parquet", len(wells))
    return wells


def get_all_wells_from_s3(sensor_parquet_s3: Optional[str] = None) -> List[Dict]:
    """
    Get all distinct wells with their latest reading from S3 parquet.
    """
    wells, _max_seen_ts = _get_all_wells_from_s3_with_checkpoint(sensor_parquet_s3)
    return wells


def _get_all_wells_from_s3_with_checkpoint(
    sensor_parquet_s3: Optional[str] = None,
    checkpoint: Optional[Dict] = None,
) -> Tuple[List[Dict], Optional[pd.Timestamp]]:
    sensor_uri = sensor_parquet_s3 or os.environ.get("SENSOR_PARQUET_S3", DEFAULT_SENSOR_PARQUET_S3)
    bucket, key = _parse_s3_uri(sensor_uri)
    start_date = _effective_partition_start_date(checkpoint)
    checkpoint_ts = _checkpoint_timestamp(checkpoint)

    if key.endswith(".parquet"):
        partition_prefix = _partition_prefix_from_sensor_uri(sensor_uri)
        partition_keys = _list_partition_parquet_keys(bucket, partition_prefix, start_date=start_date)
        if partition_keys:
            logger.info(
                "Using partitioned sensor dataset for well status from s3://%s/%s",
                bucket,
                partition_prefix,
            )
            return _latest_rows_from_partitioned_s3(
                bucket,
                partition_prefix,
                start_date=start_date,
                exclusive_start_ts=checkpoint_ts,
            )

    if key.endswith(".parquet"):
        try:
            df = _load_sensor_parquet_from_s3(sensor_uri)
            df.columns = [c.lower() for c in df.columns]
            ts_col = None
            for cand in ("timestamp", "time", "datetime"):
                if cand in df.columns:
                    ts_col = cand
                    break
            if ts_col and checkpoint_ts is not None:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
                df = df.dropna(subset=[ts_col])
                df = df[df[ts_col] > checkpoint_ts]
            max_seen_ts = None
            if ts_col and not df.empty:
                batch_max = pd.to_datetime(df[ts_col], errors="coerce", utc=True).max()
                if batch_max is not None and not pd.isna(batch_max):
                    max_seen_ts = batch_max
            return _compute_latest_readings_from_df(df), max_seen_ts
        except FileNotFoundError:
            partition_prefix = _partition_prefix_from_sensor_uri(sensor_uri)
            return _latest_rows_from_partitioned_s3(
                bucket,
                partition_prefix,
                start_date=start_date,
                exclusive_start_ts=checkpoint_ts,
            )
    return _latest_rows_from_partitioned_s3(
        bucket,
        key.rstrip("/") + "/",
        start_date=start_date,
        exclusive_start_ts=checkpoint_ts,
    )


def get_anomaly_counts(well_id: str, hours: int = 24, conn=None) -> Tuple[int, int]:
    """
    Get anomaly counts for a well in the last N hours.
    Returns (total_count, critical_count).
    If conn is provided, use it and do not close it (caller owns it).
    """
    own_conn = conn is None
    if conn is None:
        conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN SEVERITY = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_count
        FROM WELL_ANOMALIES
        WHERE WELL_ID = %s
        AND TIMESTAMP >= DATEADD(hour, -%s, CURRENT_TIMESTAMP())
        """
        
        cursor.execute(query, (well_id, hours))
        row = cursor.fetchone()
        
        if row:
            return (int(row[0] or 0), int(row[1] or 0))
        return (0, 0)
        
    except Exception as e:
        logger.error(f"Error fetching anomaly counts for {well_id}: {e}")
        return (0, 0)
    finally:
        cursor.close()
        if own_conn:
            conn.close()


def calculate_idle_score(well: Dict, lift_type: str) -> Tuple[int, List[str]]:
    """
    Calculate IDLE score based on multiple metrics.
    Returns: (total_weighted_score, list_of_triggered_reasons)
    """
    if not lift_type or lift_type not in IDLE_METRICS:
        return (0, [])
    
    metrics = IDLE_METRICS[lift_type]
    total_score = 0
    triggered_reasons = []
    
    for metric in metrics:
        field = metric['field']
        threshold = metric['threshold']
        operator = metric['operator']
        weight = metric['weight']
        description = metric['description']
        
        actual_value = well.get(field)
        
        # Skip if no value available
        if actual_value is None:
            continue
        
        try:
            actual_value = float(actual_value)
            
            # Check if metric triggers IDLE condition
            triggered = False
            if operator == 'lt' and actual_value < threshold:
                triggered = True
            elif operator == 'gt' and actual_value > threshold:
                triggered = True
            elif operator == 'eq' and actual_value == threshold:
                triggered = True
            
            if triggered:
                total_score += weight
                triggered_reasons.append(f"{description} ({field}={actual_value:.2f}, threshold={threshold})")
                
        except (ValueError, TypeError):
            continue
    
    return (total_score, triggered_reasons)


def determine_well_status(well: Dict) -> Tuple[str, str]:
    """
    Determine well status based on:
    1. Time since last reading (SHUTDOWN if > 24 hours)
    2. Multi-metric IDLE detection (IDLE if weighted score >= threshold)
    3. Default to RUNNING
    
    Returns: (status, reason)
    """
    last_timestamp = well.get('last_reading_timestamp')
    raw_lift_type = well.get('lift_type')
    lift_type = _normalize_lift_type(raw_lift_type)
    display_lift_type = raw_lift_type or lift_type
    
    # Check if we have timestamp
    if not last_timestamp:
        return ('SHUTDOWN', 'No sensor readings found')
    
    # Calculate hours since last reading
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if isinstance(last_timestamp, str):
        try:
            last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        except:
            last_timestamp = datetime.strptime(last_timestamp[:19], '%Y-%m-%d %H:%M:%S')
    
    # Handle timezone-naive comparison
    if hasattr(last_timestamp, 'tzinfo') and last_timestamp.tzinfo is not None:
        last_timestamp = last_timestamp.replace(tzinfo=None)
    
    hours_since_reading = (now - last_timestamp).total_seconds() / 3600
    
    # CONDITION 1: SHUTDOWN if no readings for > 24 hours
    if hours_since_reading > SHUTDOWN_THRESHOLD_HOURS:
        return ('SHUTDOWN', f'No sensor readings for {hours_since_reading:.1f} hours (threshold: {SHUTDOWN_THRESHOLD_HOURS}h)')
    
    # CONDITION 2: Multi-metric IDLE detection
    if lift_type and lift_type in IDLE_METRICS:
        idle_score, idle_reasons = calculate_idle_score(well, lift_type)
        
        if idle_score >= IDLE_SCORE_THRESHOLD:
            # Summarize top reasons (max 3)
            top_reasons = idle_reasons[:3]
            reason_summary = "; ".join(top_reasons)
            return ('IDLE', f'{display_lift_type} shows low activity (score={idle_score}/{IDLE_SCORE_THRESHOLD}): {reason_summary}')
    
    # DEFAULT: RUNNING
    return ('RUNNING', f'Normal operation, last reading {hours_since_reading:.1f} hours ago')


def _build_status_rows(wells: List[Dict]) -> pd.DataFrame:
    """
    For each well dict, compute status and build a status dataframe.
    """
    rows = []
    now = datetime.now(timezone.utc)

    for well in wells:
        well_id = str(well.get("well_id"))
        if not well_id or well_id == "None":
            continue

        status, reason = determine_well_status(well)

        last_timestamp = well.get("last_reading_timestamp")
        reading_age_hours = None
        if last_timestamp:
            if isinstance(last_timestamp, str):
                try:
                    last_ts = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
                except Exception:
                    last_ts = datetime.strptime(last_timestamp[:19], "%Y-%m-%d %H:%M:%S")
            else:
                last_ts = last_timestamp
            if getattr(last_ts, "tzinfo", None) is not None:
                last_ts = last_ts.replace(tzinfo=None)
            reading_age_hours = (now.replace(tzinfo=None) - last_ts).total_seconds() / 3600.0

        rows.append(
            {
                "well_id": well_id,
                "lift_type": well.get("lift_type"),
                "well_status": _get_source_well_status(well),
                "status": status,
                "last_reading_timestamp": well.get("last_reading_timestamp"),
                "last_reading_age_hours": reading_age_hours,
                "strokes_per_minute": well.get("strokes_per_minute"),
                "motor_current": well.get("motor_current"),
                "injection_rate": well.get("injection_rate"),
                "anomaly_count_24h": 0,
                "critical_anomaly_count_24h": 0,
                "status_reason": reason,
                "updated_at": now,
            }
        )

    return pd.DataFrame(rows)


def get_well_status(well_id: str) -> Optional[Dict]:
    """Get current status for a specific well from PostgreSQL (org schema)."""
    from well_status.postgres import get_well_status_row

    return get_well_status_row(well_id)


def get_all_well_statuses() -> List[Dict]:
    """Get all well statuses from PostgreSQL (org schema)."""
    from well_status.postgres import get_all_well_status_rows

    return get_all_well_status_rows()


def update_all_well_statuses() -> Dict:
    """
    Main function to update status for all wells.

    Reads sensor parquet from S3 (partition layout), computes RUNNING/IDLE/SHUTDOWN,
    and upserts rows into PostgreSQL schema ORGANIZATION_ID.well_status.
    Checkpoint remains on S3 for incremental sensor reads.
    """
    from well_status.postgres import upsert_well_status_dataframe

    logger.info("=" * 50)
    logger.info("Starting well status update job (S3 sensor -> PostgreSQL)")
    logger.info("=" * 50)

    sensor_uri = os.environ.get("SENSOR_PARQUET_S3", DEFAULT_SENSOR_PARQUET_S3)
    checkpoint_uri = _checkpoint_s3_uri(sensor_uri)
    checkpoint = _load_checkpoint(checkpoint_uri)
    force_full = os.getenv("WELL_STATUS_FORCE_FULL", "").strip().lower() in ("1", "true", "yes")
    if force_full:
        logger.info("WELL_STATUS_FORCE_FULL enabled: ignoring checkpoint, full reprocess into PostgreSQL")
        checkpoint = None
    start_date = _effective_partition_start_date(checkpoint)
    checkpoint_ts = _checkpoint_timestamp(checkpoint)

    logger.info("Well status input sensor URI: %s", sensor_uri)
    if start_date is not None:
        logger.info("Well status partition start date: %s", start_date.date().isoformat())
    logger.info("Well status checkpoint URI: %s", checkpoint_uri)
    if checkpoint_ts is not None:
        logger.info("Well status checkpoint timestamp: %s", checkpoint_ts.isoformat())
    else:
        logger.info("Well status checkpoint timestamp: none (first incremental run from start date)")

    wells, max_seen_ts = _get_all_wells_from_s3_with_checkpoint(sensor_uri, checkpoint=checkpoint)

    if not wells:
        logger.warning("No new wells/readings found after checkpoint; PostgreSQL unchanged")
        checkpoint_payload = dict(checkpoint or {})
        checkpoint_payload["organization_id"] = os.getenv("ORGANIZATION_ID")
        checkpoint_payload["last_run_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint_payload["last_run_status"] = "skipped_no_new_rows"
        if checkpoint_ts is not None:
            checkpoint_payload["sensor_high_watermark"] = checkpoint_ts.isoformat()
        _save_checkpoint(checkpoint_uri, checkpoint_payload)
        return {
            "status": "skipped",
            "reason": "no_new_rows_since_checkpoint",
            "wells_processed": 0,
            "rows_upserted": 0,
        }

    status_df = _build_status_rows(wells)
    rows_upserted = upsert_well_status_dataframe(status_df)

    status_counts = {"RUNNING": 0, "IDLE": 0, "SHUTDOWN": 0}
    if "status" in status_df.columns:
        for s in status_counts.keys():
            status_counts[s] = int((status_df["status"] == s).sum())

    logger.info("=" * 50)
    logger.info("Well status update complete")
    logger.info(
        "Summary: RUNNING=%s, IDLE=%s, SHUTDOWN=%s, rows_upserted=%s",
        status_counts["RUNNING"],
        status_counts["IDLE"],
        status_counts["SHUTDOWN"],
        rows_upserted,
    )
    logger.info("=" * 50)

    checkpoint_payload = dict(checkpoint or {})
    checkpoint_payload["organization_id"] = os.getenv("ORGANIZATION_ID")
    checkpoint_payload["last_run_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_payload["last_run_status"] = "success"
    if max_seen_ts is not None:
        checkpoint_payload["sensor_high_watermark"] = pd.to_datetime(max_seen_ts, utc=True).isoformat()
    _save_checkpoint(checkpoint_uri, checkpoint_payload)
    logger.info(
        "Saved well status checkpoint to %s (sensor_high_watermark=%s)",
        checkpoint_uri,
        checkpoint_payload.get("sensor_high_watermark"),
    )

    return {
        "status": "success",
        "wells_processed": len(status_df),
        "rows_upserted": rows_upserted,
        "summary": status_counts,
    }


def get_status_for_reading(well_id: str, lift_type: str, readings: Dict) -> str:
    """
    Get status for a new reading being inserted.
    This is called by real-time ingest paths when inserting a reading.
    
    Since we're inserting a reading, well is active (not SHUTDOWN).
    Determine RUNNING vs IDLE based on multiple production metrics.
    
    Uses the same multi-metric scoring as the background service.
    """
    lift_type = _normalize_lift_type(lift_type)
    if not lift_type or lift_type not in IDLE_METRICS:
        return 'RUNNING'
    
    # Calculate IDLE score using the same logic as background service
    idle_score, idle_reasons = calculate_idle_score(readings, lift_type)
    
    if idle_score >= IDLE_SCORE_THRESHOLD:
        logger.info(f"[{well_id}] Real-time status: IDLE (score={idle_score}, reasons={idle_reasons[:2]})")
        return 'IDLE'
    
    return 'RUNNING'


# =============================================================================
# SCHEDULER CONFIGURATION NOTE:
# Currently set to 24-hour interval for batch processing.
# When real-time data pipeline is available, change to 10-minute schedule:
#   run_scheduler(interval_hours=0.167)  # 10 minutes = 0.167 hours
# Or modify the function to accept minutes directly.
# =============================================================================

def run_scheduler(interval_hours: int = 24):
    """
    Run the well status update automatically every N hours.
    Default: every 24 hours.
    
    Args:
        interval_hours: Hours between each run (default 24)
    """
    import time
    
    interval_seconds = interval_hours * 3600
    
    logger.info(f"Starting automatic well status scheduler (interval: {interval_hours} hours)")
    
    while True:
        try:
            # Run the update
            update_all_well_statuses()
            
            # Log next run time
            next_run = datetime.now(timezone.utc) + timedelta(hours=interval_hours)
            logger.info(f"Next automatic run scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Sleep until next run
            time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            # Wait 5 minutes before retrying on error
            time.sleep(300)


if __name__ == "__main__":
    """
    Run this script directly to update all well statuses.
    
    Usage:
        python well_status_service.py          # Run once
        python well_status_service.py --auto   # Run automatically every 24 hours
        python well_status_service.py --auto 12  # Run automatically every 12 hours
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Automatic mode - run every N hours (default 24)
        interval = 24
        if len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                interval = 24
        run_scheduler(interval_hours=interval)
    else:
        # Single run mode
        update_all_well_statuses()
