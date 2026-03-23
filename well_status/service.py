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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import io

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Set to logging.WARNING to suppress INFO logs in production/cron
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SHUTDOWN_THRESHOLD_HOURS = 24  # No readings for 24 hours = SHUTDOWN

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


def _load_sensor_parquet_from_s3(uri: str) -> pd.DataFrame:
    bucket, key = _parse_s3_uri(uri)
    logger.info("Loading sensor parquet from s3://%s/%s", bucket, key)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    buf = io.BytesIO(data)
    return pd.read_parquet(buf)


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
    sensor_uri = sensor_parquet_s3 or os.environ.get("SENSOR_PARQUET_S3", DEFAULT_SENSOR_PARQUET_S3)
    df = _load_sensor_parquet_from_s3(sensor_uri)
    return _compute_latest_readings_from_df(df)


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
    lift_type = well.get('lift_type')
    
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
            return ('IDLE', f'{lift_type} shows low activity (score={idle_score}/{IDLE_SCORE_THRESHOLD}): {reason_summary}')
    
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


def _load_status_parquet_from_s3(output_s3: str) -> Optional[pd.DataFrame]:
    """
    Helper to load well_status.parquet from S3 if present.
    """
    try:
        bucket, key = _parse_s3_uri(output_s3)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(obj["Body"])
    except Exception as e:
        logger.warning("Could not load status parquet from %s: %s", output_s3, e)
        return None


def get_well_status(well_id: str) -> Optional[Dict]:
    """
    Get current status for a specific well from S3 well_status.parquet.
    """
    sensor_uri = os.environ.get("SENSOR_PARQUET_S3", DEFAULT_SENSOR_PARQUET_S3)
    default_out = sensor_uri.rsplit("/", 1)[0] + "/well_status.parquet"
    output_uri = os.environ.get("OUTPUT_PARQUET_S3", default_out)

    df = _load_status_parquet_from_s3(output_uri)
    if df is None or df.empty:
        return None

    df = df.copy()
    if "well_id" not in df.columns:
        return None
    match = df[df["well_id"].astype(str) == str(well_id)]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    return row


def get_all_well_statuses() -> List[Dict]:
    """
    Get all well statuses from S3 well_status.parquet.
    """
    sensor_uri = os.environ.get("SENSOR_PARQUET_S3", DEFAULT_SENSOR_PARQUET_S3)
    default_out = sensor_uri.rsplit("/", 1)[0] + "/well_status.parquet"
    output_uri = os.environ.get("OUTPUT_PARQUET_S3", default_out)

    df = _load_status_parquet_from_s3(output_uri)
    if df is None or df.empty:
        return []

    # Optional ordering similar to previous Snowflake query
    if "status" in df.columns and "well_id" in df.columns:
        status_order = {"SHUTDOWN": 1, "IDLE": 2}
        df = df.copy()
        df["_order"] = df["status"].map(status_order).fillna(3)
        df = df.sort_values(by=["_order", "well_id"]).drop(columns=["_order"])

    return [r._asdict() if hasattr(r, "_asdict") else dict(r) for _, r in df.iterrows()]


def update_all_well_statuses():
    """
    Main function to update status for all wells using S3 parquet only.
    Reads SENSOR_PARQUET_S3 (or default), computes status, and writes well_status.parquet
    alongside the sensor parquet.
    """
    logger.info("=" * 50)
    logger.info("Starting well status update job (S3 parquet)")
    logger.info("=" * 50)

    sensor_uri = os.environ.get("SENSOR_PARQUET_S3", DEFAULT_SENSOR_PARQUET_S3)
    default_out = sensor_uri.rsplit("/", 1)[0] + "/well_status.parquet"
    output_uri = os.environ.get("OUTPUT_PARQUET_S3", default_out)

    df_sensor = _load_sensor_parquet_from_s3(sensor_uri)
    wells = _compute_latest_readings_from_df(df_sensor)

    if not wells:
        logger.warning("No wells found in sensor parquet")
        return {}

    status_df = _build_status_rows(wells)

    bucket, key = _parse_s3_uri(output_uri)
    logger.info("Writing well status parquet to s3://%s/%s (%s rows)", bucket, key, len(status_df))
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    status_df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

    status_counts = {"RUNNING": 0, "IDLE": 0, "SHUTDOWN": 0}
    if "status" in status_df.columns:
        for s in status_counts.keys():
            status_counts[s] = int((status_df["status"] == s).sum())

    logger.info("=" * 50)
    logger.info("Well status update complete")
    logger.info(f"Summary: RUNNING={status_counts['RUNNING']}, IDLE={status_counts['IDLE']}, SHUTDOWN={status_counts['SHUTDOWN']}")
    logger.info("=" * 50)
    
    return status_counts


def get_status_for_reading(well_id: str, lift_type: str, readings: Dict) -> str:
    """
    Get status for a new reading being inserted.
    This is called by real-time ingest paths when inserting a reading.
    
    Since we're inserting a reading, well is active (not SHUTDOWN).
    Determine RUNNING vs IDLE based on multiple production metrics.
    
    Uses the same multi-metric scoring as the background service.
    """
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
