import io
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _get_bucket_and_org() -> Optional[tuple]:
    """
    Common helper to get (bucket, org_id) from env.
    Uses S3_BUCKET and ORGANIZATION_ID.
    """
    bucket = os.getenv("S3_BUCKET", "").strip()
    org = os.getenv("ORGANIZATION_ID", "").strip()
    if not bucket or not org:
        logger.warning("S3_BUCKET or ORGANIZATION_ID not set; cannot write S3 history parquet.")
        return None
    return bucket, org


def _put_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """
    Write a DataFrame as parquet to s3://bucket/key.
    """
    import boto3

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Wrote %s rows to s3://%s/%s", len(df), bucket, key)


def write_well_anomalies_history_to_s3(records: List[Dict[str, Any]]) -> Optional[str]:
    """
    Write anomaly history (equivalent to well_anomalies table) to S3 parquet under a dedicated history folder.

    Path:
        s3://S3_BUCKET/ORGANIZATION_ID/detections/anomaly_detection/history/well_anomalies_run_ts=YYYYMMDDTHHMMSSZ.parquet
    """
    if not records:
        return None
    bucket_org = _get_bucket_and_org()
    if not bucket_org:
        return None
    bucket, org = bucket_org

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{org}/detections/anomaly_detection/history/well_anomalies_run_ts={ts}.parquet"

    df = pd.DataFrame(records)
    _put_parquet_s3(df, bucket, key)
    return key


def write_anomaly_violations_history_to_s3(records: List[Dict[str, Any]]) -> Optional[str]:
    """
    Write anomaly violations history (equivalent to anomaly_violations_new table) to S3 parquet
    under a dedicated history folder.

    Path:
        s3://S3_BUCKET/ORGANIZATION_ID/detections/anomaly_detection/history/anomaly_violations_run_ts=YYYYMMDDTHHMMSSZ.parquet
    """
    if not records:
        return None
    bucket_org = _get_bucket_and_org()
    if not bucket_org:
        return None
    bucket, org = bucket_org

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{org}/detections/anomaly_detection/history/anomaly_violations_run_ts={ts}.parquet"

    df = pd.DataFrame(records)
    _put_parquet_s3(df, bucket, key)
    return key


# Canonical columns for well_sensor_readings.parquet (aligned with generate_digital_twin.py)
WELL_SENSOR_READINGS_COLUMNS = [
    "organization_id",
    "well_id",
    "timestamp",
    "lift_type",
    "tubing_pressure",
    "casing_pressure",
    "strokes_per_minute",
    "pump_fillage",
    "motor_current",
    "pump_intake_pressure",
    "motor_temp",
    "freq_hz",
    "injection_rate",
    "surface_stroke_length",
    "downhole_net_stroke",
    "injection_pressure",
    "well_status",
]


def append_well_sensor_reading_to_s3(record: Dict[str, Any]) -> Optional[str]:
    """
    Append a single sensor reading to the org's well_sensor_readings.parquet in S3.
    Path: s3://S3_BUCKET/ORGANIZATION_ID/sensor_data/well_sensor_readings.parquet
    If the object does not exist, creates a new parquet with one row.
    Record must include at least: organization_id, well_id, timestamp, lift_type;
    other keys are mapped to the canonical columns (missing columns set to None).
    """
    bucket_org = _get_bucket_and_org()
    if not bucket_org:
        return None
    bucket, org = bucket_org

    key = f"{org}/sensor_data/well_sensor_readings.parquet"

    row = {c: None for c in WELL_SENSOR_READINGS_COLUMNS}
    for k, v in record.items():
        if k in WELL_SENSOR_READINGS_COLUMNS:
            row[k] = v
    if row.get("timestamp") is not None and not isinstance(row["timestamp"], pd.Timestamp):
        try:
            row["timestamp"] = pd.to_datetime(row["timestamp"])
        except Exception:
            row["timestamp"] = pd.Timestamp.now(tz="UTC")

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            existing_df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                existing_df = pd.DataFrame(columns=WELL_SENSOR_READINGS_COLUMNS)
            else:
                logger.warning("Could not read existing well_sensor_readings.parquet: %s", e)
                existing_df = pd.DataFrame(columns=WELL_SENSOR_READINGS_COLUMNS)
        except Exception as e:
            logger.warning("Could not read existing well_sensor_readings.parquet: %s", e)
            existing_df = pd.DataFrame(columns=WELL_SENSOR_READINGS_COLUMNS)

        new_row_df = pd.DataFrame([row], columns=WELL_SENSOR_READINGS_COLUMNS)
        combined = pd.concat([existing_df, new_row_df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        _put_parquet_s3(combined, bucket, key)
        return key
    except Exception as e:
        logger.warning("Failed to append reading to well_sensor_readings.parquet in S3: %s", e)
        return None


def write_anomaly_suggestions_history_to_s3(records: List[Dict[str, Any]]) -> Optional[str]:
    """
    Write anomaly suggestions (OpenAI/LLM output) to S3 parquet under detections/anomaly_detection/.

    Path:
        s3://S3_BUCKET/ORGANIZATION_ID/detections/anomaly_detection/anomaly_suggestions_run_ts=YYYYMMDDTHHMMSSZ.parquet
    """
    if not records:
        return None
    bucket_org = _get_bucket_and_org()
    if not bucket_org:
        return None
    bucket, org = bucket_org

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{org}/detections/anomaly_detection/anomaly_suggestions_run_ts={ts}.parquet"

    df = pd.DataFrame(records)
    _put_parquet_s3(df, bucket, key)
    return key

