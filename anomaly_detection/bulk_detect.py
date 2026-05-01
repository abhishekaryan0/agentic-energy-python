import io
import logging
import os
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from anomaly_detection.detector import ML_ONLY_ANOMALY_THRESHOLD, get_detector
from anomaly_detection.llm_suggestions import generate_suggestion, persist_suggestion_to_s3, calculate_severity
from anomaly_detection.s3_history import write_anomaly_violations_history_to_s3, write_well_anomalies_history_to_s3

logger = logging.getLogger(__name__)


def _get_bucket_and_org() -> Tuple[str, str]:
    bucket = (os.getenv("S3_BUCKET") or "").strip()
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        raise RuntimeError("S3_BUCKET and ORGANIZATION_ID must be set")
    return bucket, org


def _read_sensor_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    import boto3

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _checkpoint_key(org: str) -> str:
    override = (os.getenv("ANOMALY_DETECTION_CHECKPOINT_S3_KEY") or "").strip()
    if override:
        return override.lstrip("/")
    return f"{org}/detections/anomaly_detection/checkpoints/last_processed.json"


def _load_last_processed_ts(bucket: str, key: str) -> Optional[pd.Timestamp]:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    try:
        import json

        payload = json.loads(raw.decode("utf-8"))
        ts = payload.get("last_processed_ts")
        if not ts:
            return None
        return pd.to_datetime(ts, utc=True, errors="coerce")
    except Exception:
        return None


def _save_last_processed_ts(bucket: str, key: str, ts: pd.Timestamp) -> None:
    import boto3
    import json

    s3 = boto3.client("s3")
    payload = {
        "last_processed_ts": pd.to_datetime(ts, utc=True).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode("utf-8"))


def _extract_reading_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    ignore = {"organization_id", "well_id", "timestamp", "lift_type", "well_status"}
    readings: Dict[str, Any] = {}
    for k, v in row.items():
        if k in ignore:
            continue
        if pd.isna(v):
            readings[k] = None
            continue
        try:
            readings[k] = float(v) if v is not None else None
        except Exception:
            readings[k] = None
    return readings


def run_incremental_anomaly_detection(
    *,
    window_minutes: int = 15,
    safety_lag_seconds: int = 90,
    max_wells: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Incremental anomaly detection runner (S3 only):
    - Reads s3://S3_BUCKET/ORGANIZATION_ID/sensor_data/well_sensor_readings.parquet
    - Filters to NEW rows since last checkpoint watermark and within the last `window_minutes`
    - For each well, runs detection on the newest row in the window using the prior (last_n-1) rows as history
    - Writes anomalies/violations/suggestions to S3 (same as bulk runner)
    - Updates S3 checkpoint so the same timestamps are not processed again.
    """
    if window_minutes < 1:
        raise ValueError("window_minutes must be >= 1")
    bucket, org = _get_bucket_and_org()
    sensor_key = f"{org}/sensor_data/well_sensor_readings.parquet"
    ck_key = _checkpoint_key(org)

    now_utc = datetime.now(timezone.utc)
    window_end = pd.Timestamp(now_utc - timedelta(seconds=max(0, int(safety_lag_seconds))), tz="UTC")
    window_floor = pd.Timestamp(now_utc - timedelta(minutes=int(window_minutes)), tz="UTC")

    last_processed = _load_last_processed_ts(bucket, ck_key)
    window_start = max(window_floor, last_processed) if last_processed is not None else window_floor

    logger.info(
        "Incremental anomaly detection window: (%s, %s] (checkpoint=%s)",
        window_start.isoformat(),
        window_end.isoformat(),
        ck_key,
    )

    df = _read_sensor_parquet_from_s3(bucket, sensor_key)
    if df.empty:
        return {"success": True, "message": "No sensor readings found", "wells_processed": 0, "anomalies": 0}

    df.columns = [str(c) for c in df.columns]
    if "timestamp" not in df.columns or "well_id" not in df.columns or "lift_type" not in df.columns:
        raise RuntimeError("Sensor parquet missing required columns: well_id, timestamp, lift_type")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "well_id", "lift_type"])

    # Only process NEW rows: (window_start, window_end]
    df = df[(df["timestamp"] > window_start) & (df["timestamp"] <= window_end)].copy()
    if df.empty:
        # Still advance checkpoint to window_end to avoid re-scanning the same empty interval forever.
        _save_last_processed_ts(bucket, ck_key, window_end)
        return {
            "success": True,
            "message": "No new rows in window; checkpoint advanced",
            "wells_processed": 0,
            "anomalies": 0,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
        }

    detector = get_detector()

    enable_llm = str(os.getenv("ANOMALY_ENABLE_LLM_SUGGESTIONS", "0")).strip().lower() in ("1", "true", "yes", "y")
    last_n = int(os.getenv("ANOMALY_INCREMENTAL_LAST_N", "15"))
    if last_n < 1:
        last_n = 1

    wells = list(df["well_id"].astype(str).unique())
    wells.sort()
    if max_wells is not None:
        wells = wells[: max(0, int(max_wells))]

    anomalies_written = 0
    wells_processed = 0
    details: List[Dict[str, Any]] = []

    # To build history, we need access to prior rows for the well (outside the window too).
    # We re-read from the full parquet but only keep last_n rows per well (cheap-ish).
    full_df = None
    try:
        full_df = _read_sensor_parquet_from_s3(bucket, sensor_key)
        full_df.columns = [str(c) for c in full_df.columns]
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce", utc=True)
        full_df = full_df.dropna(subset=["timestamp", "well_id", "lift_type"])
    except Exception:
        full_df = None

    max_seen_ts = None

    for well_id in wells:
        wdf = df[df["well_id"].astype(str) == well_id].sort_values("timestamp")
        if wdf.empty:
            continue
        current_row = wdf.iloc[-1].to_dict()
        current_ts: pd.Timestamp = current_row["timestamp"]
        if max_seen_ts is None or current_ts > max_seen_ts:
            max_seen_ts = current_ts

        lift_type = str(current_row.get("lift_type") or "").strip()
        if not lift_type:
            continue

        # Historical context: last_n-1 rows strictly before current_ts.
        hist_rows: List[Dict[str, Any]] = []
        if full_df is not None:
            hw = full_df[full_df["well_id"].astype(str) == well_id].copy()
            hw = hw[hw["timestamp"] < current_ts].sort_values("timestamp").tail(max(0, last_n - 1))
            if not hw.empty:
                hist_rows = hw.to_dict(orient="records")

        current_readings = _extract_reading_dict(current_row)
        historical_data = [_extract_reading_dict(r) for r in hist_rows]

        timestamp_str = pd.to_datetime(current_ts, utc=True).isoformat()
        well_status = current_row.get("well_status")

        anomaly_details = detector.check_anomaly(
            current_readings,
            historical_data=historical_data,
            well_id=str(well_id),
            lift_type=lift_type,
        )

        # Guard: ML-only score should be high enough when there are no rule violations
        try:
            if (not anomaly_details.get("violations")) and float(anomaly_details.get("anomaly_score", 0)) < float(
                ML_ONLY_ANOMALY_THRESHOLD
            ):
                anomaly_details["is_anomaly"] = False
        except Exception:
            pass

        wells_processed += 1

        if not anomaly_details.get("is_anomaly"):
            details.append(
                {
                    "well_id": str(well_id),
                    "timestamp": timestamp_str,
                    "lift_type": lift_type,
                    "anomaly": False,
                    "score": anomaly_details.get("anomaly_score", 0),
                }
            )
            continue

        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get("anomaly_score", 0), current_readings)
        violation_summary = "; ".join([v.get("violation", "") for v in (anomaly_details.get("violations") or [])]).strip()

        anomaly_id = f"{well_id}-{timestamp_str}"
        anomaly_record = {
            "id": anomaly_id,
            "organization_id": org,
            "well_id": str(well_id),
            "timestamp": timestamp_str,
            "anomaly_type": violation_summary,
            "anomaly_score": float(anomaly_details.get("anomaly_score", 0) or 0),
            "raw_values": current_readings,
            "model_name": "Rule_Based_Frontend",
            "status": "New",
            "open_up": None,
            "mute_till": None,
            "severity": calculated_severity,
            "category": lift_type,
            "user_id": None,
            "user_name": None,
            "updated_at": None,
        }

        violations_records: List[Dict[str, Any]] = []
        for v in anomaly_details.get("violations", []) or []:
            dev_pct = v.get("deviation_pct", 0.0)
            e_min = v.get("min") if v.get("min") is not None else 0.0
            e_max = v.get("max") if v.get("max") is not None else 0.0
            violations_records.append(
                {
                    "anomaly_id": anomaly_id,
                    "well_id": str(well_id),
                    "timestamp": timestamp_str,
                    "parameter_name": v.get("field"),
                    "actual_value": v.get("value"),
                    "expected_min": e_min,
                    "expected_max": e_max,
                    "unit": v.get("unit", ""),
                    "deviation_percent": dev_pct,
                    "user_id": None,
                    "user_name": None,
                    "updated_at": None,
                }
            )

        write_well_anomalies_history_to_s3([anomaly_record])
        if violations_records:
            write_anomaly_violations_history_to_s3(violations_records)

        if enable_llm:
            try:
                suggestion_result = generate_suggestion(
                    anomaly_details=anomaly_details,
                    well_id=str(well_id),
                    readings=current_readings,
                    timestamp_str=timestamp_str,
                    historical_anomalies=None,
                    lift_type=lift_type,
                )
                persist_suggestion_to_s3(
                    anomaly_details=anomaly_details,
                    suggestion_result=suggestion_result,
                    well_id=str(well_id),
                    timestamp_str=timestamp_str,
                    lift_type=lift_type,
                    well_status=str(well_status) if well_status is not None else None,
                    organization_id=org,
                    readings=current_readings,
                )
            except Exception as e:
                logger.warning("Failed to generate/persist LLM suggestion: %s", e)

        anomalies_written += 1
        details.append(
            {
                "well_id": str(well_id),
                "timestamp": timestamp_str,
                "lift_type": lift_type,
                "anomaly": True,
                "score": anomaly_details.get("anomaly_score", 0),
                "severity": calculated_severity,
            }
        )

    # Advance checkpoint to the max timestamp we processed (or window_end as fallback).
    advance_to = max_seen_ts if max_seen_ts is not None else window_end
    _save_last_processed_ts(bucket, ck_key, advance_to)

    return {
        "success": True,
        "message": "Incremental anomaly detection complete",
        "wells_processed": wells_processed,
        "anomalies": anomalies_written,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "checkpoint_s3_key": ck_key,
        "checkpoint_advanced_to": pd.to_datetime(advance_to, utc=True).isoformat(),
        "details": details,
    }


def run_bulk_anomaly_detection(
    *,
    last_n: int = 15,
    per_well: bool = True,
    max_wells: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bulk anomaly detection runner:
    - Reads s3://S3_BUCKET/ORGANIZATION_ID/sensor_data/well_sensor_readings.parquet
    - For each well (default), takes latest `last_n` rows by timestamp
    - Runs anomaly detection on the newest row using previous rows as historical_data
    - If anomaly detected, writes:
        - detections/anomaly_detection/history/well_anomalies_*.parquet
        - detections/anomaly_detection/history/anomaly_violations_*.parquet
        - detections/anomaly_detection/anomaly_suggestions_*.parquet (OpenAI)
    """
    if last_n < 1:
        raise ValueError("last_n must be >= 1")

    bucket, org = _get_bucket_and_org()
    sensor_key = f"{org}/sensor_data/well_sensor_readings.parquet"

    logger.info("Loading sensor parquet from s3://%s/%s", bucket, sensor_key)
    df = _read_sensor_parquet_from_s3(bucket, sensor_key)
    if df.empty:
        return {"success": True, "message": "No sensor readings found", "wells_processed": 0, "anomalies": 0}

    # Normalize columns (some writers may create mixed-case)
    df.columns = [str(c) for c in df.columns]
    if "timestamp" not in df.columns or "well_id" not in df.columns or "lift_type" not in df.columns:
        raise RuntimeError("Sensor parquet missing required columns: well_id, timestamp, lift_type")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "well_id", "lift_type"])

    detector = get_detector()

    anomalies_written = 0
    wells_processed = 0
    details: List[Dict[str, Any]] = []

    if per_well:
        wells = list(df["well_id"].astype(str).unique())
        wells.sort()
        if max_wells is not None:
            wells = wells[: max(0, int(max_wells))]
        groups = [(w, df[df["well_id"].astype(str) == w]) for w in wells]
    else:
        groups = [("__ALL__", df)]

    for well_id, wdf in groups:
        wdf = wdf.sort_values("timestamp")
        tail = wdf.tail(last_n)
        if tail.empty:
            continue

        rows = tail.to_dict(orient="records")
        current = rows[-1]
        historical_rows = rows[:-1]

        lift_type = str(current.get("lift_type") or "").strip()
        if not lift_type:
            continue

        current_readings = _extract_reading_dict(current)
        historical_data = [_extract_reading_dict(r) for r in historical_rows]

        ts: datetime = current["timestamp"]
        timestamp_str = ts.isoformat()
        well_status = current.get("well_status")

        anomaly_details = detector.check_anomaly(
            current_readings,
            historical_data=historical_data,
            well_id=str(well_id),
            lift_type=lift_type,
        )

        # Guard: ML-only score should be high enough when there are no rule violations
        try:
            if (not anomaly_details.get("violations")) and float(anomaly_details.get("anomaly_score", 0)) < float(
                ML_ONLY_ANOMALY_THRESHOLD
            ):
                anomaly_details["is_anomaly"] = False
        except Exception:
            pass

        wells_processed += 1

        if not anomaly_details.get("is_anomaly"):
            details.append(
                {
                    "well_id": str(well_id),
                    "timestamp": timestamp_str,
                    "lift_type": lift_type,
                    "anomaly": False,
                    "score": anomaly_details.get("anomaly_score", 0),
                }
            )
            continue

        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get("anomaly_score", 0), current_readings)
        violation_summary = "; ".join([v.get("violation", "") for v in (anomaly_details.get("violations") or [])]).strip()

        anomaly_id = f"{well_id}-{timestamp_str}"
        anomaly_record = {
            "id": anomaly_id,
            "organization_id": org,
            "well_id": str(well_id),
            "timestamp": timestamp_str,
            "anomaly_type": violation_summary,
            "anomaly_score": float(anomaly_details.get("anomaly_score", 0) or 0),
            "raw_values": current_readings,
            "model_name": "Rule_Based_Frontend",
            "status": "New",
            "open_up": None,
            "mute_till": None,
            "severity": calculated_severity,
            "category": lift_type,
            "user_id": None,
            "user_name": None,
            "updated_at": None,
        }

        violations_records: List[Dict[str, Any]] = []
        for v in anomaly_details.get("violations", []) or []:
            dev_pct = v.get("deviation_pct", 0.0)
            e_min = v.get("min") if v.get("min") is not None else 0.0
            e_max = v.get("max") if v.get("max") is not None else 0.0
            violations_records.append(
                {
                    "anomaly_id": anomaly_id,
                    "well_id": str(well_id),
                    "timestamp": timestamp_str,
                    "parameter_name": v.get("field"),
                    "actual_value": v.get("value"),
                    "expected_min": e_min,
                    "expected_max": e_max,
                    "unit": v.get("unit", ""),
                    "deviation_percent": dev_pct,
                    "user_id": None,
                    "user_name": None,
                    "updated_at": None,
                }
            )

        write_well_anomalies_history_to_s3([anomaly_record])
        if violations_records:
            write_anomaly_violations_history_to_s3(violations_records)

        suggestion_result = generate_suggestion(
            anomaly_details=anomaly_details,
            well_id=str(well_id),
            readings=current_readings,
            timestamp_str=timestamp_str,
            historical_anomalies=None,
            lift_type=lift_type,
        )
        try:
            persist_suggestion_to_s3(
                anomaly_details=anomaly_details,
                suggestion_result=suggestion_result,
                well_id=str(well_id),
                timestamp_str=timestamp_str,
                lift_type=lift_type,
                well_status=str(well_status) if well_status is not None else None,
                organization_id=org,
                readings=current_readings,
            )
        except Exception as e:
            logger.warning("Failed to persist suggestion to S3: %s", e)

        anomalies_written += 1
        details.append(
            {
                "well_id": str(well_id),
                "timestamp": timestamp_str,
                "lift_type": lift_type,
                "anomaly": True,
                "score": anomaly_details.get("anomaly_score", 0),
                "severity": calculated_severity,
            }
        )

    return {
        "success": True,
        "message": "Bulk anomaly detection complete",
        "wells_processed": wells_processed,
        "anomalies": anomalies_written,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "last_n": last_n,
        "details": details,
    }

