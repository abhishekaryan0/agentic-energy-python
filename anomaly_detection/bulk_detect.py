import io
import logging
import os
from datetime import datetime, timezone
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

