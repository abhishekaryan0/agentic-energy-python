import logging
import os
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from anomaly_detection.detection_filters import should_skip_detection_row
from anomaly_detection.detector import ML_ONLY_ANOMALY_THRESHOLD, get_detector
from anomaly_detection.llm_suggestions import generate_suggestion, persist_suggestion_to_s3, calculate_severity
from anomaly_detection.s3_history import (
    load_recent_anomaly_suggestions_for_well,
    normalize_violation_record,
    round_readings_for_storage,
    write_anomaly_violations_history_to_s3,
    write_well_anomalies_history_to_s3,
)

logger = logging.getLogger(__name__)

_historical_suggestions_cache: Dict[str, List[Dict[str, Any]]] = {}


def _get_bucket_and_org() -> Tuple[str, str]:
    bucket = (os.getenv("S3_BUCKET") or "").strip()
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        raise RuntimeError("S3_BUCKET and ORGANIZATION_ID must be set")
    return bucket, org


def _load_sensor_dataframe(bucket: str, org: str) -> pd.DataFrame:
    """Load sensor readings (bulk / legacy path): recent N partition days."""
    from anomaly_detection.sensor_data import load_well_sensor_readings_from_s3

    raw = os.getenv("ANOMALY_SENSOR_PARTITION_DAYS", "30").strip()
    max_days = int(raw) if raw.isdigit() else 30
    return load_well_sensor_readings_from_s3(
        bucket, org, max_partition_days=max_days, prefer_partitions=True
    )


def _force_window_enabled() -> bool:
    return (os.getenv("ANOMALY_DETECTION_FORCE_WINDOW") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _force_window_days() -> int:
    raw = (os.getenv("ANOMALY_DETECTION_FORCE_WINDOW_DAYS") or "30").strip()
    return max(1, int(raw)) if raw.isdigit() else 30


def _bootstrap_partition_days() -> int:
    raw = (
        os.getenv("ANOMALY_BOOTSTRAP_PARTITION_DAYS")
        or os.getenv("ANOMALY_SENSOR_PARTITION_DAYS")
        or "30"
    ).strip()
    return max(1, int(raw)) if raw.isdigit() else 30


def _history_partition_days() -> int:
    """Extra calendar days of partitions loaded before work window (detector history)."""
    raw = (os.getenv("ANOMALY_HISTORY_PARTITION_DAYS") or "3").strip()
    return max(0, int(raw)) if raw.isdigit() else 3


def _load_sensor_for_detection_window(
    bucket: str,
    org: str,
    *,
    window_end: pd.Timestamp,
    last_processed: Optional[pd.Timestamp],
    phase: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load partitions covering the detection window (not a fixed last-N-days cap).

    Incremental: from last_processed date (minus history buffer) through window_end date.
    Bootstrap: last ANOMALY_BOOTSTRAP_PARTITION_DAYS through window_end date.
    """
    from anomaly_detection.sensor_data import load_well_sensor_readings_from_s3_date_range

    end_date = pd.to_datetime(window_end, utc=True).date()
    meta: Dict[str, Any] = {"phase": phase, "partition_end_date": end_date.isoformat()}

    if phase == "bootstrap":
        span_days = _bootstrap_partition_days()
        start_date = end_date - timedelta(days=span_days - 1)
        meta["bootstrap_partition_days"] = span_days
    else:
        if last_processed is None:
            raise RuntimeError("incremental load requires last_processed")
        lp = pd.to_datetime(last_processed, utc=True)
        start_date = lp.date()
        history_days = _history_partition_days()
        if history_days > 0:
            start_date = start_date - timedelta(days=history_days)
        meta["last_processed_data_ts"] = lp.isoformat()
        meta["history_partition_days"] = history_days

    if _force_window_enabled():
        force_start = end_date - timedelta(days=_force_window_days() - 1)
        if force_start < start_date:
            start_date = force_start
        meta["force_window"] = True
        meta["force_window_days"] = _force_window_days()

    meta["partition_start_date"] = start_date.isoformat()

    full_df, keys_loaded = load_well_sensor_readings_from_s3_date_range(
        bucket,
        org,
        start_date=start_date,
        end_date=end_date,
    )
    meta["partitions_loaded"] = len(keys_loaded)
    meta["partition_keys_sample"] = keys_loaded[:5]
    if len(keys_loaded) > 5:
        meta["partition_keys_omitted"] = len(keys_loaded) - 5

    logger.info(
        "Sensor load complete org=%s phase=%s partitions=%s date_range=[%s,%s] rows=%s",
        org,
        phase,
        len(keys_loaded),
        start_date,
        end_date,
        len(full_df),
    )
    return full_df, meta


def _checkpoint_key(org: str) -> str:
    override = (os.getenv("ANOMALY_DETECTION_CHECKPOINT_S3_KEY") or "").strip()
    if override:
        return override.lstrip("/")
    return f"{org}/detections/anomaly_detection/checkpoints/last_processed.json"


def _legacy_checkpoint_key(canonical_key: str) -> Optional[str]:
    """Older deployments wrote under ``_checkpoints/`` instead of ``checkpoints/``."""
    legacy = canonical_key.replace("/checkpoints/", "/_checkpoints/", 1)
    if legacy == canonical_key:
        return None
    return legacy


def _checkpoint_lookup_keys(canonical_key: str) -> List[str]:
    legacy = _legacy_checkpoint_key(canonical_key)
    keys = [canonical_key]
    if legacy:
        keys.append(legacy)
    return keys


def _checkpoint_payload(
    *,
    last_checked_ts: Optional[pd.Timestamp],
    last_processed_data_ts: Optional[pd.Timestamp],
) -> Dict[str, str]:
    payload: Dict[str, str] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_checked_ts is not None:
        payload["last_checked_ts"] = pd.to_datetime(last_checked_ts, utc=True).isoformat()
    if last_processed_data_ts is not None:
        payload["last_processed_data_ts"] = pd.to_datetime(
            last_processed_data_ts, utc=True
        ).isoformat()
    return payload


def _coerce_checkpoint_ts(raw) -> Optional[pd.Timestamp]:
    if raw is None or raw == "":
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return ts


def _parse_checkpoint_payload(
    raw: bytes, uri: str
) -> Dict[str, Optional[pd.Timestamp]]:
    import json

    payload = json.loads(raw.decode("utf-8"))
    last_checked_raw = payload.get("last_checked_ts")
    last_processed_raw = payload.get("last_processed_data_ts")

    # Backward compatibility with the old single-field checkpoint payload.
    legacy_field = payload.get("last_processed_ts")
    used_legacy_field = False
    if last_checked_raw is None and legacy_field is not None:
        last_checked_raw = legacy_field
        used_legacy_field = True
    if last_processed_raw is None and legacy_field is not None:
        last_processed_raw = legacy_field
        used_legacy_field = True

    last_checked_ts = _coerce_checkpoint_ts(last_checked_raw)
    last_processed_data_ts = _coerce_checkpoint_ts(last_processed_raw)

    if last_processed_data_ts is None:
        logger.warning(
            "Checkpoint at %s has no usable last_processed_data_ts (keys=%s) — bootstrap",
            uri,
            sorted(payload.keys()),
        )
    else:
        logger.info(
            "Checkpoint loaded from %s last_processed_data_ts=%s last_checked_ts=%s legacy_field=%s",
            uri,
            last_processed_data_ts.isoformat(),
            last_checked_ts.isoformat() if last_checked_ts is not None else "none",
            used_legacy_field,
        )
    return {
        "last_checked_ts": last_checked_ts,
        "last_processed_data_ts": last_processed_data_ts,
    }


def _load_checkpoint(bucket: str, key: str) -> Dict[str, Optional[pd.Timestamp]]:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    not_found: List[str] = []
    for try_key in _checkpoint_lookup_keys(key):
        uri = f"s3://{bucket}/{try_key}"
        try:
            obj = s3.get_object(Bucket=bucket, Key=try_key)
            raw = obj["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                not_found.append(uri)
                continue
            raise
        try:
            parsed = _parse_checkpoint_payload(raw, uri)
        except Exception as e:
            logger.warning("Failed to parse checkpoint at %s: %s", uri, e)
            continue

        if try_key != key and parsed.get("last_processed_data_ts") is not None:
            logger.info(
                "Migrating legacy checkpoint %s -> s3://%s/%s",
                uri,
                bucket,
                key,
            )
            _save_checkpoint(
                bucket,
                key,
                last_checked_ts=parsed.get("last_checked_ts"),
                last_processed_data_ts=parsed.get("last_processed_data_ts"),
            )
        return parsed

    logger.info(
        "Checkpoint not found (%s) — will run bootstrap",
        ", ".join(not_found) if not_found else f"s3://{bucket}/{key}",
    )
    return {"last_checked_ts": None, "last_processed_data_ts": None}


def _save_checkpoint(
    bucket: str,
    key: str,
    *,
    last_checked_ts: Optional[pd.Timestamp],
    last_processed_data_ts: Optional[pd.Timestamp],
) -> None:
    import boto3
    import json

    s3 = boto3.client("s3")
    payload = _checkpoint_payload(
        last_checked_ts=last_checked_ts,
        last_processed_data_ts=last_processed_data_ts,
    )
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
    return round_readings_for_storage(readings)


def _llm_suggestions_enabled() -> bool:
    explicit = (os.getenv("ANOMALY_ENABLE_LLM_SUGGESTIONS") or "").strip().lower()
    if explicit in ("0", "false", "no", "n"):
        return False
    if explicit in ("1", "true", "yes", "y"):
        return True
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _historical_anomalies_for_well(well_id: str, organization_id: str) -> List[Dict[str, Any]]:
    cache_key = f"{organization_id}:{well_id}"
    if cache_key not in _historical_suggestions_cache:
        try:
            _historical_suggestions_cache[cache_key] = load_recent_anomaly_suggestions_for_well(
                well_id=str(well_id),
                organization_id=organization_id,
                limit=10,
            )
        except Exception as e:
            logger.warning("Could not load historical suggestions for well=%s: %s", well_id, e)
            _historical_suggestions_cache[cache_key] = []
    return _historical_suggestions_cache[cache_key]


def _maybe_persist_llm_suggestion(
    *,
    anomaly_details: Dict[str, Any],
    well_id: str,
    current_readings: Dict[str, Any],
    timestamp_str: str,
    lift_type: str,
    well_status: Any,
    organization_id: str,
) -> Dict[str, Any]:
    """Generate and persist LLM suggestion when enabled. Always call on flagged anomalies."""
    if not _llm_suggestions_enabled():
        return {"llm_suggestion": False, "llm_skipped_reason": "disabled"}
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        logger.warning("ANOMALY_ENABLE_LLM_SUGGESTIONS is on but OPENAI_API_KEY is missing")
        return {"llm_suggestion": False, "llm_skipped_reason": "missing_openai_api_key"}
    try:
        suggestion_result = generate_suggestion(
            anomaly_details=anomaly_details,
            well_id=str(well_id),
            readings=current_readings,
            timestamp_str=timestamp_str,
            historical_anomalies=_historical_anomalies_for_well(str(well_id), organization_id),
            lift_type=lift_type,
        )
        persist_suggestion_to_s3(
            anomaly_details=anomaly_details,
            suggestion_result=suggestion_result,
            well_id=str(well_id),
            timestamp_str=timestamp_str,
            lift_type=lift_type,
            well_status=str(well_status) if well_status is not None else None,
            organization_id=organization_id,
            readings=current_readings,
        )
        return {
            "llm_suggestion": True,
            "alert_title": suggestion_result.get("alert_title"),
            "severity": suggestion_result.get("severity"),
            "confidence": suggestion_result.get("confidence"),
            "description": suggestion_result.get("description"),
        }
    except Exception as e:
        logger.warning(
            "Failed to generate/persist LLM suggestion well=%s ts=%s: %s",
            well_id,
            timestamp_str,
            e,
        )
        return {"llm_suggestion": False, "llm_error": str(e)}


def _normalize_sensor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    if "timestamp" not in df.columns or "well_id" not in df.columns or "lift_type" not in df.columns:
        raise RuntimeError("Sensor parquet missing required columns: well_id, timestamp, lift_type")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp", "well_id", "lift_type"])


def _history_last_n() -> int:
    seq_len = max(int(os.getenv("ANOMALY_LSTM_SEQ_LEN", "24") or 24), 24)
    default_last_n = max(seq_len * 2, 48)
    raw = os.getenv("ANOMALY_INCREMENTAL_LAST_N", str(default_last_n)).strip()
    if raw.isdigit():
        return max(seq_len, int(raw))
    return default_last_n


def _bootstrap_row_limit() -> int:
    """
    Max rows on first run (no checkpoint). 0 = process every row in the bootstrap partition window.
    """
    raw = (os.getenv("ANOMALY_BOOTSTRAP_ROWS") or os.getenv("ANOMALY_LAST_ROWS") or "100").strip()
    if not raw.isdigit():
        return 0
    return max(0, int(raw))


def _persist_anomaly_if_detected(
    *,
    detector,
    org: str,
    well_id: str,
    current_row: Dict[str, Any],
    full_df: Optional[pd.DataFrame],
    last_n: int,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Run detection for one row. Returns (detail, wrote_anomaly, wrote_llm_suggestion)."""
    current_ts: pd.Timestamp = current_row["timestamp"]
    lift_type = str(current_row.get("lift_type") or "").strip()
    if not lift_type:
        return {"well_id": str(well_id), "skipped": True, "reason": "missing lift_type"}, False, False

    skip_reason = should_skip_detection_row(current_row, lift_type)
    if skip_reason:
        return {"well_id": str(well_id), "skipped": True, "reason": skip_reason}, False, False

    hist_rows: List[Dict[str, Any]] = []
    if full_df is not None:
        hw = full_df[full_df["well_id"].astype(str) == str(well_id)].copy()
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

    try:
        if (not anomaly_details.get("violations")) and float(anomaly_details.get("anomaly_score", 0)) < float(
            ML_ONLY_ANOMALY_THRESHOLD
        ):
            anomaly_details["is_anomaly"] = False
    except Exception:
        pass

    detail = {
        "well_id": str(well_id),
        "timestamp": timestamp_str,
        "lift_type": lift_type,
        "anomaly": bool(anomaly_details.get("is_anomaly")),
        "score": anomaly_details.get("anomaly_score", 0),
    }

    if not anomaly_details.get("is_anomaly"):
        return detail, False, False

    calculated_severity = calculate_severity(
        anomaly_details, anomaly_details.get("anomaly_score", 0), current_readings
    )
    violation_summary = "; ".join(
        [v.get("violation", "") for v in (anomaly_details.get("violations") or [])]
    ).strip()

    anomaly_id = f"{well_id}-{timestamp_str}"
    anomaly_record = {
        "id": anomaly_id,
        "organization_id": org,
        "well_id": str(well_id),
        "timestamp": timestamp_str,
        "anomaly_type": violation_summary,
        "anomaly_score": float(anomaly_details.get("anomaly_score", 0) or 0),
        "raw_values": current_readings,
        "model_name": "anomaly_detection_ensemble",
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
        norm = normalize_violation_record(v)
        violations_records.append(
            {
                "anomaly_id": anomaly_id,
                "well_id": str(well_id),
                "timestamp": timestamp_str,
                "parameter_name": norm["field"],
                "actual_value": norm["observed"] if norm["observed"] is not None else 0.0,
                "expected_min": norm["expected_min"] if norm["expected_min"] is not None else 0.0,
                "expected_max": norm["expected_max"] if norm["expected_max"] is not None else 0.0,
                "unit": norm["unit"],
                "deviation_percent": norm["deviation_percent"] if norm["deviation_percent"] is not None else 0.0,
                "user_id": None,
                "user_name": None,
                "updated_at": None,
            }
        )

    write_well_anomalies_history_to_s3([anomaly_record])
    if violations_records:
        write_anomaly_violations_history_to_s3(violations_records)

    llm_info = _maybe_persist_llm_suggestion(
        anomaly_details=anomaly_details,
        well_id=str(well_id),
        current_readings=current_readings,
        timestamp_str=timestamp_str,
        lift_type=lift_type,
        well_status=well_status,
        organization_id=org,
    )

    detail["severity"] = calculated_severity
    detail.update(llm_info)
    return detail, True, bool(llm_info.get("llm_suggestion"))


def _run_rows_detection_loop(
    df: pd.DataFrame,
    full_df: Optional[pd.DataFrame],
    org: str,
    *,
    max_wells: Optional[int] = None,
    last_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate each row in df (sorted by time); persist detections when flagged."""
    if last_n is None:
        last_n = _history_last_n()
    if last_n < 1:
        last_n = 1

    detector = get_detector()
    work = df.sort_values(["timestamp", "well_id"]).copy()
    if max_wells is not None:
        allowed = set(work["well_id"].astype(str).unique()[: max(0, int(max_wells))])
        work = work[work["well_id"].astype(str).isin(allowed)]

    anomalies_written = 0
    llm_suggestions_written = 0
    rows_processed = 0
    wells_seen: set = set()
    details: List[Dict[str, Any]] = []
    max_seen_ts: Optional[pd.Timestamp] = None

    for _idx, row in work.iterrows():
        current_row = row.to_dict()
        well_id = str(current_row.get("well_id") or "")
        if not well_id:
            continue
        current_ts = pd.to_datetime(current_row.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(current_ts):
            continue
        current_row["timestamp"] = current_ts
        if max_seen_ts is None or current_ts > max_seen_ts:
            max_seen_ts = current_ts

        detail, wrote, llm_wrote = _persist_anomaly_if_detected(
            detector=detector,
            org=org,
            well_id=well_id,
            current_row=current_row,
            full_df=full_df,
            last_n=last_n,
        )
        rows_processed += 1
        wells_seen.add(well_id)
        if wrote:
            anomalies_written += 1
        if llm_wrote:
            llm_suggestions_written += 1
        details.append(detail)

    if anomalies_written and _llm_suggestions_enabled():
        logger.info(
            "LLM suggestions phase complete: %s/%s anomalies persisted",
            llm_suggestions_written,
            anomalies_written,
        )

    return {
        "wells_processed": len(wells_seen),
        "rows_processed": rows_processed,
        "anomalies": anomalies_written,
        "llm_suggestions_written": llm_suggestions_written,
        "details": details,
        "max_seen_ts": max_seen_ts,
    }


def _run_well_detection_loop(
    df: pd.DataFrame,
    full_df: Optional[pd.DataFrame],
    org: str,
    *,
    max_wells: Optional[int] = None,
    last_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Backward-compatible: one latest row per well in df."""
    per_well = []
    for well_id in df["well_id"].astype(str).unique():
        wdf = df[df["well_id"].astype(str) == well_id].sort_values("timestamp")
        if not wdf.empty:
            per_well.append(wdf.iloc[-1:])
    if not per_well:
        return {"wells_processed": 0, "rows_processed": 0, "anomalies": 0, "details": [], "max_seen_ts": None}
    return _run_rows_detection_loop(
        pd.concat(per_well, ignore_index=True), full_df, org, max_wells=max_wells, last_n=last_n
    )


def run_recent_rows_anomaly_detection(
    *,
    last_rows: int = 1000,
    max_wells: Optional[int] = None,
) -> Dict[str, Any]:
    """Detect on the newest row per well in the last `last_rows` readings (no checkpoint)."""
    if last_rows < 1:
        raise ValueError("last_rows must be >= 1")

    bucket, org = _get_bucket_and_org()

    logger.info(
        "Loading sensor data for org %s (last %s rows, partition days=%s)",
        org,
        last_rows,
        os.getenv("ANOMALY_SENSOR_PARTITION_DAYS", "30"),
    )
    full_df = _normalize_sensor_dataframe(_load_sensor_dataframe(bucket, org))
    if full_df.empty:
        return {"success": True, "message": "No sensor readings found", "wells_processed": 0, "anomalies": 0}

    slice_df = full_df.sort_values("timestamp").tail(int(last_rows))
    outcome = _run_rows_detection_loop(slice_df, full_df, org, max_wells=max_wells)

    return {
        "success": True,
        "message": f"Recent-rows anomaly detection complete (last {last_rows} readings)",
        "llm_suggestions_enabled": _llm_suggestions_enabled(),
        "wells_processed": outcome["wells_processed"],
        "anomalies": outcome["anomalies"],
        "llm_suggestions_written": outcome.get("llm_suggestions_written", 0),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "last_rows": int(last_rows),
        "slice_rows": len(slice_df),
        "slice_start": slice_df["timestamp"].min().isoformat(),
        "slice_end": slice_df["timestamp"].max().isoformat(),
        "details": outcome["details"],
    }


def run_incremental_anomaly_detection(
    *,
    window_minutes: int = 15,
    safety_lag_seconds: int = 90,
    max_wells: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scheduled anomaly detection (S3 + trained models):

    **First run (no checkpoint):** loads bootstrap partition window and evaluates
    all rows (or the last `ANOMALY_BOOTSTRAP_ROWS` when set > 0).

    **Later runs:** loads every partition from last_processed (minus history buffer)
    through window_end; processes all rows with timestamp > last_processed_data_ts;
    runs LLM suggestions in the same job for each new anomaly; advances checkpoint
    to the max processed sensor timestamp.
    """
    bucket, org = _get_bucket_and_org()
    ck_key = _checkpoint_key(org)
    bootstrap_rows = _bootstrap_row_limit()
    last_n = _history_last_n()

    reset = (os.getenv("ANOMALY_RESET_CHECKPOINT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    checkpoint = (
        {"last_checked_ts": None, "last_processed_data_ts": None}
        if reset
        else _load_checkpoint(bucket, ck_key)
    )
    last_checked_ts = checkpoint.get("last_checked_ts")
    last_processed = checkpoint.get("last_processed_data_ts")
    if last_processed is not None and pd.isna(last_processed):
        last_processed = None
    if last_checked_ts is not None and pd.isna(last_checked_ts):
        last_checked_ts = None

    now_utc = datetime.now(timezone.utc)
    window_end = pd.Timestamp(now_utc - timedelta(seconds=max(0, int(safety_lag_seconds))))
    phase = "bootstrap" if last_processed is None else "incremental"

    logger.info(
        "Anomaly detection starting org=%s phase=%s checkpoint_key=%s "
        "last_processed_data_ts=%s last_checked_ts=%s window_end=%s "
        "bootstrap_rows_cap=%s history_last_n=%s",
        org,
        phase,
        ck_key,
        last_processed.isoformat() if last_processed is not None else "none",
        last_checked_ts.isoformat() if last_checked_ts is not None else "none",
        window_end.isoformat(),
        bootstrap_rows if bootstrap_rows > 0 else "all",
        last_n,
    )

    raw_df, load_meta = _load_sensor_for_detection_window(
        bucket,
        org,
        window_end=window_end,
        last_processed=last_processed,
        phase=phase,
    )
    full_df = _normalize_sensor_dataframe(raw_df)
    if full_df.empty:
        logger.info("No sensor readings in partition window; saving last_checked_ts only")
        retained_processed_ts = (
            pd.to_datetime(last_processed, utc=True) if last_processed is not None else None
        )
        _save_checkpoint(
            bucket,
            ck_key,
            last_checked_ts=window_end,
            last_processed_data_ts=retained_processed_ts,
        )
        return {
            "success": True,
            "message": "No sensor readings found in partition window",
            "phase": phase,
            "wells_processed": 0,
            "rows_processed": 0,
            "anomalies": 0,
            "load_meta": load_meta,
            "checkpoint_s3_key": ck_key,
            "checkpoint_last_checked_ts": window_end.isoformat(),
            "checkpoint_last_processed_data_ts": (
                retained_processed_ts.isoformat() if retained_processed_ts is not None else None
            ),
        }

    if last_processed is None:
        candidate = full_df[full_df["timestamp"] <= window_end].sort_values("timestamp")
        if bootstrap_rows > 0:
            work_df = candidate.tail(bootstrap_rows).copy()
        else:
            work_df = candidate.copy()
        window_start = work_df["timestamp"].min() if not work_df.empty else window_end
        logger.info(
            "Bootstrap detection rows=%s (cap=%s) ts_range=[%s, %s] partitions=%s",
            len(work_df),
            bootstrap_rows if bootstrap_rows > 0 else "none",
            window_start,
            work_df["timestamp"].max() if not work_df.empty else window_end,
            load_meta.get("partitions_loaded"),
        )
    else:
        effective_start = pd.to_datetime(last_processed, utc=True)
        work_df = full_df[
            (full_df["timestamp"] > effective_start) & (full_df["timestamp"] <= window_end)
        ].copy()
        window_start = effective_start
        logger.info(
            "Incremental detection rows=%s ts_range=(%s, %s] partitions=%s loaded_rows=%s",
            len(work_df),
            effective_start.isoformat(),
            window_end.isoformat(),
            load_meta.get("partitions_loaded"),
            len(full_df),
        )

    if work_df.empty:
        retained_processed_ts = (
            pd.to_datetime(last_processed, utc=True) if last_processed is not None else None
        )
        _save_checkpoint(
            bucket,
            ck_key,
            last_checked_ts=window_end,
            last_processed_data_ts=retained_processed_ts,
        )
        logger.info(
            "No new sensor rows after checkpoint; last_checked_ts=%s last_processed_data_ts=%s",
            window_end.isoformat(),
            retained_processed_ts.isoformat() if retained_processed_ts is not None else "none",
        )
        return {
            "success": True,
            "message": "No new rows after checkpoint",
            "phase": phase,
            "wells_processed": 0,
            "rows_processed": 0,
            "anomalies": 0,
            "load_meta": load_meta,
            "window_start": pd.to_datetime(window_start, utc=True).isoformat(),
            "window_end": window_end.isoformat(),
            "checkpoint_s3_key": ck_key,
            "checkpoint_advanced_to": (
                retained_processed_ts.isoformat() if retained_processed_ts is not None else None
            ),
            "checkpoint_last_checked_ts": window_end.isoformat(),
            "checkpoint_last_processed_data_ts": (
                retained_processed_ts.isoformat() if retained_processed_ts is not None else None
            ),
        }

    logger.info(
        "Running detection on %s new row(s) across %s well(s) (LLM enabled=%s)",
        len(work_df),
        work_df["well_id"].astype(str).nunique(),
        _llm_suggestions_enabled(),
    )
    outcome = _run_rows_detection_loop(
        work_df, full_df, org, max_wells=max_wells, last_n=last_n
    )

    max_seen_ts = outcome.get("max_seen_ts")
    advance_to = (
        pd.to_datetime(max_seen_ts, utc=True)
        if max_seen_ts is not None
        else pd.to_datetime(work_df["timestamp"].max(), utc=True)
    )
    _save_checkpoint(
        bucket,
        ck_key,
        last_checked_ts=window_end,
        last_processed_data_ts=advance_to,
    )
    logger.info(
        "Checkpoint saved s3://%s/%s last_checked_ts=%s last_processed_data_ts=%s "
        "(previous last_processed=%s)",
        bucket,
        ck_key,
        window_end.isoformat(),
        advance_to.isoformat(),
        last_processed.isoformat() if last_processed is not None else "none",
    )

    return {
        "success": True,
        "message": f"{phase} anomaly detection complete",
        "phase": phase,
        "llm_suggestions_enabled": _llm_suggestions_enabled(),
        "wells_processed": outcome["wells_processed"],
        "rows_processed": outcome.get("rows_processed", 0),
        "anomalies": outcome["anomalies"],
        "llm_suggestions_written": outcome.get("llm_suggestions_written", 0),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap_rows_cap": bootstrap_rows if phase == "bootstrap" else None,
        "rows_in_scope": len(work_df),
        "load_meta": load_meta,
        "window_start": pd.to_datetime(window_start, utc=True).isoformat(),
        "window_end": window_end.isoformat(),
        "checkpoint_s3_key": ck_key,
        "checkpoint_advanced_to": pd.to_datetime(advance_to, utc=True).isoformat(),
        "checkpoint_last_checked_ts": window_end.isoformat(),
        "checkpoint_last_processed_data_ts": pd.to_datetime(
            advance_to, utc=True
        ).isoformat(),
        "details": outcome["details"],
    }


def run_bulk_anomaly_detection(
    *,
    last_n: Optional[int] = None,
    per_well: bool = True,
    max_wells: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bulk anomaly detection runner:
    - Reads sensor data from s3://S3_BUCKET/ORGANIZATION_ID/sensor_data/ (date partitions or legacy parquet)
    - For each well (default), takes latest `last_n` rows by timestamp
    - Runs anomaly detection on the newest row using previous rows as historical_data
    - If anomaly detected, writes:
        - detections/anomaly_detection/history/well_anomalies_*.parquet
        - detections/anomaly_detection/history/anomaly_violations_*.parquet
        - detections/anomaly_detection/anomaly_suggestions_*.parquet (OpenAI)
    """
    if last_n is None:
        last_n = _history_last_n()
    if last_n < 1:
        raise ValueError("last_n must be >= 1")

    bucket, org = _get_bucket_and_org()

    logger.info("Loading sensor data for org %s", org)
    df = _load_sensor_dataframe(bucket, org)
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
    llm_suggestions_written = 0
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

        ts: datetime = current["timestamp"]
        timestamp_str = pd.to_datetime(ts, utc=True).isoformat()

        skip_reason = should_skip_detection_row(current, lift_type)
        if skip_reason:
            details.append(
                {
                    "well_id": str(well_id),
                    "timestamp": timestamp_str,
                    "lift_type": lift_type,
                    "skipped": True,
                    "reason": skip_reason,
                }
            )
            continue

        current_readings = _extract_reading_dict(current)
        historical_data = [_extract_reading_dict(r) for r in historical_rows]
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
            norm = normalize_violation_record(v)
            violations_records.append(
                {
                    "anomaly_id": anomaly_id,
                    "well_id": str(well_id),
                    "timestamp": timestamp_str,
                    "parameter_name": norm["field"],
                    "actual_value": norm["observed"] if norm["observed"] is not None else 0.0,
                    "expected_min": norm["expected_min"] if norm["expected_min"] is not None else 0.0,
                    "expected_max": norm["expected_max"] if norm["expected_max"] is not None else 0.0,
                    "unit": norm["unit"],
                    "deviation_percent": norm["deviation_percent"] if norm["deviation_percent"] is not None else 0.0,
                    "user_id": None,
                    "user_name": None,
                    "updated_at": None,
                }
            )

        write_well_anomalies_history_to_s3([anomaly_record])
        if violations_records:
            write_anomaly_violations_history_to_s3(violations_records)

        llm_info = _maybe_persist_llm_suggestion(
            anomaly_details=anomaly_details,
            well_id=str(well_id),
            current_readings=current_readings,
            timestamp_str=timestamp_str,
            lift_type=lift_type,
            well_status=well_status,
            organization_id=org,
        )
        if llm_info.get("llm_suggestion"):
            llm_suggestions_written += 1

        anomalies_written += 1
        details.append(
            {
                "well_id": str(well_id),
                "timestamp": timestamp_str,
                "lift_type": lift_type,
                "anomaly": True,
                "score": anomaly_details.get("anomaly_score", 0),
                "severity": calculated_severity,
                **llm_info,
            }
        )

    return {
        "success": True,
        "message": "Bulk anomaly detection complete",
        "llm_suggestions_enabled": _llm_suggestions_enabled(),
        "wells_processed": wells_processed,
        "anomalies": anomalies_written,
        "llm_suggestions_written": llm_suggestions_written,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "last_n": last_n,
        "details": details,
    }

