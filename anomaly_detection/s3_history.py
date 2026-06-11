import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


WELL_ANOMALIES_COLUMNS = [
    "id",
    "organization_id",
    "well_id",
    "timestamp",
    "anomaly_type",
    "anomaly_score",
    "raw_values",
    "model_name",
    "status",
    "open_up",
    "mute_till",
    "severity",
    "category",
    "user_id",
    "user_name",
    "updated_at",
]

ANOMALY_VIOLATIONS_COLUMNS = [
    "anomaly_id",
    "well_id",
    "timestamp",
    "parameter_name",
    "actual_value",
    "expected_min",
    "expected_max",
    "unit",
    "deviation_percent",
    "user_id",
    "user_name",
    "updated_at",
]

ANOMALY_SUGGESTIONS_COLUMNS = [
    "id",
    "well_id",
    "timestamp",
    "alert_title",
    "severity",
    "status",
    "open_up",
    "mute_till",
    "confidence",
    "description",
    "suggested_actions",
    "explanation",
    "created_at",
    "historical_context",
    "risk_analysis",
    "asset_id",
    "reason",
    "raw_anomaly_data",
    "lift_type",
    "category",
    "well_status",
    "organization_id",
    "user_id",
    "user_name",
    "updated_at",
]

# Fixed nested JSON shapes — must not change between runs (parquet / downstream consumers).
DEFAULT_ASSET_HISTORY: Dict[str, str] = {
    "commissioned": "Data Not Available",
    "operating_hours": "Data Not Available",
    "last_inspection": "Data Not Available",
}

DEFAULT_RISK_ANALYSIS: Dict[str, Any] = {
    "severity_breakdown": {
        "safety_risk": "",
        "production_impact": "",
        "financial_exposure": "",
    },
    "mitigation_strategies": {
        "immediate": "",
        "short_term": "",
        "long_term": "",
    },
    "decision_support": "",
}

DEFAULT_HISTORICAL_CONTEXT: Dict[str, Any] = {
    "asset_history": dict(DEFAULT_ASSET_HISTORY),
    "similar_incidents": [],
    "key_learnings": "No similar historical incidents found.",
}

ALLOWED_SEVERITIES = frozenset({"CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"})
ALLOWED_STATUSES = frozenset({"ACTIVE", "New", "RESOLVED", "MUTED"})


def _json_dumps_safe(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _coerce_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _normalize_well_anomalies_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        row = {k: r.get(k) for k in WELL_ANOMALIES_COLUMNS}
        row["anomaly_score"] = _to_float_or_none(row.get("anomaly_score"))
        row["raw_values"] = _json_dumps_safe(row.get("raw_values"))
        rows.append(row)

    df = pd.DataFrame(rows, columns=WELL_ANOMALIES_COLUMNS)
    df = _coerce_string_columns(
        df,
        [
            "id",
            "organization_id",
            "well_id",
            "timestamp",
            "anomaly_type",
            "raw_values",
            "model_name",
            "status",
            "open_up",
            "mute_till",
            "severity",
            "category",
            "user_id",
            "user_name",
            "updated_at",
        ],
    )
    return df


def _normalize_anomaly_violations_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        row = {k: r.get(k) for k in ANOMALY_VIOLATIONS_COLUMNS}
        row["actual_value"] = _to_float_or_none(row.get("actual_value"))
        row["expected_min"] = _to_float_or_none(row.get("expected_min"))
        row["expected_max"] = _to_float_or_none(row.get("expected_max"))
        row["deviation_percent"] = _to_float_or_none(row.get("deviation_percent"))
        rows.append(row)

    df = pd.DataFrame(rows, columns=ANOMALY_VIOLATIONS_COLUMNS)
    df = _coerce_string_columns(
        df,
        [
            "anomaly_id",
            "well_id",
            "timestamp",
            "parameter_name",
            "unit",
            "user_id",
            "user_name",
            "updated_at",
        ],
    )
    return df


def _to_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _normalize_severity(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip().upper()
    aliases = {
        "MODERATE": "MEDIUM",
        "MED": "MEDIUM",
        "CRIT": "CRITICAL",
    }
    text = aliases.get(text, text)
    return text if text in ALLOWED_SEVERITIES else "UNKNOWN"


def _normalize_status(value: Any) -> str:
    if value is None:
        return "ACTIVE"
    text = str(value).strip()
    upper = text.upper()
    if upper in {"ACTIVE", "NEW", "RESOLVED", "MUTED"}:
        return "ACTIVE" if upper == "NEW" else upper
    return text if text in ALLOWED_STATUSES else "ACTIVE"


def _normalize_confidence(value: Any) -> str:
    if value is None:
        return "0%"
    text = str(value).strip()
    if text.endswith("%"):
        return text
    try:
        num = float(text)
        if num <= 1.0:
            num *= 100.0
        return f"{int(round(max(0, min(99, num))))}%"
    except Exception:
        return "0%"


def _normalize_similar_incident(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "well": "",
            "date": "",
            "issue": "",
            "severity": "UNKNOWN",
            "violations": [],
        }
    violations_out: List[Dict[str, str]] = []
    for v in item.get("violations") or []:
        if not isinstance(v, dict):
            continue
        violations_out.append(
            {
                "field": _to_str_or_none(v.get("field")) or "",
                "violation": _to_str_or_none(v.get("violation")) or "",
            }
        )
    return {
        "well": _to_str_or_none(item.get("well")) or "",
        "date": _to_str_or_none(item.get("date")) or "",
        "issue": _to_str_or_none(item.get("issue")) or "",
        "severity": _normalize_severity(item.get("severity")),
        "violations": violations_out,
    }


def _normalize_historical_context(value: Any) -> Dict[str, Any]:
    ctx = dict(DEFAULT_HISTORICAL_CONTEXT)
    if not isinstance(value, dict):
        return ctx

    asset_history = value.get("asset_history") if isinstance(value.get("asset_history"), dict) else {}
    ctx["asset_history"] = {
        "commissioned": _to_str_or_none(asset_history.get("commissioned")) or DEFAULT_ASSET_HISTORY["commissioned"],
        "operating_hours": _to_str_or_none(asset_history.get("operating_hours"))
        or DEFAULT_ASSET_HISTORY["operating_hours"],
        "last_inspection": _to_str_or_none(asset_history.get("last_inspection"))
        or DEFAULT_ASSET_HISTORY["last_inspection"],
    }

    similar: List[Dict[str, Any]] = []
    for item in value.get("similar_incidents") or []:
        similar.append(_normalize_similar_incident(item))
    ctx["similar_incidents"] = similar

    key_learnings = value.get("key_learnings")
    ctx["key_learnings"] = _to_str_or_none(key_learnings) or DEFAULT_HISTORICAL_CONTEXT["key_learnings"]
    return ctx


def _normalize_risk_analysis(value: Any) -> Dict[str, Any]:
    out = json.loads(json.dumps(DEFAULT_RISK_ANALYSIS))
    if not isinstance(value, dict):
        return out

    breakdown = value.get("severity_breakdown") if isinstance(value.get("severity_breakdown"), dict) else {}
    for key in out["severity_breakdown"]:
        out["severity_breakdown"][key] = _to_str_or_none(breakdown.get(key)) or ""

    strategies = value.get("mitigation_strategies") if isinstance(value.get("mitigation_strategies"), dict) else {}
    for key in out["mitigation_strategies"]:
        out["mitigation_strategies"][key] = _to_str_or_none(strategies.get(key)) or ""

    out["decision_support"] = _to_str_or_none(value.get("decision_support")) or ""
    return out


def _normalize_suggested_actions(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except Exception:
            return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [_to_str_or_none(item) or "" for item in value]
    return [_to_str_or_none(value) or ""]


def _normalize_raw_anomaly_data(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {"message": value}
    if not isinstance(value, dict):
        return {"value": _to_str_or_none(value) or ""}

    out: Dict[str, Any] = {}
    for key in (
        "is_anomaly",
        "anomaly_score",
        "rule_score",
        "ml_score",
        "category",
        "lift_type",
        "violations",
    ):
        if key in value:
            out[key] = value[key]

    violations = out.get("violations")
    if isinstance(violations, list):
        normalized_violations: List[Dict[str, Any]] = []
        for v in violations:
            if not isinstance(v, dict):
                continue
            normalized_violations.append(
                {
                    "field": _to_str_or_none(v.get("field")),
                    "value": v.get("value"),
                    "min": v.get("min"),
                    "max": v.get("max"),
                    "unit": _to_str_or_none(v.get("unit")),
                    "deviation_pct": v.get("deviation_pct"),
                    "violation": _to_str_or_none(v.get("violation")),
                }
            )
        out["violations"] = normalized_violations
    return out


def normalize_anomaly_suggestion_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a record with fixed column names, order, and coerced types for S3 parquet.
    Nested JSON fields use canonical shapes so schema never drifts between runs.
    """
    historical_context = _normalize_historical_context(record.get("historical_context"))
    risk_analysis = _normalize_risk_analysis(record.get("risk_analysis"))
    suggested_actions = _normalize_suggested_actions(record.get("suggested_actions"))
    raw_anomaly_data = _normalize_raw_anomaly_data(record.get("raw_anomaly_data"))

    lift_type = _to_str_or_none(record.get("lift_type"))
    category = _to_str_or_none(record.get("category")) or lift_type

    normalized = {
        "id": _to_str_or_none(record.get("id")),
        "well_id": _to_str_or_none(record.get("well_id")),
        "timestamp": _to_str_or_none(record.get("timestamp")),
        "alert_title": _to_str_or_none(record.get("alert_title")),
        "severity": _normalize_severity(record.get("severity")),
        "status": _normalize_status(record.get("status")),
        "open_up": None,
        "mute_till": None,
        "confidence": _normalize_confidence(record.get("confidence")),
        "description": _to_str_or_none(record.get("description")),
        "suggested_actions": suggested_actions,
        "explanation": _to_str_or_none(record.get("explanation")),
        "created_at": _to_str_or_none(record.get("created_at")),
        "historical_context": historical_context,
        "risk_analysis": risk_analysis,
        "asset_id": _to_str_or_none(record.get("asset_id")),
        "reason": None,
        "raw_anomaly_data": raw_anomaly_data,
        "lift_type": lift_type,
        "category": category,
        "well_status": _to_str_or_none(record.get("well_status")),
        "organization_id": _to_str_or_none(record.get("organization_id")),
        "user_id": None,
        "user_name": None,
        "updated_at": None,
    }
    return {k: normalized[k] for k in ANOMALY_SUGGESTIONS_COLUMNS}


def _normalize_anomaly_suggestions_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        row = normalize_anomaly_suggestion_record(r)
        row["suggested_actions"] = _json_dumps_safe(row.get("suggested_actions"))
        row["historical_context"] = _json_dumps_safe(row.get("historical_context"))
        row["risk_analysis"] = _json_dumps_safe(row.get("risk_analysis"))
        row["raw_anomaly_data"] = _json_dumps_safe(row.get("raw_anomaly_data"))
        rows.append(row)

    df = pd.DataFrame(rows, columns=ANOMALY_SUGGESTIONS_COLUMNS)
    df = _coerce_string_columns(df, ANOMALY_SUGGESTIONS_COLUMNS)
    return df


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

    df = _normalize_well_anomalies_records(records)
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

    df = _normalize_anomaly_violations_records(records)
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

    df = _normalize_anomaly_suggestions_records(records)
    _put_parquet_s3(df, bucket, key)
    return key

