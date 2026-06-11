"""PostgreSQL connection and upsert helpers."""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_batch

from s3_to_postgres.config import parse_database_url

logger = logging.getLogger(__name__)

# Columns that exist in the backend-managed anomaly_suggestions table.
POSTGRES_ANOMALY_SUGGESTIONS_COLUMNS: Sequence[str] = (
    "id",
    "organization_id",
    "well_id",
    "asset_id",
    "lift_type",
    "timestamp",
    "well_status",
    "alert_title",
    "severity",
    "status",
    "confidence",
    "description",
    "suggested_actions",
    "explanation",
    "historical_context",
    "risk_analysis",
    "raw_anomaly_data",
    "created_at",
)

JSONB_COLUMNS = frozenset(
    {"suggested_actions", "historical_context", "risk_analysis", "raw_anomaly_data"}
)

POSTGRES_ANOMALY_VIOLATIONS_COLUMNS: Sequence[str] = (
    "anomaly_id",
    "well_id",
    "timestamp",
    "parameter_name",
    "actual_value",
    "expected_min",
    "expected_max",
    "unit",
    "deviation_percent",
    "organization_id",
)

POSTGRES_ANOMALY_VIOLATIONS_CONFLICT_COLUMNS: Sequence[str] = (
    "anomaly_id",
    "parameter_name",
)

FLOAT_COLUMNS = frozenset(
    {"actual_value", "expected_min", "expected_max", "deviation_percent"}
)

POSTGRES_ANOMALY_REVIEWS_COLUMNS: Sequence[str] = (
    "id",
    "well_id",
    "event_date",
    "detected_at",
    "anomaly_code",
    "category",
    "severity",
    "title",
    "ui_text",
    "impact_value",
    "confidence",
    "chart_data",
    "status",
    "reason",
    "impact_metrics",
    "updated_at",
    "mute_till",
    "open_up",
    "user_id",
    "user_name",
    "organization_id",
    "created_at",
)

POSTGRES_OPERATION_RECOMMENDATIONS_COLUMNS: Sequence[str] = (
    "id",
    "well_id",
    "category",
    "action",
    "status",
    "priority",
    "confidence_percent",
    "production_increase_bbl_day",
    "production_increase_percent",
    "daily_expense_benefit_usd",
    "implementation_cost_usd",
    "net_daily_benefit_usd",
    "asset_value_increase_usd",
    "time_reduced_hours",
    "detailed_analysis",
    "current_performance",
    "historical_trend",
    "conservative_approach",
    "aggressive_optimization",
    "hybrid_model",
    "recommended_model",
    "citations",
    "referenced_data",
    "supporting_analysis",
    "data_quality",
    "reason",
    "expected_impact",
    "metrics_json",
    "created_at",
    "updated_at",
    "status_reason",
    "mute_till",
    "open_up",
    "user_id",
    "user_name",
    "organization_id",
)

REVIEW_JSONB_COLUMNS = frozenset({"ui_text", "chart_data", "impact_metrics"})
OPS_JSONB_COLUMNS = frozenset({"metrics_json"})
OPS_FLOAT_COLUMNS = frozenset(
    {
        "confidence_percent",
        "production_increase_bbl_day",
        "production_increase_percent",
        "daily_expense_benefit_usd",
        "implementation_cost_usd",
        "net_daily_benefit_usd",
        "asset_value_increase_usd",
        "time_reduced_hours",
    }
)
REVIEW_FLOAT_COLUMNS = frozenset({"impact_value", "confidence"})
REVIEW_TIMESTAMP_COLUMNS = frozenset(
    {"event_date", "detected_at", "updated_at", "mute_till", "open_up", "created_at"}
)
OPS_TIMESTAMP_COLUMNS = frozenset({"created_at", "updated_at", "mute_till", "open_up"})

_SCHEMA_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def _validate_schema_name(schema: str) -> str:
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"Invalid PostgreSQL schema name: {schema!r}")
    return schema


@contextmanager
def postgres_connection(database_url: str, schema: str):
    clean_url, sslmode = parse_database_url(database_url)
    schema = _validate_schema_name(schema)
    conn = psycopg2.connect(clean_url, sslmode=sslmode)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _parse_json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _to_timestamp(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        try:
            value = pd.to_datetime(value, utc=True)
        except Exception:
            return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _to_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _to_float(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dataframe_to_anomaly_suggestion_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in POSTGRES_ANOMALY_SUGGESTIONS_COLUMNS:
            value = record.get(column)
            if column == "organization_id" and (value is None or (isinstance(value, float) and pd.isna(value))):
                value = organization_id
            if column in JSONB_COLUMNS:
                value = _parse_json_value(value)
            elif column in {"timestamp", "created_at"}:
                value = _to_timestamp(value)
            else:
                value = _to_scalar(value)
            row[column] = value

        if not row.get("id"):
            continue
        if not row.get("well_id"):
            continue
        if row.get("timestamp") is None:
            continue
        rows.append(row)
    return rows


def dataframe_to_anomaly_violation_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in POSTGRES_ANOMALY_VIOLATIONS_COLUMNS:
            value = record.get(column)
            if column == "organization_id" and (value is None or (isinstance(value, float) and pd.isna(value))):
                value = organization_id
            elif column in FLOAT_COLUMNS:
                value = _to_float(value)
            elif column == "timestamp":
                value = _to_timestamp(value)
            else:
                value = _to_scalar(value)
            row[column] = value

        if not row.get("anomaly_id"):
            continue
        if not row.get("well_id"):
            continue
        if not row.get("parameter_name"):
            continue
        if row.get("timestamp") is None:
            continue
        rows.append(row)
    return rows


# Backward-compatible alias
dataframe_to_rows = dataframe_to_anomaly_suggestion_rows


def upsert_anomaly_suggestions(conn, rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    columns = list(POSTGRES_ANOMALY_SUGGESTIONS_COLUMNS)
    update_columns = [c for c in columns if c != "id"]
    placeholders = ", ".join(["%s"] * len(columns))
    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    query = f"""
        INSERT INTO anomaly_suggestions ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (id) DO UPDATE SET {update_clause}
    """

    params_list = []
    for row in rows:
        params: List[Any] = []
        for column in columns:
            value = row.get(column)
            if column in JSONB_COLUMNS and value is not None:
                value = Json(value)
            params.append(value)
        params_list.append(params)

    with conn.cursor() as cur:
        execute_batch(cur, query, params_list, page_size=100)
    return len(params_list)


def upsert_anomaly_violations(conn, rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    columns = list(POSTGRES_ANOMALY_VIOLATIONS_COLUMNS)
    conflict_cols = list(POSTGRES_ANOMALY_VIOLATIONS_CONFLICT_COLUMNS)
    update_columns = [c for c in columns if c not in conflict_cols]
    placeholders = ", ".join(["%s"] * len(columns))
    conflict_clause = ", ".join(conflict_cols)
    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    query = f"""
        INSERT INTO anomaly_violations ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
    """

    params_list = []
    for row in rows:
        params = [row.get(column) for column in columns]
        params_list.append(params)

    with conn.cursor() as cur:
        execute_batch(cur, query, params_list, page_size=100)
    return len(params_list)


def _build_rows(
    df: pd.DataFrame,
    organization_id: str,
    columns: Sequence[str],
    *,
    json_columns: frozenset[str],
    float_columns: frozenset[str],
    timestamp_columns: frozenset[str],
    required_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in columns:
            value = record.get(column)
            if column == "organization_id" and (value is None or (isinstance(value, float) and pd.isna(value))):
                value = organization_id
            elif column in json_columns:
                value = _parse_json_value(value)
            elif column in float_columns:
                value = _to_float(value)
            elif column in timestamp_columns:
                value = _to_timestamp(value)
            else:
                value = _to_scalar(value)
            row[column] = value

        if any(not row.get(field) for field in required_fields):
            continue
        rows.append(row)
    return rows


def dataframe_to_anomaly_review_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    rows = _build_rows(
        df,
        organization_id,
        POSTGRES_ANOMALY_REVIEWS_COLUMNS,
        json_columns=REVIEW_JSONB_COLUMNS,
        float_columns=REVIEW_FLOAT_COLUMNS,
        timestamp_columns=REVIEW_TIMESTAMP_COLUMNS,
        required_fields=("id", "well_id"),
    )
    for row in rows:
        if row.get("created_at") is None:
            row["created_at"] = row.get("detected_at")
    return rows


def dataframe_to_operation_recommendation_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    rows = _build_rows(
        df,
        organization_id,
        POSTGRES_OPERATION_RECOMMENDATIONS_COLUMNS,
        json_columns=OPS_JSONB_COLUMNS,
        float_columns=OPS_FLOAT_COLUMNS,
        timestamp_columns=OPS_TIMESTAMP_COLUMNS,
        required_fields=("id", "well_id"),
    )
    for row in rows:
        if row.get("created_at") is None and row.get("updated_at") is not None:
            row["created_at"] = row["updated_at"]
    return rows


def _upsert_by_id(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Dict[str, Any]],
    json_columns: frozenset[str],
) -> int:
    rows = list(rows)
    if not rows:
        return 0

    col_list = list(columns)
    update_columns = [c for c in col_list if c != "id"]
    placeholders = ", ".join(["%s"] * len(col_list))
    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    query = f"""
        INSERT INTO {table} ({", ".join(col_list)})
        VALUES ({placeholders})
        ON CONFLICT (id) DO UPDATE SET {update_clause}
    """

    params_list = []
    for row in rows:
        params: List[Any] = []
        for column in col_list:
            value = row.get(column)
            if column in json_columns and value is not None:
                value = Json(value)
            params.append(value)
        params_list.append(params)

    with conn.cursor() as cur:
        execute_batch(cur, query, params_list, page_size=100)
    return len(params_list)


def upsert_anomaly_reviews(conn, rows: Iterable[Dict[str, Any]]) -> int:
    return _upsert_by_id(conn, "anomaly_reviews", POSTGRES_ANOMALY_REVIEWS_COLUMNS, rows, REVIEW_JSONB_COLUMNS)


def upsert_operation_recommendations(conn, rows: Iterable[Dict[str, Any]]) -> int:
    return _upsert_by_id(
        conn,
        "operation_recommendations",
        POSTGRES_OPERATION_RECOMMENDATIONS_COLUMNS,
        rows,
        OPS_JSONB_COLUMNS,
    )
