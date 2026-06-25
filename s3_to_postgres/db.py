"""PostgreSQL connection and upsert helpers."""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.errors import DeadlockDetected
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

POSTGRES_DAILY_PRODUCTION_COLUMNS: Sequence[str] = (
    "organization_id",
    "well_id",
    "date",
    "oil_volume",
    "gas_volume",
    "water_volume",
    "boe",
    "lift_type",
    "status",
    "notes",
    "test_oil",
    "test_water",
    "test_gas",
    "facility_id",
)

POSTGRES_DAILY_PRODUCTION_CONFLICT_COLUMNS: Sequence[str] = ("well_id", "date")

POSTGRES_MONTHLY_PRODUCTION_COLUMNS: Sequence[str] = (
    "organization_id",
    "well_id",
    "month",
    "oil_volume",
    "gas_volume",
    "water_volume",
    "boe",
    "production_days",
    "oil_rate",
    "gas_rate",
    "water_rate",
    "boe_rate",
    "lift_type",
    "status",
    "test_oil",
    "test_water",
    "test_gas",
    "facility_id",
)

POSTGRES_MONTHLY_PRODUCTION_CONFLICT_COLUMNS: Sequence[str] = ("well_id", "month")

PRODUCTION_FLOAT_COLUMNS = frozenset(
    {
        "oil_volume",
        "gas_volume",
        "water_volume",
        "boe",
        "test_oil",
        "test_water",
        "test_gas",
        "oil_rate",
        "gas_rate",
        "water_rate",
        "boe_rate",
    }
)
PRODUCTION_DATE_COLUMNS = frozenset({"date", "month"})

DEFAULT_RETENTION_YEARS = 2

_PRODUCTION_LOCK_CLASS = 90421

_PRODUCTION_TABLE_COLUMNS: Dict[str, Sequence[tuple[str, str]]] = {
    "daily_production": (
        ("organization_id", "VARCHAR(36)"),
        ("well_id", "VARCHAR(100) NOT NULL"),
        ("date", "DATE NOT NULL"),
        ("oil_volume", "DOUBLE PRECISION"),
        ("gas_volume", "DOUBLE PRECISION"),
        ("water_volume", "DOUBLE PRECISION"),
        ("boe", "DOUBLE PRECISION"),
        ("lift_type", "VARCHAR(50)"),
        ("status", "VARCHAR(50)"),
        ("notes", "VARCHAR(500)"),
        ("test_oil", "DOUBLE PRECISION"),
        ("test_water", "DOUBLE PRECISION"),
        ("test_gas", "DOUBLE PRECISION"),
        ("facility_id", "VARCHAR(100)"),
    ),
    "monthly_production": (
        ("organization_id", "VARCHAR(36)"),
        ("well_id", "VARCHAR(100) NOT NULL"),
        ("month", "DATE NOT NULL"),
        ("oil_volume", "DOUBLE PRECISION"),
        ("gas_volume", "DOUBLE PRECISION"),
        ("water_volume", "DOUBLE PRECISION"),
        ("boe", "DOUBLE PRECISION"),
        ("production_days", "INTEGER"),
        ("oil_rate", "DOUBLE PRECISION"),
        ("gas_rate", "DOUBLE PRECISION"),
        ("water_rate", "DOUBLE PRECISION"),
        ("boe_rate", "DOUBLE PRECISION"),
        ("lift_type", "VARCHAR(50)"),
        ("status", "VARCHAR(50)"),
        ("test_oil", "DOUBLE PRECISION"),
        ("test_water", "DOUBLE PRECISION"),
        ("test_gas", "DOUBLE PRECISION"),
        ("facility_id", "VARCHAR(100)"),
    ),
}

T = TypeVar("T")

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


def _to_display_float(value: Any) -> Any:
    """Sensor/violation floats stored with 2 decimal places for UI."""
    parsed = _to_float(value)
    if parsed is None:
        return None
    return round(parsed, 2)


def _normalize_raw_anomaly_data_for_ui(value: Any) -> Any:
    data = _parse_json_value(value)
    if not isinstance(data, dict):
        return data
    violations = data.get("violations")
    if not isinstance(violations, list):
        return data
    normalized: List[Dict[str, Any]] = []
    for item in violations:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        for key in (
            "value",
            "min",
            "max",
            "observed",
            "expected_min",
            "expected_max",
            "deviation_pct",
            "deviation_percent",
        ):
            if key in row:
                rounded = _to_display_float(row[key])
                if rounded is not None:
                    row[key] = rounded
        normalized.append(row)
    out = dict(data)
    out["violations"] = normalized
    return out


def _source_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Map only parquet columns present in S3 (organization_id is always allowed)."""
    df_cols = set(df.columns)
    return [column for column in columns if column in df_cols or column == "organization_id"]


def _table_columns(conn, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            """,
            (table,),
        )
        return {row[0] for row in cur.fetchall()}


def _table_varchar_limits(conn, table: str) -> Dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
              AND character_maximum_length IS NOT NULL
            """,
            (table,),
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def _fit_varchar(value: Any, max_length: int) -> Any:
    if value is None or not isinstance(value, str):
        return value
    if len(value) <= max_length:
        return value
    return value[:max_length]


def _resolve_upsert_columns(
    conn,
    table: str,
    preferred_columns: Sequence[str],
    rows: List[Dict[str, Any]],
) -> List[str]:
    """Intersect preferred columns with row data and the live PostgreSQL table."""
    pg_cols = _table_columns(conn, table)
    row_keys = {key for row in rows for key in row}
    if pg_cols:
        resolved = [
            column
            for column in preferred_columns
            if column in pg_cols and column in row_keys
        ]
    else:
        resolved = [column for column in preferred_columns if column in row_keys]
    if not resolved:
        raise ValueError(f"No overlapping columns to upsert into {table}")
    skipped = [column for column in preferred_columns if column not in resolved]
    if skipped:
        logger.info("Skipping columns for %s: %s", table, ", ".join(skipped))
    return resolved


def dataframe_to_anomaly_suggestion_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    source_columns = _source_columns(df, POSTGRES_ANOMALY_SUGGESTIONS_COLUMNS)
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in source_columns:
            value = record.get(column)
            if column == "organization_id" and (value is None or (isinstance(value, float) and pd.isna(value))):
                value = organization_id
            if column in JSONB_COLUMNS:
                value = _parse_json_value(value)
                if column == "raw_anomaly_data":
                    value = _normalize_raw_anomaly_data_for_ui(value)
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
    source_columns = _source_columns(df, POSTGRES_ANOMALY_VIOLATIONS_COLUMNS)
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in source_columns:
            value = record.get(column)
            if column == "organization_id" and (value is None or (isinstance(value, float) and pd.isna(value))):
                value = organization_id
            elif column in FLOAT_COLUMNS:
                value = _to_display_float(value)
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


def _upsert_by_id(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Dict[str, Any]],
    json_columns: frozenset[str],
) -> int:
    return _upsert_on_conflict(
        conn,
        table,
        columns,
        ("id",),
        rows,
        json_columns=json_columns,
    )


def upsert_anomaly_suggestions(conn, rows: Iterable[Dict[str, Any]]) -> int:
    return _upsert_by_id(
        conn,
        "anomaly_suggestions",
        POSTGRES_ANOMALY_SUGGESTIONS_COLUMNS,
        rows,
        JSONB_COLUMNS,
    )


def upsert_anomaly_violations(conn, rows: Iterable[Dict[str, Any]]) -> int:
    return _upsert_on_conflict(
        conn,
        "anomaly_violations",
        POSTGRES_ANOMALY_VIOLATIONS_COLUMNS,
        POSTGRES_ANOMALY_VIOLATIONS_CONFLICT_COLUMNS,
        rows,
    )


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
    source_columns = _source_columns(df, columns)
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in source_columns:
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
    return rows


def dataframe_to_operation_recommendation_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    return _build_rows(
        df,
        organization_id,
        POSTGRES_OPERATION_RECOMMENDATIONS_COLUMNS,
        json_columns=OPS_JSONB_COLUMNS,
        float_columns=OPS_FLOAT_COLUMNS,
        timestamp_columns=OPS_TIMESTAMP_COLUMNS,
        required_fields=("id", "well_id"),
    )


def _upsert_on_conflict(
    conn,
    table: str,
    preferred_columns: Sequence[str],
    conflict_columns: Sequence[str],
    rows: Iterable[Dict[str, Any]],
    *,
    json_columns: frozenset[str] = frozenset(),
) -> int:
    rows = list(rows)
    if not rows:
        return 0

    col_list = _resolve_upsert_columns(conn, table, preferred_columns, rows)
    varchar_limits = _table_varchar_limits(conn, table)
    conflict_cols = [column for column in conflict_columns if column in col_list]
    if not conflict_cols:
        raise ValueError(f"Conflict columns {conflict_columns} not available for {table} upsert")
    update_columns = [column for column in col_list if column not in conflict_cols]
    placeholders = ", ".join(["%s"] * len(col_list))
    conflict_clause = ", ".join(conflict_cols)
    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    query = f"""
        INSERT INTO {table} ({", ".join(col_list)})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
    """

    params_list = []
    for row in rows:
        params: List[Any] = []
        for column in col_list:
            value = row.get(column)
            if column in json_columns and value is not None:
                value = Json(value)
            elif column in varchar_limits:
                value = _fit_varchar(value, varchar_limits[column])
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


def _ensure_table_columns(conn, table: str, column_defs: Sequence[tuple[str, str]]) -> None:
    existing = _table_columns(conn, table)
    with conn.cursor() as cur:
        for column_name, column_type in column_defs:
            if column_name in existing:
                continue
            cur.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
            )
            logger.info("Added missing column %s.%s", table, column_name)


def try_acquire_production_sync_lock(conn, organization_id: str) -> bool:
    """Serialize production sync per organization to avoid prune/upsert deadlocks."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(%s, hashtext(%s))",
            (_PRODUCTION_LOCK_CLASS, organization_id),
        )
        return bool(cur.fetchone()[0])


def release_production_sync_lock(conn, organization_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(%s, hashtext(%s))",
            (_PRODUCTION_LOCK_CLASS, organization_id),
        )


def run_with_deadlock_retry(
    conn,
    operation: Callable[[], T],
    *,
    max_attempts: int = 5,
) -> T:
    delay = 0.25
    for attempt in range(max_attempts):
        try:
            return operation()
        except DeadlockDetected:
            conn.rollback()
            if attempt >= max_attempts - 1:
                raise
            logger.warning(
                "Deadlock detected, retrying (%s/%s)",
                attempt + 1,
                max_attempts,
            )
            time.sleep(delay)
            delay = min(delay * 2, 4.0)
    raise RuntimeError("unreachable")


def _production_table_has_conflict_target(
    conn,
    table: str,
    conflict_columns: Sequence[str],
) -> bool:
    required = set(conflict_columns)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            JOIN pg_namespace n ON t.relnamespace = n.oid
            WHERE n.nspname = current_schema()
              AND t.relname = %s
              AND c.contype IN ('p', 'u')
            """,
            (table,),
        )
        for (definition,) in cur.fetchall():
            if all(column in definition for column in required):
                return True
        cur.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = %s
              AND indexdef ILIKE 'CREATE UNIQUE INDEX%%'
            """,
            (table,),
        )
        for (definition,) in cur.fetchall():
            if all(column in definition for column in required):
                return True
    return False


def _drop_empty_legacy_production_table(
    conn,
    table: str,
    conflict_columns: Sequence[str],
) -> None:
    if not _table_columns(conn, table):
        return
    if _production_table_has_conflict_target(conn, table, conflict_columns):
        return
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cur.fetchone()[0]
    if row_count > 0:
        raise ValueError(
            f"{table} exists with an incompatible schema and {row_count} rows; "
            "manual migration is required before S3 sync"
        )
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE {table} CASCADE")
    logger.warning(
        "Dropped empty legacy %s table (missing unique constraint on %s)",
        table,
        ", ".join(conflict_columns),
    )


def ensure_production_tables(conn) -> None:
    """Create production tables matching S3 parquet when they do not exist yet."""
    _drop_empty_legacy_production_table(conn, "daily_production", ("well_id", "date"))
    _drop_empty_legacy_production_table(conn, "monthly_production", ("well_id", "month"))
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_production (
                organization_id VARCHAR(36),
                well_id VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                oil_volume DOUBLE PRECISION,
                gas_volume DOUBLE PRECISION,
                water_volume DOUBLE PRECISION,
                boe DOUBLE PRECISION,
                lift_type VARCHAR(50),
                status VARCHAR(50),
                notes VARCHAR(500),
                test_oil DOUBLE PRECISION,
                test_water DOUBLE PRECISION,
                test_gas DOUBLE PRECISION,
                facility_id VARCHAR(100),
                PRIMARY KEY (well_id, date)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_production (
                organization_id VARCHAR(36),
                well_id VARCHAR(100) NOT NULL,
                month DATE NOT NULL,
                oil_volume DOUBLE PRECISION,
                gas_volume DOUBLE PRECISION,
                water_volume DOUBLE PRECISION,
                boe DOUBLE PRECISION,
                production_days INTEGER,
                oil_rate DOUBLE PRECISION,
                gas_rate DOUBLE PRECISION,
                water_rate DOUBLE PRECISION,
                boe_rate DOUBLE PRECISION,
                lift_type VARCHAR(50),
                status VARCHAR(50),
                test_oil DOUBLE PRECISION,
                test_water DOUBLE PRECISION,
                test_gas DOUBLE PRECISION,
                facility_id VARCHAR(100),
                PRIMARY KEY (well_id, month)
            )
            """
        )
        daily_cols = _table_columns(conn, "daily_production")
        if "well_id" in daily_cols and "date" in daily_cols:
            if not _production_table_has_conflict_target(conn, "daily_production", ("well_id", "date")):
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS daily_production_well_id_date_uidx
                    ON daily_production (well_id, date)
                    """
                )
    for table, column_defs in _PRODUCTION_TABLE_COLUMNS.items():
        if _table_columns(conn, table):
            _ensure_table_columns(conn, table, column_defs)


def dataframe_to_daily_production_rows(df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
    rows = _build_rows(
        df,
        organization_id,
        POSTGRES_DAILY_PRODUCTION_COLUMNS,
        json_columns=frozenset(),
        float_columns=PRODUCTION_FLOAT_COLUMNS,
        timestamp_columns=PRODUCTION_DATE_COLUMNS,
        required_fields=("well_id", "date"),
    )
    for row in rows:
        for column in PRODUCTION_FLOAT_COLUMNS:
            if column in row and row[column] is not None:
                row[column] = _to_display_float(row[column])
    return rows


def upsert_daily_production(conn, rows: Iterable[Dict[str, Any]]) -> int:
    return run_with_deadlock_retry(
        conn,
        lambda: _upsert_on_conflict(
            conn,
            "daily_production",
            POSTGRES_DAILY_PRODUCTION_COLUMNS,
            POSTGRES_DAILY_PRODUCTION_CONFLICT_COLUMNS,
            rows,
        ),
    )


def refresh_monthly_production(
    conn,
    organization_id: str,
    month_starts: Iterable[Any],
) -> int:
    """Aggregate daily_production into monthly_production for each month start date."""
    from datetime import date, timedelta

    columns = list(POSTGRES_MONTHLY_PRODUCTION_COLUMNS)
    conflict_cols = list(POSTGRES_MONTHLY_PRODUCTION_CONFLICT_COLUMNS)
    update_columns = [c for c in columns if c not in conflict_cols]
    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    conflict_clause = ", ".join(conflict_cols)

    def _refresh() -> int:
        refreshed = 0
        with conn.cursor() as cur:
            for raw_month in month_starts:
                if isinstance(raw_month, pd.Timestamp):
                    month_start = raw_month.date()
                elif isinstance(raw_month, date):
                    month_start = raw_month.replace(day=1)
                else:
                    month_start = pd.to_datetime(raw_month).date().replace(day=1)
                if month_start.day != 1:
                    month_start = month_start.replace(day=1)
                month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)

                cur.execute(
                    f"""
                    INSERT INTO monthly_production ({", ".join(columns)})
                    SELECT
                        organization_id,
                        well_id,
                        date_trunc('month', date)::date AS month,
                        ROUND(SUM(COALESCE(oil_volume, 0))::numeric, 2) AS oil_volume,
                        ROUND(SUM(COALESCE(gas_volume, 0))::numeric, 2) AS gas_volume,
                        ROUND(SUM(COALESCE(water_volume, 0))::numeric, 2) AS water_volume,
                        ROUND(SUM(COALESCE(boe, 0))::numeric, 2) AS boe,
                        COUNT(*)::int AS production_days,
                        ROUND(AVG(oil_volume)::numeric, 2) AS oil_rate,
                        ROUND(AVG(gas_volume)::numeric, 2) AS gas_rate,
                        ROUND(AVG(water_volume)::numeric, 2) AS water_rate,
                        ROUND(AVG(boe)::numeric, 2) AS boe_rate,
                        MAX(lift_type) AS lift_type,
                        MAX(status) AS status,
                        ROUND(SUM(COALESCE(test_oil, 0))::numeric, 2) AS test_oil,
                        ROUND(SUM(COALESCE(test_water, 0))::numeric, 2) AS test_water,
                        ROUND(SUM(COALESCE(test_gas, 0))::numeric, 2) AS test_gas,
                        MAX(facility_id) AS facility_id
                    FROM daily_production
                    WHERE organization_id = %s
                      AND date >= %s
                      AND date < %s
                    GROUP BY organization_id, well_id, date_trunc('month', date)::date
                    ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
                    """,
                    (organization_id, month_start, month_end),
                )
                refreshed += cur.rowcount
        return refreshed

    return run_with_deadlock_retry(conn, _refresh)


def retention_min_date(years: int = DEFAULT_RETENTION_YEARS) -> "date":
    from datetime import date, timedelta

    return date.today() - timedelta(days=365 * years)


def prune_production_outside_retention(
    conn,
    organization_id: str,
    min_date: "date",
) -> Dict[str, int]:
    """Keep only rows inside the retention window (default: last 2 years)."""

    def _prune() -> Dict[str, int]:
        min_month = min_date.replace(day=1)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM daily_production WHERE organization_id = %s AND date < %s",
                (organization_id, min_date),
            )
            daily_deleted = cur.rowcount
            cur.execute(
                "DELETE FROM monthly_production WHERE organization_id = %s AND month < %s",
                (organization_id, min_month),
            )
            monthly_deleted = cur.rowcount
        logger.info(
            "Pruned production outside retention (from %s): %s daily rows, %s monthly rows",
            min_date.isoformat(),
            daily_deleted,
            monthly_deleted,
        )
        return {"daily_deleted": daily_deleted, "monthly_deleted": monthly_deleted}

    return run_with_deadlock_retry(conn, _prune)
