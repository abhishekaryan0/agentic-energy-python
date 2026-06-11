"""PostgreSQL persistence for well status rows (org schema = ORGANIZATION_ID)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from psycopg2.extras import execute_batch

from s3_to_postgres.db import postgres_connection
from well_status.config import WellStatusSettings, load_settings

logger = logging.getLogger(__name__)

# Matches backend-managed well_status table in PostgreSQL.
POSTGRES_WELL_STATUS_COLUMNS: Sequence[str] = (
    "well_id",
    "organization_id",
    "lift_type",
    "status",
    "last_reading_timestamp",
    "last_reading_age_hours",
    "strokes_per_minute",
    "motor_current",
    "injection_rate",
    "anomaly_count_24h",
    "critical_anomaly_count_24h",
    "status_reason",
    "updated_at",
)

FLOAT_COLUMNS = frozenset(
    {
        "last_reading_age_hours",
        "strokes_per_minute",
        "motor_current",
        "injection_rate",
    }
)
INT_COLUMNS = frozenset({"anomaly_count_24h", "critical_anomaly_count_24h"})
TIMESTAMP_COLUMNS = frozenset({"last_reading_timestamp", "updated_at"})

SELECT_COLUMNS: Sequence[str] = POSTGRES_WELL_STATUS_COLUMNS


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


def _to_int(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dataframe_to_well_status_rows(
    df: pd.DataFrame, organization_id: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for column in POSTGRES_WELL_STATUS_COLUMNS:
            value = record.get(column)
            if column == "organization_id" and (
                value is None or (isinstance(value, float) and pd.isna(value))
            ):
                value = organization_id
            elif column in FLOAT_COLUMNS:
                value = _to_float(value)
            elif column in INT_COLUMNS:
                value = _to_int(value)
            elif column in TIMESTAMP_COLUMNS:
                value = _to_scalar(value)
            else:
                value = _to_scalar(value)
            row[column] = value
        if not row.get("well_id") or not row.get("status"):
            continue
        rows.append(row)
    return rows


def upsert_well_statuses(conn, rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    columns = list(POSTGRES_WELL_STATUS_COLUMNS)
    update_columns = [c for c in columns if c != "well_id"]
    placeholders = ", ".join(["%s"] * len(columns))
    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    query = f"""
        INSERT INTO well_status ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (well_id) DO UPDATE SET {update_clause}
    """
    params_list = [[row.get(column) for column in columns] for row in rows]
    with conn.cursor() as cur:
        execute_batch(cur, query, params_list, page_size=100)
    return len(params_list)


def upsert_well_status_dataframe(
    df: pd.DataFrame,
    settings: Optional[WellStatusSettings] = None,
) -> int:
    if df is None or df.empty:
        return 0
    settings = settings or load_settings()
    rows = dataframe_to_well_status_rows(df, settings.organization_id)
    if not rows:
        return 0
    with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
        count = upsert_well_statuses(conn, rows)
    logger.info("Upserted %s well_status rows to schema %s", count, settings.postgres_schema)
    return count


def get_well_status_row(
    well_id: str, settings: Optional[WellStatusSettings] = None
) -> Optional[Dict[str, Any]]:
    settings = settings or load_settings()
    col_list = ", ".join(SELECT_COLUMNS)
    with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {col_list} FROM well_status WHERE well_id = %s",
                (str(well_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return dict(zip(SELECT_COLUMNS, row))


def get_all_well_status_rows(
    settings: Optional[WellStatusSettings] = None,
) -> List[Dict[str, Any]]:
    settings = settings or load_settings()
    col_list = ", ".join(SELECT_COLUMNS)
    with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {col_list}
                FROM well_status
                ORDER BY CASE status
                    WHEN 'SHUTDOWN' THEN 1
                    WHEN 'IDLE' THEN 2
                    ELSE 3
                END, well_id
                """
            )
            return [dict(zip(SELECT_COLUMNS, row)) for row in cur.fetchall()]
