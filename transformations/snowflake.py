"""Snowflake readers for transformation exports."""

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from transformations.config import SnowflakeConnectionConfig, TransformationConfig
from transformations.schemas import TableExportSpec

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class ExportWindow:
    """Optional exclusive-start / inclusive-end bounds for incremental export."""

    exclusive_start: Optional[object] = None
    inclusive_end: Optional[object] = None


def quote_identifier(value: str) -> str:
    """
    Quote Snowflake identifiers created by the project with unquoted names.

    This intentionally accepts only simple identifiers so env-driven database and
    schema values cannot become arbitrary SQL.
    """
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return f'"{value.upper()}"'


def connect_snowflake(config: SnowflakeConnectionConfig):
    """Create a Snowflake connector connection."""
    try:
        from snowflake.connector import connect as snowflake_connect
    except ImportError as exc:
        raise RuntimeError(
            "snowflake-connector-python is required for transformations"
        ) from exc
    kwargs = dict(config.as_connect_kwargs())
    # Large fetch_pandas_batches downloads can exceed default read timeouts.
    raw = (os.getenv("SNOWFLAKE_SOCKET_TIMEOUT") or "600").strip()
    try:
        kwargs["socket_timeout"] = int(raw)
    except ValueError:
        kwargs["socket_timeout"] = 600
    return snowflake_connect(**kwargs)


def _snowflake_physical_column(logical_name: str, spec: TableExportSpec) -> str:
    """Map logical parquet column name to Snowflake table column (quoted)."""
    if spec.source_table == "WELL_SENSOR_READINGS" and logical_name == "freq_hz":
        return quote_identifier("drive_frequency")
    return quote_identifier(logical_name)


def build_select_sql(
    config: TransformationConfig,
    spec: TableExportSpec,
    window: Optional[ExportWindow] = None,
) -> Tuple[str, List[object]]:
    """Build parameterized SELECT SQL for a supported export spec."""
    database = quote_identifier(config.snowflake.database or "")
    schema = quote_identifier(config.snowflake.schema or "")
    table = quote_identifier(spec.source_table)
    qualified_table = f"{database}.{schema}.{table}"

    select_columns = [
        f'{_snowflake_physical_column(column, spec)} AS "{column}"'
        for column in spec.columns
    ]
    sql = [f"SELECT {', '.join(select_columns)}", f"FROM {qualified_table}"]
    params: List[object] = []
    clauses = []

    if config.organization_id and spec.organization_column in spec.columns:
        clauses.append(f"{quote_identifier(spec.organization_column)} = %s")
        params.append(config.organization_id)

    if window and window.exclusive_start is not None and spec.watermark_column:
        col = quote_identifier(spec.watermark_column)
        clauses.append(f"{col} > %s")
        params.append(window.exclusive_start)

    if window and window.inclusive_end is not None and spec.watermark_column:
        col = quote_identifier(spec.watermark_column)
        clauses.append(f"{col} <= %s")
        params.append(window.inclusive_end)

    if clauses:
        sql.append("WHERE " + " AND ".join(clauses))

    order_columns = [column for column in spec.order_by if column in spec.columns]
    if order_columns:
        sql.append(
            "ORDER BY "
            + ", ".join(quote_identifier(column) for column in order_columns)
        )

    return "\n".join(sql), params


def count_rows_for_organization(
    conn,
    database: str,
    schema: str,
    table_name: str,
    organization_id: str,
) -> int:
    """COUNT(*) for rows scoped to one tenant (matches export WHERE clause)."""
    qdb = quote_identifier(database)
    qs = quote_identifier(schema)
    qt = quote_identifier(table_name)
    qorg = quote_identifier("organization_id")
    sql = f"SELECT COUNT(*) FROM {qdb}.{qs}.{qt} WHERE {qorg} = %s"
    cur = conn.cursor()
    try:
        cur.execute(sql, (organization_id,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        cur.close()


def count_rows_for_export_window(
    conn,
    config: TransformationConfig,
    spec: TableExportSpec,
    window: Optional[ExportWindow] = None,
) -> int:
    """COUNT(*) with the same WHERE clause as build_select_sql (incremental window)."""
    database = quote_identifier(config.snowflake.database or "")
    schema = quote_identifier(config.snowflake.schema or "")
    table = quote_identifier(spec.source_table)
    qualified_table = f"{database}.{schema}.{table}"

    params: List[object] = []
    clauses = []

    if config.organization_id and spec.organization_column in spec.columns:
        clauses.append(f"{quote_identifier(spec.organization_column)} = %s")
        params.append(config.organization_id)

    if window and window.exclusive_start is not None and spec.watermark_column:
        col = quote_identifier(spec.watermark_column)
        clauses.append(f"{col} > %s")
        params.append(window.exclusive_start)

    if window and window.inclusive_end is not None and spec.watermark_column:
        col = quote_identifier(spec.watermark_column)
        clauses.append(f"{col} <= %s")
        params.append(window.inclusive_end)

    sql = f"SELECT COUNT(*) FROM {qualified_table}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        cur.close()


def count_incremental_new_rows(
    conn,
    config: TransformationConfig,
    specs: List[TableExportSpec],
    checkpoint,
) -> Dict[str, int]:
    """
    Rows in Snowflake since checkpoint watermarks (same filter as incremental export).

    Requires checkpoint with at least one watermark; otherwise returns empty dict.
    """
    from transformations.checkpoint import production_window, sensor_window

    has_wm = bool(
        checkpoint.production_high_watermark or checkpoint.sensor_high_watermark
    )
    if not has_wm:
        return {}

    out: Dict[str, int] = {}
    seen: set[str] = set()
    for spec in specs:
        if spec.source_table in seen:
            continue
        seen.add(spec.source_table)
        if spec.watermark_column == "date":
            start, end = production_window(checkpoint)
            window = ExportWindow(exclusive_start=start, inclusive_end=end)
        else:
            start, end = sensor_window(checkpoint)
            window = ExportWindow(exclusive_start=start, inclusive_end=end)
        out[spec.source_table] = count_rows_for_export_window(
            conn, config, spec, window=window
        )
    return out


def preflight_org_row_counts(
    conn,
    config: TransformationConfig,
    specs: List[TableExportSpec],
) -> Dict[str, int]:
    """
    Per Snowflake source table: how many rows exist for this organization_id.

    Used to validate tenant scope before streaming (same filter as RED_BUFF-style data).
    """
    org = (config.organization_id or "").strip()
    if not org:
        return {}
    db = config.snowflake.database or ""
    schema = config.snowflake.schema or ""
    out: Dict[str, int] = {}
    seen: set[str] = set()
    for spec in specs:
        if spec.source_table in seen:
            continue
        seen.add(spec.source_table)
        out[spec.source_table] = count_rows_for_organization(
            conn, db, schema, spec.source_table, org
        )
    return out


def get_snowflake_watermarks(
    conn,
    config: TransformationConfig,
) -> Tuple[Optional[date], Optional[datetime]]:
    """Return (max production date, max sensor timestamp) for the configured org."""
    org = config.organization_id
    if not org:
        return None, None

    database = quote_identifier(config.snowflake.database or "")
    schema = quote_identifier(config.snowflake.schema or "")
    prod_table = f"{database}.{schema}.{quote_identifier('WELL_DAILY_PRODUCTION')}"
    sensor_table = f"{database}.{schema}.{quote_identifier('WELL_SENSOR_READINGS')}"
    org_col = quote_identifier("organization_id")

    cur = conn.cursor()
    try:
        cur.execute(
            f'SELECT MAX("DATE") FROM {prod_table} WHERE {org_col} = %s',
            (org,),
        )
        prod_row = cur.fetchone()
        last_date = prod_row[0] if prod_row and prod_row[0] else None
        if last_date is not None and hasattr(last_date, "date"):
            last_date = last_date.date()

        cur.execute(
            f'SELECT MAX("TIMESTAMP") FROM {sensor_table} WHERE {org_col} = %s',
            (org,),
        )
        sensor_row = cur.fetchone()
        last_ts = sensor_row[0] if sensor_row and sensor_row[0] else None
        return last_date, last_ts
    finally:
        cur.close()


def iter_table_batches(
    conn,
    config: TransformationConfig,
    spec: TableExportSpec,
    window: Optional[ExportWindow] = None,
) -> Iterator[pd.DataFrame]:
    """Yield pandas DataFrames from Snowflake for a single export spec."""
    sql, params = build_select_sql(config, spec, window=window)
    logger.info("Exporting Snowflake table %s", spec.source_table)
    if window and (window.exclusive_start or window.inclusive_end):
        logger.info(
            "  window: (%s, %s]",
            window.exclusive_start,
            window.inclusive_end,
        )

    cur = conn.cursor()
    try:
        cur.arraysize = config.batch_size
        cur.execute(sql, params)
        for batch in cur.fetch_pandas_batches():
            if batch is None:
                continue
            batch.columns = [str(column).lower() for column in batch.columns]
            yield batch.reindex(columns=list(spec.columns))
    finally:
        cur.close()
