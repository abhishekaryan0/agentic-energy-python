"""Rebuild monthly_production from daily_production in PostgreSQL."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import boto3

from s3_to_postgres.config import Settings, load_settings
from s3_to_postgres.db import (
    ensure_production_tables,
    postgres_connection,
    prune_production_outside_retention,
    refresh_monthly_production,
    retention_min_date,
)
from s3_to_postgres.sync_daily_production import DEFAULT_BACKFILL_YEARS, resolve_retention_years

logger = logging.getLogger(__name__)


def _month_range(min_date: date, max_date: date) -> List[date]:
    months: List[date] = []
    current = min_date.replace(day=1)
    end = max_date.replace(day=1)
    while current <= end:
        months.append(current)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def sync_monthly_production(
    event: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Aggregate daily_production into monthly_production.

    On incremental runs, refreshes the current calendar month from daily rows.
    On backfill runs, refreshes every month in the backfill window (default 2 years).
    """
    event = event or {}
    overrides = dict(event.get("settings_overrides") or {})
    overrides.setdefault("sync_table", "monthly_production")
    settings = settings or load_settings(overrides)

    sync_mode = str(event.get("sync_mode") or "incremental").strip().lower()
    retention_years = resolve_retention_years(event)
    retention_start = retention_min_date(retention_years)
    today = date.today()

    if sync_mode == "backfill":
        months = _month_range(retention_start, today)
    else:
        months = [today.replace(day=1)]

    with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
        ensure_production_tables(conn)
        monthly_rows = refresh_monthly_production(conn, settings.organization_id, months)
        pruned = prune_production_outside_retention(conn, settings.organization_id, retention_start)

    result: Dict[str, Any] = {
        "table": settings.sync_table,
        "schema": settings.postgres_schema,
        "sync_mode": sync_mode,
        "retention_years": retention_years,
        "retention_min_date": retention_start.isoformat(),
        "months_refreshed": [m.isoformat() for m in months],
        "monthly_rows_refreshed": monthly_rows,
        "daily_rows_deleted": pruned["daily_deleted"],
        "monthly_rows_deleted": pruned["monthly_deleted"],
        "status": "ok",
    }
    logger.info(
        "Monthly production refresh complete: %s months, %s rows into %s.monthly_production",
        len(months),
        monthly_rows,
        settings.postgres_schema,
    )
    return result
