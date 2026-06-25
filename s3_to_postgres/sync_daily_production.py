"""Sync daily_production parquet partitions from S3 into PostgreSQL."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set

import boto3
import pandas as pd

from s3_to_postgres.checkpoint import load_checkpoint, save_checkpoint
from s3_to_postgres.config import Settings, load_settings
from s3_to_postgres.db import (
    DEFAULT_RETENTION_YEARS,
    dataframe_to_daily_production_rows,
    ensure_production_tables,
    postgres_connection,
    prune_production_outside_retention,
    refresh_monthly_production,
    release_production_sync_lock,
    retention_min_date,
    try_acquire_production_sync_lock,
    upsert_daily_production,
)
from s3_to_postgres.s3_io import (
    DAILY_PRODUCTION_KEY_FRAGMENT,
    event_detail_s3_key,
    list_daily_production_keys,
    list_legacy_daily_production_key,
    parse_partition_date,
    read_parquet_from_s3,
)

logger = logging.getLogger(__name__)

DEFAULT_BACKFILL_YEARS = DEFAULT_RETENTION_YEARS


def resolve_retention_years(event: Dict[str, Any]) -> int:
    raw_years = event.get("backfill_years")
    if raw_years is None:
        raw_years = event.get("settings_overrides", {}).get("backfill_years")
    if raw_years is None:
        raw_years = DEFAULT_RETENTION_YEARS
    return max(1, int(raw_years))


def resolve_max_files(event: Dict[str, Any]) -> Optional[int]:
    raw = event.get("max_files")
    if raw is None:
        raw = event.get("settings_overrides", {}).get("max_files")
    if raw is None:
        return None
    return max(1, int(raw))


def _months_to_refresh(partition_dates: List[date]) -> List[date]:
    months: Set[date] = set()
    for partition_date in partition_dates:
        months.add(partition_date.replace(day=1))
    return sorted(months)


def _month_range(min_date: date, max_date: date) -> List[date]:
    months: List[date] = []
    current = min_date.replace(day=1)
    end = max_date.replace(day=1)
    while current <= end:
        months.append(current)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def _months_for_run(
    sync_mode: str,
    partition_dates: List[date],
    retention_start: date,
) -> List[date]:
    if sync_mode == "backfill":
        return _month_range(retention_start, date.today())
    if partition_dates:
        return _months_to_refresh(partition_dates)
    return [date.today().replace(day=1)]


def _filter_df_to_retention_window(df: pd.DataFrame, retention_start: date) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    dates = pd.to_datetime(df["date"], errors="coerce")
    return df.loc[dates.dt.date >= retention_start].copy()


def _select_production_keys_to_sync(
    all_keys: List[str],
    checkpoint,
    *,
    sync_mode: str,
    single_key: Optional[str],
    retention_start: date,
    reset_checkpoint: bool,
) -> List[str]:
    if single_key:
        if DAILY_PRODUCTION_KEY_FRAGMENT not in single_key:
            raise ValueError(f"Not a daily_production object: {single_key}")
        return [single_key]

    if sync_mode == "backfill":
        candidates = [
            key
            for key in all_keys
            if parse_partition_date(key) is None or parse_partition_date(key) >= retention_start
        ]
    else:
        candidates = list(all_keys)

    if reset_checkpoint:
        return candidates

    synced = checkpoint.synced_key_set
    return [key for key in candidates if key not in synced]


def _list_production_keys(
    s3,
    settings: Settings,
    *,
    retention_start: date,
) -> List[str]:
    keys = list_daily_production_keys(
        s3,
        settings.s3_bucket,
        settings.detection_prefix,
        min_partition_date=retention_start,
    )
    if not keys:
        legacy_key = list_legacy_daily_production_key(s3, settings.s3_bucket, settings.detection_prefix)
        if legacy_key:
            keys.append(legacy_key)
    keys.sort()
    return keys


def _finalize_production_sync(
    conn,
    *,
    organization_id: str,
    sync_mode: str,
    partition_dates: List[date],
    retention_start: date,
    prune: bool,
) -> Dict[str, Any]:
    months = _months_for_run(sync_mode, partition_dates, retention_start)
    monthly_rows = refresh_monthly_production(conn, organization_id, months)
    pruned = {"daily_deleted": 0, "monthly_deleted": 0}
    if prune:
        pruned = prune_production_outside_retention(conn, organization_id, retention_start)
    return {
        "months_refreshed": [m.isoformat() for m in months],
        "monthly_rows_refreshed": monthly_rows,
        "daily_rows_deleted": pruned["daily_deleted"],
        "monthly_rows_deleted": pruned["monthly_deleted"],
    }


def sync_daily_production(
    event: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    event = event or {}
    overrides = dict(event.get("settings_overrides") or {})
    overrides.setdefault("sync_table", "daily_production")
    settings = settings or load_settings(overrides)

    retention_years = resolve_retention_years(event)
    retention_start = retention_min_date(retention_years)
    sync_mode = str(event.get("sync_mode") or "incremental").strip().lower()
    max_files = resolve_max_files(event)
    reset_checkpoint = bool(event.get("reset_checkpoint"))

    s3 = boto3.client("s3", region_name=settings.aws_region)
    checkpoint = load_checkpoint(
        s3,
        settings.s3_bucket,
        settings.checkpoint_key,
        settings.organization_id,
        settings.sync_table,
    )
    if reset_checkpoint:
        checkpoint.synced_keys = []

    all_keys = _list_production_keys(s3, settings, retention_start=retention_start)
    single_key = event_detail_s3_key(event)
    keys_to_sync = _select_production_keys_to_sync(
        all_keys,
        checkpoint,
        sync_mode=sync_mode,
        single_key=single_key,
        retention_start=retention_start,
        reset_checkpoint=reset_checkpoint,
    )
    pending_count = len(keys_to_sync)
    if max_files is not None and len(keys_to_sync) > max_files:
        keys_to_sync = keys_to_sync[:max_files]

    result: Dict[str, Any] = {
        "table": settings.sync_table,
        "schema": settings.postgres_schema,
        "sync_mode": sync_mode,
        "retention_years": retention_years,
        "retention_min_date": retention_start.isoformat(),
        "files_found": len(all_keys),
        "files_to_sync": pending_count,
        "files_synced": [],
        "rows_upserted": 0,
        "monthly_rows_refreshed": 0,
        "daily_rows_deleted": 0,
        "monthly_rows_deleted": 0,
        "status": "ok",
    }

    total_rows = 0
    synced_files: List[str] = []
    partition_dates: List[date] = []
    lock_held = False

    with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
        ensure_production_tables(conn)
        if not try_acquire_production_sync_lock(conn, settings.organization_id):
            conn.commit()
            result["status"] = "skipped"
            result["skip_reason"] = "production_sync_in_progress"
            logger.warning(
                "Skipping daily_production sync for %s: another sync holds the advisory lock",
                settings.organization_id,
            )
            return result

        lock_held = True
        try:
            if keys_to_sync:
                for key in keys_to_sync:
                    logger.info("Syncing s3://%s/%s", settings.s3_bucket, key)
                    df = read_parquet_from_s3(s3, settings.s3_bucket, key)
                    df = _filter_df_to_retention_window(df, retention_start)
                    if df.empty:
                        logger.info(
                            "No rows in retention window for s3://%s/%s",
                            settings.s3_bucket,
                            key,
                        )
                        synced_files.append(key)
                    else:
                        rows = dataframe_to_daily_production_rows(df, settings.organization_id)
                        count = upsert_daily_production(conn, rows)
                        total_rows += count
                        synced_files.append(key)
                        partition_date = parse_partition_date(key)
                        if partition_date is not None:
                            partition_dates.append(partition_date)

                    checkpoint.synced_keys.append(key)
                    conn.commit()
                    checkpoint.synced_keys = sorted(set(checkpoint.synced_keys))
                    save_checkpoint(
                        s3,
                        settings.s3_bucket,
                        settings.checkpoint_key,
                        checkpoint,
                        status="partial" if pending_count > len(synced_files) else "ok",
                        rows_upserted=total_rows,
                    )

            is_complete = pending_count <= len(synced_files)
            finalized = _finalize_production_sync(
                conn,
                organization_id=settings.organization_id,
                sync_mode=sync_mode if is_complete else "incremental",
                partition_dates=partition_dates,
                retention_start=retention_start,
                prune=is_complete,
            )
            conn.commit()
            result.update(finalized)
        finally:
            if lock_held:
                conn.rollback()
                try:
                    release_production_sync_lock(conn, settings.organization_id)
                    conn.commit()
                except Exception:
                    conn.rollback()

    if not keys_to_sync:
        result["status"] = "skipped"
    elif pending_count > len(synced_files):
        result["status"] = "partial"
        result["files_remaining"] = pending_count - len(synced_files)
    else:
        checkpoint.synced_keys = sorted(set(checkpoint.synced_keys))
        save_checkpoint(
            s3,
            settings.s3_bucket,
            settings.checkpoint_key,
            checkpoint,
            status=result["status"],
            rows_upserted=total_rows,
        )

    result["files_synced"] = synced_files
    result["rows_upserted"] = total_rows
    logger.info(
        "Sync complete: %s files, %s rows upserted, %s monthly refreshed, "
        "%s daily / %s monthly rows pruned (retention from %s), status=%s",
        len(synced_files),
        total_rows,
        result["monthly_rows_refreshed"],
        result["daily_rows_deleted"],
        result["monthly_rows_deleted"],
        retention_start.isoformat(),
        result["status"],
    )
    return result
