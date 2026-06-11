"""Sync anomaly_violations parquet files from S3 into PostgreSQL."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import boto3

from s3_to_postgres.checkpoint import load_checkpoint, save_checkpoint
from s3_to_postgres.config import Settings, load_settings
from s3_to_postgres.db import (
    dataframe_to_anomaly_violation_rows,
    postgres_connection,
    upsert_anomaly_violations,
)
from s3_to_postgres.s3_io import (
    ANOMALY_VIOLATIONS_KEY_FRAGMENT,
    event_detail_s3_key,
    list_anomaly_violation_keys,
    read_parquet_from_s3,
)
from s3_to_postgres.sync_common import select_keys_to_sync

logger = logging.getLogger(__name__)


def sync_anomaly_violations(
    event: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    event = event or {}
    overrides = dict(event.get("settings_overrides") or {})
    overrides.setdefault("sync_table", "anomaly_violations")
    settings = settings or load_settings(overrides)

    s3 = boto3.client("s3", region_name=settings.aws_region)
    checkpoint = load_checkpoint(
        s3,
        settings.s3_bucket,
        settings.checkpoint_key,
        settings.organization_id,
        settings.sync_table,
    )

    all_keys = list_anomaly_violation_keys(s3, settings.s3_bucket, settings.detection_prefix)
    sync_mode = str(event.get("sync_mode") or "incremental").strip().lower()
    single_key = event_detail_s3_key(event)
    keys_to_sync = select_keys_to_sync(
        all_keys,
        checkpoint,
        sync_mode=sync_mode,
        single_key=single_key,
        key_fragment=ANOMALY_VIOLATIONS_KEY_FRAGMENT,
    )

    result: Dict[str, Any] = {
        "table": settings.sync_table,
        "schema": settings.postgres_schema,
        "sync_mode": sync_mode,
        "files_found": len(all_keys),
        "files_to_sync": len(keys_to_sync),
        "files_synced": [],
        "rows_upserted": 0,
        "status": "ok",
    }

    if not keys_to_sync:
        save_checkpoint(s3, settings.s3_bucket, settings.checkpoint_key, checkpoint, status="skipped", rows_upserted=0)
        result["status"] = "skipped"
        return result

    total_rows = 0
    synced_files: List[str] = []

    with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
        for key in keys_to_sync:
            logger.info("Syncing s3://%s/%s", settings.s3_bucket, key)
            df = read_parquet_from_s3(s3, settings.s3_bucket, key)
            rows = dataframe_to_anomaly_violation_rows(df, settings.organization_id)
            count = upsert_anomaly_violations(conn, rows)
            total_rows += count
            synced_files.append(key)
            checkpoint.synced_keys.append(key)

    checkpoint.synced_keys = sorted(set(checkpoint.synced_keys))
    save_checkpoint(
        s3,
        settings.s3_bucket,
        settings.checkpoint_key,
        checkpoint,
        status="ok",
        rows_upserted=total_rows,
    )

    result["files_synced"] = synced_files
    result["rows_upserted"] = total_rows
    logger.info(
        "Sync complete: %s files, %s rows upserted into %s.anomaly_violations",
        len(synced_files),
        total_rows,
        settings.postgres_schema,
    )
    return result
