"""AWS Lambda entry point for S3 detection parquet -> PostgreSQL sync."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List

from s3_to_postgres.config import load_skip_sync_tables, resolve_organization_ids
from s3_to_postgres.sync_anomaly_reviews import sync_anomaly_reviews
from s3_to_postgres.sync_anomaly_suggestions import sync_anomaly_suggestions
from s3_to_postgres.sync_daily_production import sync_daily_production
from s3_to_postgres.sync_monthly_production import sync_monthly_production
from s3_to_postgres.sync_operation_recommendations import sync_operation_recommendations

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

SYNC_HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "anomaly_suggestions": sync_anomaly_suggestions,
    "anomaly_reviews": sync_anomaly_reviews,
    "operation_recommendations": sync_operation_recommendations,
    "daily_production": sync_daily_production,
    "monthly_production": sync_monthly_production,
}


def run_sync(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run sync for one or more organizations and tables."""
    event = event or {}
    table = event.get("table") or os.getenv("SYNC_TABLE", "anomaly_suggestions")
    tables: List[str]
    if table == "all":
        skip_tables = load_skip_sync_tables(event)
        tables = [name for name in SYNC_HANDLERS if name not in skip_tables]
        if skip_tables:
            logger.info("Skipping sync tables: %s", ", ".join(sorted(skip_tables)))
    else:
        tables = [table]

    org_ids = resolve_organization_ids(event)
    results: List[Dict[str, Any]] = []

    for org_id in org_ids:
        for name in tables:
            sync_fn = SYNC_HANDLERS.get(name)
            if sync_fn is None:
                raise ValueError(f"Unsupported table: {name}")
            payload = {k: v for k, v in event.items() if k != "table"}
            payload["settings_overrides"] = {
                **(event.get("settings_overrides") or {}),
                "sync_table": name,
                "organization_id": org_id,
            }
            result = sync_fn(payload)
            result["organization_id"] = org_id
            results.append(result)

    return results


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler.

    Event examples:
      {"sync_mode": "incremental", "table": "anomaly_suggestions"}
      {"sync_mode": "backfill", "table": "all"}
      {"sync_mode": "incremental", "table": "all", "organization_id": "..."}
    """
    try:
        results = run_sync(event or {})
        body = results[0] if len(results) == 1 else {"status": "ok", "results": results}
        return {
            "statusCode": 200,
            "body": json.dumps(body, default=str),
        }
    except Exception as exc:
        logger.exception("S3 to Postgres sync failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
