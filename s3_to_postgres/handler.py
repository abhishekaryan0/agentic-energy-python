"""AWS Lambda entry point for S3 detection parquet -> PostgreSQL sync."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List

from s3_to_postgres.sync_anomaly_reviews import sync_anomaly_reviews
from s3_to_postgres.sync_anomaly_suggestions import sync_anomaly_suggestions
from s3_to_postgres.sync_anomaly_violations import sync_anomaly_violations
from s3_to_postgres.sync_operation_recommendations import sync_operation_recommendations

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

SYNC_HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "anomaly_suggestions": sync_anomaly_suggestions,
    "anomaly_violations": sync_anomaly_violations,
    "anomaly_reviews": sync_anomaly_reviews,
    "operation_recommendations": sync_operation_recommendations,
}


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler.

    Event examples:
      {"sync_mode": "incremental", "table": "anomaly_suggestions"}
      {"sync_mode": "backfill", "table": "anomaly_violations"}
      {"sync_mode": "backfill", "table": "all"}
      {"s3_key": "org-id/detections/anomaly_detection/history/anomaly_violations_run_ts=....parquet"}
    """
    try:
        event = event or {}
        table = event.get("table") or os.getenv("SYNC_TABLE", "anomaly_suggestions")
        tables: List[str]
        if table == "all":
            tables = list(SYNC_HANDLERS.keys())
        else:
            tables = [table]

        results = []
        for name in tables:
            sync_fn = SYNC_HANDLERS.get(name)
            if sync_fn is None:
                raise ValueError(f"Unsupported table: {name}")
            payload = {k: v for k, v in event.items() if k != "table"}
            payload["settings_overrides"] = {
                **(event.get("settings_overrides") or {}),
                "sync_table": name,
            }
            results.append(sync_fn(payload))

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
