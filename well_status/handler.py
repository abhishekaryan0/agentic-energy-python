"""AWS Lambda entry point for well status updates."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from well_status.service import update_all_well_statuses

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Read sensor parquet from S3, compute status, upsert to PostgreSQL."""
    try:
        if event and event.get("force_full"):
            os.environ["WELL_STATUS_FORCE_FULL"] = "1"
        result = update_all_well_statuses()
        status = result.get("status", "success")
        return {
            "statusCode": 200 if status in ("success", "skipped") else 500,
            "body": json.dumps(result, default=str),
        }
    except Exception as exc:
        logger.exception("Well status update failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
