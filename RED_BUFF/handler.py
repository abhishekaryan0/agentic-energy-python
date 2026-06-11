"""AWS Lambda entry point for Red Bluff synthetic data generation."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

_RB = Path(__file__).resolve().parent
_ROOT = _RB.parent
for _path in (_ROOT, _RB):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pipeline import reset_organization_data, run_red_buff_pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Generate missing Red Bluff synthetic days through today (Snowflake).

    Event (optional):
      {"reset": true}  — wipe org data before generating
    """
    event = event or {}
    org_id = (os.getenv("ORGANIZATION_ID") or "").strip()
    if not org_id:
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": "ORGANIZATION_ID is required"}),
        }

    try:
        if event.get("reset"):
            cleared = reset_organization_data(org_id, ensure_tables=True)
            logger.info("Reset complete: %s", cleared)

        result = run_red_buff_pipeline(organization_id=org_id, ensure_tables=True)
        daily = int(result.get("daily_rows") or 0)
        sensor = int(result.get("sensor_rows") or 0)
        status = "success" if daily or sensor or result.get("message") else "skipped"
        body = {
            "status": status,
            "organization_id": org_id,
            "is_full_backfill": result.get("is_full_backfill"),
            "daily_rows": daily,
            "sensor_rows": sensor,
            "master_history_rows": int(result.get("master_history_rows") or 0),
            "invoice_rows": int(result.get("invoice_rows") or 0),
            "message": result.get("message"),
        }
        return {
            "statusCode": 200 if status in ("success", "skipped") else 500,
            "body": json.dumps(body, default=str),
        }
    except Exception as exc:
        logger.exception("Red Bluff pipeline failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
