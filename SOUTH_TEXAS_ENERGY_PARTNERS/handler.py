"""AWS Lambda entry point for South Texas Energy Partners synthetic data generation."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

_STEP = Path(__file__).resolve().parent
_ROOT = _STEP.parent
for _path in (_ROOT, _STEP):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from config import STEP_ORGANIZATION_ID
from pipeline import load_master_to_snowflake, reset_organization_data, run_step_pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Generate missing STEP synthetic days through today (Snowflake).

    Event (optional):
      {"reset": true}         — wipe org data before generating
      {"from_master": true}   — bulk load master JSON (dev only)
    """
    event = event or {}
    org_id = (os.getenv("STEP_ORGANIZATION_ID") or STEP_ORGANIZATION_ID).strip()
    if not org_id:
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": "STEP_ORGANIZATION_ID is required"}),
        }

    try:
        if event.get("reset"):
            cleared = reset_organization_data(org_id, ensure_tables=True)
            logger.info("Reset complete: %s", cleared)

        if event.get("from_master"):
            result = load_master_to_snowflake(organization_id=org_id, ensure_tables=True)
            body = {
                "status": "success",
                "mode": "from_master",
                "organization_id": org_id,
                "daily_rows": int(result.get("daily_rows") or 0),
                "sensor_rows": int(result.get("sensor_rows") or 0),
                "master_history_rows": int(result.get("master_history_rows") or 0),
                "invoice_rows": int(result.get("invoice_rows") or 0),
            }
        else:
            result = run_step_pipeline(organization_id=org_id, ensure_tables=True)
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
            "statusCode": 200,
            "body": json.dumps(body, default=str),
        }
    except Exception as exc:
        logger.exception("STEP pipeline failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
