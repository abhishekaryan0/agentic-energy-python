"""AWS Lambda entry point: Apex Permian synthetic data → Snowflake."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from APEX_PERMIAN.apex_permian_pipeline import reset_organization_data, run_apex_permian_pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Generate missing Apex Permian synthetic days through today (Snowflake).

    Event (optional): {"reset": true}
    Env: ORGANIZATION_ID (default 147a92b9-f90a-473d-b443-e6e668c4fbb7 for apex-permian Lambda)
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

        result = run_apex_permian_pipeline(organization_id=org_id, ensure_tables=True)
        daily = int(result.get("daily_rows") or 0)
        sensor = int(result.get("sensor_rows") or 0)
        status = "success" if daily or sensor or result.get("message") else "skipped"
        body = {
            "status": status,
            "organization_id": org_id,
            "is_full_backfill": result.get("is_full_backfill"),
            "daily_rows": daily,
            "sensor_rows": sensor,
            "message": result.get("message"),
        }
        return {
            "statusCode": 200 if status in ("success", "skipped") else 500,
            "body": json.dumps(body, default=str),
        }
    except Exception as exc:
        logger.exception("Apex Permian pipeline failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
