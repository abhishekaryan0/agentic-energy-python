"""AWS Lambda entry for RED_BUFF V2 → Snowflake *_V2 tables."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from .config import RED_BUFF_ORGANIZATION_ID
from .pipeline import reset_organization_data, run_step_v2_pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Generate RED_BUFF V2 synthetic data into Snowflake (*_V2).

    Event (optional):
      {"reset": true}
      {"force_full_backfill": true, "well_offset": 0, "well_limit": 10}
      {"include_supporting": true}
    """
    event = event or {}
    org_id = (os.getenv("RED_BUFF_ORGANIZATION_ID") or RED_BUFF_ORGANIZATION_ID).strip()
    if not org_id:
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": "RED_BUFF_ORGANIZATION_ID is required"}),
        }

    try:
        if event.get("reset"):
            cleared = reset_organization_data(org_id, ensure_tables=True)
            logger.info("V2 reset complete: %s", cleared)

        well_offset = _as_int(event.get("well_offset"), 0) or 0
        well_limit = _as_int(event.get("well_limit"), None)
        force_full = bool(event.get("force_full_backfill") or event.get("force_full"))
        include_supporting = event.get("include_supporting")
        if include_supporting is not None:
            include_supporting = bool(include_supporting)

        result = run_step_v2_pipeline(
            organization_id=org_id,
            ensure_tables=True,
            well_offset=well_offset,
            well_limit=well_limit,
            force_full_backfill=force_full,
            include_supporting=include_supporting,
        )
        daily = int(result.get("daily_rows") or 0)
        sensor = int(result.get("sensor_rows") or 0)
        status = "success" if daily or sensor or result.get("message") else "skipped"
        body = {
            "status": status,
            "organization_id": org_id,
            "is_full_backfill": result.get("is_full_backfill"),
            "well_offset": result.get("well_offset"),
            "wells_processed": result.get("wells_processed"),
            "wells_total": result.get("wells_total"),
            "next_well_offset": result.get("next_well_offset"),
            "supporting_loaded": result.get("supporting_loaded"),
            "daily_rows": daily,
            "sensor_rows": sensor,
            "master_history_rows": int(result.get("master_history_rows") or 0),
            "invoice_rows": int(result.get("invoice_rows") or 0),
            "revenue_rows": int(result.get("revenue_rows") or 0),
            "message": result.get("message"),
        }
        return {"statusCode": 200, "body": json.dumps(body, default=str)}
    except Exception as exc:
        logger.exception("RED_BUFF V2 pipeline failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
