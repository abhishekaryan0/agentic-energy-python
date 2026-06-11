"""AWS Lambda entry point for Snowflake -> S3 parquet export."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from transformations.config import TransformationConfig
from transformations.run import run_transformations

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler. Event example: {"mode": "incremental"}
    Env vars mirror the ECS task definition (S3_BUCKET, ORGANIZATION_ID, SNOWFLAKE_*, etc.).
    """
    event = event or {}
    try:
        config = TransformationConfig.from_env()
        if event.get("mode"):
            config.mode = str(event["mode"]).strip()
        result = run_transformations(config, dry_run=bool(event.get("dry_run")))
        status = result.get("status", "error")
        return {
            "statusCode": 200 if status in ("success", "skipped") else 500,
            "body": json.dumps(result, default=str),
        }
    except Exception as exc:
        logger.exception("Transformations export failed")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(exc)}),
        }
