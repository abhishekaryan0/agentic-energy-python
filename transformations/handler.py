"""AWS Lambda entry point for Snowflake -> S3 parquet export."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from transformations.config import TransformationConfig, resolve_organization_ids
from transformations.run import run_transformations

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def run_transformations_for_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Export Snowflake sensor tables for one or more organizations."""
    event = event or {}
    org_ids = resolve_organization_ids(event)
    base = TransformationConfig.from_env()
    if event.get("mode"):
        base.mode = str(event["mode"]).strip()
    dry_run = bool(event.get("dry_run"))
    multi_org = len(org_ids) > 1

    results = []
    for org_id in org_ids:
        config = TransformationConfig(
            snowflake=base.snowflake,
            s3_bucket=base.s3_bucket,
            organization_id=org_id,
            output_prefix=None if multi_org else base.output_prefix,
            tables=list(base.tables),
            batch_size=base.batch_size,
            write_empty_files=base.write_empty_files,
            aws_region=base.aws_region,
            mode=base.mode,
            write_partitions=base.write_partitions,
            write_snapshots=base.write_snapshots,
            merge_snapshots=base.merge_snapshots,
        )
        logger.info("Transform export for organization_id=%s", org_id)
        results.append(run_transformations(config, dry_run=dry_run))

    if len(results) == 1:
        return results[0]

    statuses = {r.get("status") for r in results}
    if statuses <= {"success", "skipped"}:
        overall = "success" if "success" in statuses else "skipped"
    else:
        overall = "error"
    return {
        "status": overall,
        "organization_ids": org_ids,
        "results": results,
    }


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler. Event examples:
      {"mode": "incremental"}
      {"mode": "full", "organization_id": "147a92b9-f90a-473d-b443-e6e668c4fbb7"}
    Env: S3_BUCKET, ORGANIZATION_IDS or ORGANIZATION_ID, SNOWFLAKE_*, etc.
  Each org writes to s3://$S3_BUCKET/{ORGANIZATION_ID}/sensor_data/ (same layout per tenant).
    """
    event = event or {}
    try:
        result = run_transformations_for_event(event)
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
