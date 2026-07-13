"""AWS Lambda entry point for Snowflake -> S3 parquet export."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from transformations.config import TransformationConfig, resolve_organization_ids
from transformations.run import run_transformations
from transformations.synth_export import run_synth_export

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
    if event.get("tables"):
        raw = event["tables"]
        if isinstance(raw, str):
            base.tables = [p.strip() for p in raw.split(",") if p.strip()]
        elif isinstance(raw, list):
            base.tables = [str(p).strip() for p in raw if str(p).strip()]
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
        ).with_org_snowflake_schema()
        logger.info(
            "Transform export for organization_id=%s schema=%s",
            org_id,
            config.snowflake.schema,
        )
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


def run_synth_export_for_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Export Snowflake SYNTH_* tables (master + invoices) for one organization."""
    event = event or {}
    org_ids = resolve_organization_ids(event)
    if len(org_ids) != 1:
        raise ValueError("synth export requires exactly one organization_id")
    base = TransformationConfig.from_env()
    mode = str(event.get("mode") or base.mode).strip()
    dry_run = bool(event.get("dry_run"))
    config = TransformationConfig(
        snowflake=base.snowflake,
        s3_bucket=base.s3_bucket,
        organization_id=org_ids[0],
        output_prefix=base.output_prefix,
        tables=list(base.tables),
        batch_size=base.batch_size,
        write_empty_files=base.write_empty_files,
        aws_region=base.aws_region,
        mode=mode,
        write_partitions=base.write_partitions,
        write_snapshots=base.write_snapshots,
        merge_snapshots=base.merge_snapshots,
    ).with_org_snowflake_schema()
    logger.info(
        "Synth export for organization_id=%s schema=%s mode=%s",
        org_ids[0],
        config.snowflake.schema,
        mode,
    )
    return run_synth_export(config, mode=mode, dry_run=dry_run)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler. Event examples:
      {"mode": "incremental"}
      {"mode": "full", "organization_id": "147a92b9-f90a-473d-b443-e6e668c4fbb7"}
      {"mode": "full", "organization_id": "d270f89a-...", "synth": true}
    Env: S3_BUCKET, ORGANIZATION_IDS or ORGANIZATION_ID, SNOWFLAKE_*, etc.
  Each org writes to s3://$S3_BUCKET/{ORGANIZATION_ID}/sensor_data/ (same layout per tenant).
    """
    event = event or {}
    try:
        if event.get("synth") or event.get("export_type") == "synth":
            result = run_synth_export_for_event(event)
        else:
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
