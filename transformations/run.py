#!/usr/bin/env python3
"""CLI entrypoint for exporting Snowflake tables into S3 parquet files."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from transformations.config import TransformationConfig
from transformations.export import run_export
from transformations.schemas import get_export_specs
from transformations.snowflake import ExportWindow, build_select_sql, connect_snowflake

logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")


def _planned_exports(config: TransformationConfig) -> List[Dict[str, Any]]:
    from transformations.checkpoint import load_checkpoint, production_window, sensor_window
    from transformations.s3 import build_s3_key

    specs = get_export_specs(config.tables)
    prefix = config.sensor_data_prefix()
    checkpoint = None
    if config.organization_id:
        checkpoint = load_checkpoint(
            config.s3_bucket,
            prefix,
            config.organization_id,
            region_name=config.aws_region,
        )

    planned = []
    for spec in specs:
        window = None
        if config.mode == "incremental":
            if spec.watermark_column == "date":
                start, end = production_window(checkpoint)
                window = ExportWindow(exclusive_start=start, inclusive_end=end)
            else:
                start, end = sensor_window(checkpoint)
                window = ExportWindow(exclusive_start=start, inclusive_end=end)

        sql, params = build_select_sql(config, spec, window=window)
        planned.append(
            {
                "table": spec.source_table,
                "snapshot": f"s3://{config.s3_bucket}/{build_s3_key(prefix, spec.output_filename)}",
                "partition_prefix": f"s3://{config.s3_bucket}/{prefix}/{spec.partition_subdir}/date=",
                "columns": len(spec.columns),
                "filtered_by_organization": bool(config.organization_id),
                "window": {
                    "exclusive_start": str(window.exclusive_start) if window else None,
                    "inclusive_end": str(window.inclusive_end) if window else None,
                },
                "sql": sql,
                "params": params,
            }
        )
    return planned


def run_transformations(
    config: Optional[TransformationConfig] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run configured Snowflake-to-S3 parquet exports."""
    config = config or TransformationConfig.from_env()
    config.validate()
    specs = get_export_specs(config.tables)

    if dry_run:
        return {
            "status": "success",
            "dry_run": True,
            "mode": config.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exports": _planned_exports(config),
        }

    result = run_export(config, specs)
    result["dry_run"] = False
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform Snowflake canonical tables into S3 parquet files."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help="Tenant UUID (default: ORGANIZATION_ID from environment).",
    )
    parser.add_argument(
        "--tables",
        help=(
            "Comma-separated table aliases to export. Defaults to TRANSFORM_TABLES "
            "or well_sensor_readings,well_daily_production."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        help="S3 prefix for output files. Defaults to {ORGANIZATION_ID}/sensor_data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Snowflake fetch batch size. Defaults to TRANSFORM_BATCH_SIZE or 10000.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "incremental"),
        help="Export mode. full = entire org history; incremental = since checkpoint.",
    )
    parser.add_argument(
        "--no-partitions",
        action="store_true",
        help="Skip writing date=YYYY-MM-DD partition files.",
    )
    parser.add_argument(
        "--write-snapshots",
        action="store_true",
        help="Also write root snapshot parquet files (daily_production.parquet, well_sensor_readings.parquet).",
    )
    parser.add_argument(
        "--no-merge-snapshots",
        action="store_true",
        help="When --write-snapshots: incremental does not merge into existing snapshot files.",
    )
    parser.add_argument(
        "--skip-empty-files",
        action="store_true",
        help="Do not upload parquet files when a source query returns zero rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print planned exports without reading or writing data.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level. Default: INFO.",
    )
    return parser


def main() -> None:
    _load_project_env()
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = TransformationConfig.from_env()
    if args.organization_id:
        config.organization_id = args.organization_id.strip()
    if args.tables:
        config.tables = [part.strip() for part in args.tables.split(",") if part.strip()]
    if args.output_prefix:
        config.output_prefix = args.output_prefix
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.mode:
        config.mode = args.mode
    if args.no_partitions:
        config.write_partitions = False
    if args.write_snapshots:
        config.write_snapshots = True
    if args.no_merge_snapshots:
        config.merge_snapshots = False
    if args.skip_empty_files:
        config.write_empty_files = False

    config = config.with_org_snowflake_schema()
    result = run_transformations(config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
    status = result.get("status")
    sys.exit(0 if status in ("success", "skipped") else 1)


if __name__ == "__main__":
    main()
