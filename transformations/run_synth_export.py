#!/usr/bin/env python3
"""Export Snowflake SYNTH_* tables to S3 master + financial parquet (not sensor_data)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from transformations.config import TransformationConfig
from transformations.synth_export import run_synth_export
from transformations.synth_schemas import DEFAULT_SYNTH_SOURCES

logger = logging.getLogger(__name__)


def _load_env() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _config_from_args(args) -> TransformationConfig:
    config = TransformationConfig.from_env()
    if args.organization_id:
        config.organization_id = args.organization_id.strip()
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.mode:
        config.mode = args.mode
    return config


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(
        description="Export Snowflake SYNTH_* (master + invoices) to S3 parquet."
    )
    parser.add_argument("--organization-id", default=None)
    parser.add_argument(
        "--mode",
        choices=("full", "incremental"),
        default=None,
        help="full = all history/invoices; incremental = since synth checkpoint.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=f"Comma-separated: {','.join(DEFAULT_SYNTH_SOURCES)} (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import os

    if args.mode:
        os.environ["SYNTH_EXPORT_MODE"] = args.mode

    config = _config_from_args(args)
    config.validate()
    config = config.with_org_snowflake_schema()

    source_names = None
    if args.sources:
        source_names = [p.strip() for p in args.sources.split(",") if p.strip()]

    export_mode = (
        args.mode
        or os.getenv("SYNTH_EXPORT_MODE")
        or os.getenv("TRANSFORM_MODE", "full")
    ).strip().lower()

    result: Dict[str, Any] = run_synth_export(
        config,
        mode=export_mode,
        source_names=source_names,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
