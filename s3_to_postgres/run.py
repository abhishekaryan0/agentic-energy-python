"""
Local / manual entry point.

Usage:
  export S3_BUCKET=...
  export ORGANIZATION_ID=...                        # single tenant
  export ORGANIZATION_IDS=org-a,org-b               # or multiple tenants
  export DATABASE_URL=...
  python -m s3_to_postgres.run --table anomaly_suggestions
  python -m s3_to_postgres.run --table daily_production --sync-mode backfill --backfill-years 2
  python -m s3_to_postgres.run --table all --sync-mode backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from s3_to_postgres.handler import run_sync

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Sync detection parquet from S3 to PostgreSQL")
    parser.add_argument(
        "--sync-mode",
        choices=["incremental", "backfill"],
        default="incremental",
        help="incremental: only new files; backfill: all files not in checkpoint",
    )
    parser.add_argument("--s3-key", help="Sync a single S3 object key")
    parser.add_argument(
        "--backfill-years",
        type=int,
        default=2,
        help="For production backfill: only sync partitions from the last N years (default: 2)",
    )
    parser.add_argument(
        "--table",
        default="anomaly_suggestions",
        choices=[
            "anomaly_suggestions",
            "anomaly_reviews",
            "operation_recommendations",
            "daily_production",
            "monthly_production",
            "all",
        ],
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit daily_production files per run (resume via checkpoint on next invoke)",
    )
    parser.add_argument(
        "--organization-id",
        help="Sync a single organization (overrides ORGANIZATION_IDS env)",
    )
    args = parser.parse_args()

    event: dict = {
        "sync_mode": args.sync_mode,
        "table": args.table,
        "backfill_years": args.backfill_years,
    }
    if args.s3_key:
        event["s3_key"] = args.s3_key
    if args.max_files is not None:
        event["max_files"] = args.max_files
    if args.organization_id:
        event["organization_id"] = args.organization_id

    results = run_sync(event)

    if len(results) == 1:
        print(json.dumps(results[0], indent=2, default=str))
        return 0 if results[0].get("status") != "error" else 1

    print(json.dumps({"status": "ok", "results": results}, indent=2, default=str))
    failed = any(item.get("status") == "error" for item in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
