"""
Local / manual entry point.

Usage:
  export S3_BUCKET=...
  export ORGANIZATION_ID=...
  export DATABASE_URL=...
  python -m s3_to_postgres.run --table anomaly_suggestions
  python -m s3_to_postgres.run --table anomaly_violations --sync-mode backfill
  python -m s3_to_postgres.run --table daily_production --sync-mode backfill --backfill-years 2
  python -m s3_to_postgres.run --table all --sync-mode backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from s3_to_postgres.handler import SYNC_HANDLERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
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
            "anomaly_violations",
            "anomaly_reviews",
            "operation_recommendations",
            "daily_production",
            "monthly_production",
            "all",
        ],
    )
    args = parser.parse_args()

    event = {"sync_mode": args.sync_mode, "table": args.table, "backfill_years": args.backfill_years}
    if args.s3_key:
        event["s3_key"] = args.s3_key

    tables = list(SYNC_HANDLERS.keys()) if args.table == "all" else [args.table]
    results = []
    for table in tables:
        payload = {k: v for k, v in event.items() if k != "table"}
        payload["settings_overrides"] = {"sync_table": table}
        results.append(SYNC_HANDLERS[table](payload))

    if len(results) == 1:
        print(json.dumps(results[0], indent=2, default=str))
        return 0 if results[0].get("status") != "error" else 1

    print(json.dumps({"status": "ok", "results": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
