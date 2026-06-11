#!/usr/bin/env python3
"""
Run the full Red Bluff synthetic pipeline — Snowflake only.

Empty org: full history (spud → today) for all wells.
Has data: append from last daily date + 1 through today.

Requires .env: SNOWFLAKE_USER, SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE,
ORGANIZATION_ID, and one of:
  - SNOWFLAKE_TOKEN (programmatic access token), or
  - SNOWFLAKE_PASSWORD (+ SNOWFLAKE_PASSCODE if MFA TOTP is enabled)

Usage:
  python RED_BUFF/run_incremental_snowflake.py --reset   # wipe org data first
  python RED_BUFF/run_incremental_snowflake.py          # generate through today
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_RED_BUFF_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RED_BUFF_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_RED_BUFF_DIR) not in sys.path:
    sys.path.insert(0, str(_RED_BUFF_DIR))

from pipeline import reset_organization_data, run_red_buff_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Run Red Bluff synthetic pipeline into Snowflake (through today)."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help="Tenant UUID (default: ORGANIZATION_ID env)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all synthetic data for this org before generating",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip Snowflake CREATE TABLE setup",
    )
    args = parser.parse_args()

    org_id = args.organization_id or os.getenv("ORGANIZATION_ID")
    if not org_id:
        parser.error("Set ORGANIZATION_ID or pass --organization-id")

    if args.reset:
        cleared = reset_organization_data(org_id, ensure_tables=not args.skip_setup)
        print("Reset complete:", cleared)

    result = run_red_buff_pipeline(
        organization_id=org_id,
        ensure_tables=not args.skip_setup,
    )
    ws = result["window_start"]
    we = result["window_end"]
    ws_s = ws.date().isoformat() if hasattr(ws, "date") else str(ws)
    we_s = we.date().isoformat() if hasattr(we, "date") else str(we)
    mode = "full backfill" if result.get("is_full_backfill") else "incremental"
    print(
        f"Done — {mode}, window {ws_s} – {we_s}\n"
        f"  WELL_DAILY_PRODUCTION:     {result['daily_rows']:,}\n"
        f"  WELL_SENSOR_READINGS:      {result['sensor_rows']:,}\n"
        f"  SYNTH_ASSET_DAILY_HISTORY: {result['master_history_rows']:,}\n"
        f"  SYNTH_INVOICES:            {result['invoice_rows']:,}"
    )
    if result.get("message"):
        print(f"  Note: {result['message']}")


if __name__ == "__main__":
    main()
