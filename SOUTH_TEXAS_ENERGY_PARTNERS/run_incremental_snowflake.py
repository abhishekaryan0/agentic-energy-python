#!/usr/bin/env python3
"""
Run the South Texas Energy Partners synthetic pipeline — Snowflake only.

Production path (same as RED_BUFF): twin engine → Snowflake directly.
  python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py
  python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py --reset

Optional dev shortcut — bulk load pre-built master JSON:
  python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py --from-master

Target: AT_RISK_ASSETS.SOUTH_TEXAS_SENSOR_DATA
Env: STEP_ORGANIZATION_ID, SNOWFLAKE_* (see README.md)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_STEP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _STEP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_STEP_DIR) not in sys.path:
    sys.path.insert(0, str(_STEP_DIR))

from config import STEP_ORGANIZATION_ID
from pipeline import (
    load_master_to_snowflake,
    reset_organization_data,
    run_step_pipeline,
)
from snowflake_io import snowflake_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Run STEP synthetic pipeline into Snowflake (SOUTH_TEXAS_SENSOR_DATA schema)."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help=f"Tenant UUID (default: STEP_ORGANIZATION_ID env or {STEP_ORGANIZATION_ID})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all synthetic data for this org before generating",
    )
    parser.add_argument(
        "--from-master",
        action="store_true",
        help="Bulk load South_Texas_Energy_Partners_Master.json instead of live generation",
    )
    parser.add_argument(
        "--master-path",
        default=None,
        help="Path to master JSON (default: package South_Texas_Energy_Partners_Master.json)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip Snowflake CREATE TABLE setup",
    )
    args = parser.parse_args()

    org_id = args.organization_id or STEP_ORGANIZATION_ID

    cfg = snowflake_config()
    print(f"Snowflake target: {cfg['database']}.{cfg['schema']}")
    print(f"Organization ID:  {org_id}")

    if args.reset:
        cleared = reset_organization_data(org_id, ensure_tables=not args.skip_setup)
        print("Reset complete:", cleared)

    if args.from_master:
        master_path = Path(args.master_path) if args.master_path else None
        result = load_master_to_snowflake(
            organization_id=org_id,
            master_path=master_path,
            ensure_tables=not args.skip_setup,
        )
        print(
            f"Done — bulk load from master JSON\n"
            f"  Schema:                    {result.get('schema')}\n"
            f"  WELL_DAILY_PRODUCTION:     {result['daily_rows']:,}\n"
            f"  WELL_SENSOR_READINGS:      {result['sensor_rows']:,}\n"
            f"  SYNTH_ASSETS:              {result['asset_rows']:,}\n"
            f"  SYNTH_PERSONNEL:           {result['personnel_rows']:,}\n"
            f"  SYNTH_ASSET_DAILY_HISTORY: {result['master_history_rows']:,}\n"
            f"  SYNTH_INVOICES:            {result['invoice_rows']:,}"
        )
        return

    result = run_step_pipeline(
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
