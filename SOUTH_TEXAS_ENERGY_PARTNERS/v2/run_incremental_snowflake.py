#!/usr/bin/env python3
"""
Run STEP V2 synthetic pipeline — same CLI as v1, into *_V2 Snowflake tables.

  python3 -m SOUTH_TEXAS_ENERGY_PARTNERS.v2.run_incremental_snowflake
  python3 -m SOUTH_TEXAS_ENERGY_PARTNERS.v2.run_incremental_snowflake --reset

Uses local v2/generate_digital_twin.py (same layout as v1).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parent
_STEP_DIR = _V2_DIR.parent
_PROJECT_ROOT = _STEP_DIR.parent
for _p in (_PROJECT_ROOT, _STEP_DIR, _V2_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from SOUTH_TEXAS_ENERGY_PARTNERS.v2.config import STEP_ORGANIZATION_ID
from SOUTH_TEXAS_ENERGY_PARTNERS.v2.pipeline import reset_organization_data, run_step_v2_pipeline
from SOUTH_TEXAS_ENERGY_PARTNERS.v2.snowflake_io import snowflake_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Run STEP V2 synthetic pipeline into Snowflake (*_V2 tables)."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help=f"Tenant UUID (default: {STEP_ORGANIZATION_ID})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all V2 synthetic data for this org before generating",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip Snowflake CREATE TABLE setup",
    )
    args = parser.parse_args()

    org_id = args.organization_id or STEP_ORGANIZATION_ID
    cfg = snowflake_config()
    print(f"Snowflake target: {cfg['database']}.{cfg['schema']} (*_V2)")
    print(f"Organization ID:  {org_id}")

    if args.reset:
        cleared = reset_organization_data(org_id, ensure_tables=not args.skip_setup)
        print("Reset complete:", cleared)

    result = run_step_v2_pipeline(
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
        f"  WELL_DAILY_PRODUCTION_V2:     {result['daily_rows']:,}\n"
        f"  WELL_SENSOR_READINGS_V2:      {result['sensor_rows']:,}\n"
        f"  SYNTH_ASSET_DAILY_HISTORY_V2: {result['master_history_rows']:,}\n"
        f"  SYNTH_INVOICES_V2:            {result['invoice_rows']:,}\n"
        f"  SYNTH_REVENUE_V2:             {result.get('revenue_rows', 0):,}"
    )
    if result.get("message"):
        print(f"  Note: {result['message']}")


if __name__ == "__main__":
    main()
