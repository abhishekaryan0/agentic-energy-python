#!/usr/bin/env python3
"""
Run the Apex Permian synthetic pipeline into Snowflake (incremental through today).

Uses the same field names and shapes as `APEX_PERMIAN/generate_digital_twin.py`.

Usage:
  python -m APEX_PERMIAN.run_incremental_snowflake
  python -m APEX_PERMIAN.run_incremental_snowflake --reset
  python -m APEX_PERMIAN.run_incremental_snowflake --organization-id 147a92b9-f90a-473d-b443-e6e668c4fbb7
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from APEX_PERMIAN.apex_permian_pipeline import reset_organization_data, run_apex_permian_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apex Permian synthetic pipeline → Snowflake")
    parser.add_argument("--organization-id", default=None, help="Tenant UUID (default: ORGANIZATION_ID env)")
    parser.add_argument("--reset", action="store_true", help="Delete org Snowflake rows before generating")
    parser.add_argument("--skip-setup", action="store_true", help="Skip Snowflake CREATE TABLE setup")
    args = parser.parse_args()

    org_id = (args.organization_id or os.getenv("ORGANIZATION_ID") or "").strip()
    if not org_id:
        parser.error("Set ORGANIZATION_ID or pass --organization-id")

    if args.reset:
        cleared = reset_organization_data(org_id, ensure_tables=not args.skip_setup)
        print("Reset complete:", cleared)

    result = run_apex_permian_pipeline(organization_id=org_id, ensure_tables=not args.skip_setup)
    ws = result["window_start"]
    we = result["window_end"]
    ws_s = ws.date().isoformat() if hasattr(ws, "date") else str(ws)
    we_s = we.date().isoformat() if hasattr(we, "date") else str(we)
    mode = "full backfill" if result.get("is_full_backfill") else "incremental"
    print(
        f"Done — {mode}, window {ws_s} – {we_s}\n"
        f"  WELL_DAILY_PRODUCTION: {result['daily_rows']:,}\n"
        f"  WELL_SENSOR_READINGS:  {result['sensor_rows']:,}"
    )
    if result.get("message"):
        print(f"  Note: {result['message']}")


if __name__ == "__main__":
    main()
