#!/usr/bin/env python3
"""Verify Snowflake connectivity and row counts for the STEP pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LEGACY_ORGANIZATION_LABEL, STEP_ORGANIZATION_ID
from snowflake_io import build_connect_kwargs, connect, qualified_table, snowflake_config


def _count_table(cur, table: str, org_id: str) -> int:
    cur.execute(
        f'SELECT COUNT(*) FROM {qualified_table(table)} WHERE "ORGANIZATION_ID" = %s',
        (org_id,),
    )
    return int(cur.fetchone()[0])


def main() -> int:
    cfg = snowflake_config()
    print("Snowflake config (secrets hidden):")
    for key in ("user", "account", "warehouse", "database", "schema", "role"):
        print(f"  {key}: {cfg.get(key)}")
    print(f"  STEP_ORGANIZATION_ID: {STEP_ORGANIZATION_ID}")

    kw = build_connect_kwargs(cfg, mode="auto")
    print(f"\nAuth mode: {kw.get('authenticator', 'password')}")

    tables = [
        "WELL_DAILY_PRODUCTION",
        "WELL_SENSOR_READINGS",
        "SYNTH_ASSETS",
        "SYNTH_PERSONNEL",
        "SYNTH_ASSET_DAILY_HISTORY",
        "SYNTH_INVOICES",
    ]

    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
        )
        row = cur.fetchone()
        print(f"\nConnected OK: user={row[0]} role={row[1]} db={row[2]} schema={row[3]}")

        for label, org_id in (
            ("STEP (canonical)", STEP_ORGANIZATION_ID),
            ("legacy label", LEGACY_ORGANIZATION_LABEL),
        ):
            print(f"\nRow counts — {label} ({org_id}):")
            for table in tables:
                try:
                    print(f"  {table}: {_count_table(cur, table, org_id):,}")
                except Exception as exc:
                    print(f"  {table}: error ({exc})")

        conn.close()
        return 0
    except Exception as exc:
        print(f"\nConnection failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
