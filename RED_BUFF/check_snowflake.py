#!/usr/bin/env python3
"""Verify Snowflake .env and connectivity for the RED_BUFF pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake_io import build_connect_kwargs, connect, snowflake_config


def main() -> int:
    cfg = snowflake_config()
    print("Snowflake config (secrets hidden):")
    for key in ("user", "account", "warehouse", "database", "schema", "role"):
        print(f"  {key}: {cfg.get(key)}")
    print(f"  token set: {bool((cfg.get('token') or '').strip())}")
    print(f"  password set: {bool(cfg.get('password'))}")
    print(f"  passcode set: {bool((cfg.get('passcode') or '').strip())}")

    kw = build_connect_kwargs(cfg, mode="auto")
    print(f"\nAuth mode: {kw.get('authenticator', 'password')}")

    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
        )
        row = cur.fetchone()
        print(f"\nConnected OK: user={row[0]} role={row[1]} db={row[2]} schema={row[3]}")
        conn.close()
        return 0
    except Exception as exc:
        print(f"\nConnection failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
