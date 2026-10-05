"""Snowflake connectivity and append helpers for RED_BUFF V2 (*_V2 tables)."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import RED_BUFF_V2_SNOWFLAKE_SCHEMA, TABLE_NAMES
from .tables import ALL_TABLES

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_dotenv() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            pass


def snowflake_config() -> Dict[str, Any]:
    _load_dotenv()
    return {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "AT_RISK_ASSETS"),
        "schema": RED_BUFF_V2_SNOWFLAKE_SCHEMA,
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "token": os.getenv("SNOWFLAKE_TOKEN"),
        "passcode": os.getenv("SNOWFLAKE_PASSCODE"),
        "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR"),
    }


def _clean_connect_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None and v != ""}


def build_connect_kwargs(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "auto",
) -> Dict[str, Any]:
    cfg = cfg or snowflake_config()
    missing = [k for k in ("user", "account", "warehouse", "role") if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake env: {', '.join(missing)}")

    kwargs: Dict[str, Any] = {
        "user": cfg["user"],
        "account": cfg["account"],
        "warehouse": cfg["warehouse"],
        "database": cfg.get("database"),
        "schema": cfg.get("schema"),
        "role": cfg["role"],
    }
    network_policy = (os.getenv("SNOWFLAKE_NETWORK_POLICY") or "").strip()
    if network_policy:
        kwargs["network_policy"] = network_policy

    token = (cfg.get("token") or "").strip()
    authenticator = (cfg.get("authenticator") or "").strip()
    passcode = (cfg.get("passcode") or "").strip()
    password = (cfg.get("password") or "").strip() or None
    use_password = mode == "password" or (
        mode == "auto"
        and os.getenv("SNOWFLAKE_USE_PASSWORD", "").strip().lower() in ("1", "true", "yes")
    )

    if token and mode in ("auto", "token") and not use_password:
        kwargs["password"] = token
        return _clean_connect_kwargs(kwargs)

    if authenticator and not use_password and mode != "password":
        kwargs["authenticator"] = authenticator
        if password:
            kwargs["password"] = password
        return _clean_connect_kwargs(kwargs)

    if not password:
        raise ValueError(
            "Snowflake auth: set SNOWFLAKE_TOKEN (PAT), or SNOWFLAKE_PASSWORD "
            "(use SNOWFLAKE_USE_PASSWORD=1 to force password when a token is also set)."
        )
    kwargs["password"] = password
    if passcode:
        kwargs["passcode"] = passcode
    return _clean_connect_kwargs(kwargs)


def connect():
    try:
        from snowflake.connector import connect as snowflake_connect
        from snowflake.connector.errors import DatabaseError
    except ImportError as exc:
        raise RuntimeError(
            "snowflake-connector-python is required. pip install snowflake-connector-python"
        ) from exc

    def _mfa_help() -> RuntimeError:
        return RuntimeError(
            "Snowflake MFA (TOTP) is required. Either:\n"
            "  1. Regenerate SNOWFLAKE_TOKEN in Snowflake, or\n"
            "  2. Run with a fresh 6-digit code:\n"
            "     SNOWFLAKE_PASSCODE=123456 python3 -m RED_BUFF.v2.run_incremental_snowflake"
        )

    cfg = snowflake_config()
    kwargs = dict(build_connect_kwargs(cfg, mode="auto"))
    raw = (os.getenv("SNOWFLAKE_SOCKET_TIMEOUT") or "600").strip()
    try:
        kwargs["socket_timeout"] = int(raw)
    except ValueError:
        kwargs["socket_timeout"] = 600
    try:
        return snowflake_connect(**kwargs)
    except DatabaseError as exc:
        err = str(exc).lower()
        if "mfa" in err or "totp" in err:
            raise _mfa_help() from exc
        if "expired" in err and cfg.get("token") and cfg.get("password"):
            logger.warning("SNOWFLAKE_TOKEN expired — retrying with SNOWFLAKE_PASSWORD")
            try:
                return snowflake_connect(**build_connect_kwargs(cfg, mode="password"))
            except DatabaseError as exc2:
                if "mfa" in str(exc2).lower() or "totp" in str(exc2).lower():
                    raise _mfa_help() from exc2
                raise
        raise


def qualified_table(table: str) -> str:
    cfg = snowflake_config()
    return f'"{cfg["database"]}"."{cfg["schema"]}"."{table.upper()}"'


def ensure_schema(conn=None) -> None:
    """Create database, schema, and all RED_BUFF V2 tables."""
    own = conn is None
    if own:
        conn = connect()
    try:
        cfg = snowflake_config()
        db = cfg["database"]
        schema = cfg["schema"]
        qual = f"{db}.{schema}"
        cur = conn.cursor()
        try:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {qual}")
            cur.execute(f"USE DATABASE {db}")
            cur.execute(f"USE SCHEMA {schema}")
            for table in ALL_TABLES:
                cur.execute(table.ddl(f"{qual}.{table.name}"))
        finally:
            cur.close()
        logger.info("RED_BUFF V2 tables ensured in %s (%s)", qual, ", ".join(TABLE_NAMES))
    finally:
        if own:
            conn.close()


def synth_assets_exist(conn, organization_id: str) -> bool:
    sql = f"""
        SELECT COUNT(*) FROM {qualified_table("SYNTH_ASSETS_V2")}
        WHERE "ORGANIZATION_ID" = %s
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, (organization_id,))
        row = cur.fetchone()
        return bool(row and row[0] and int(row[0]) > 0)
    finally:
        cur.close()


def get_watermarks(conn, organization_id: str) -> Tuple[Optional[date], Optional[datetime]]:
    """Return (max daily production date, max sensor date) for V2 tables."""
    prod_sql = f"""
        SELECT MAX("DATE") AS MAX_DATE
        FROM {qualified_table("WELL_DAILY_PRODUCTION_V2")}
        WHERE "ORGANIZATION_ID" = %s
    """
    sensor_sql = f"""
        SELECT MAX("DATE") AS MAX_DATE
        FROM {qualified_table("WELL_SENSOR_READINGS_V2")}
        WHERE "ORGANIZATION_ID" = %s
    """
    cur = conn.cursor()
    try:
        cur.execute(prod_sql, (organization_id,))
        prod_row = cur.fetchone()
        last_date = prod_row[0] if prod_row and prod_row[0] else None
        if last_date is not None and hasattr(last_date, "date"):
            last_date = last_date.date()

        cur.execute(sensor_sql, (organization_id,))
        sensor_row = cur.fetchone()
        last_sensor = sensor_row[0] if sensor_row and sensor_row[0] else None
        if last_sensor is not None and hasattr(last_sensor, "date"):
            last_sensor = datetime.combine(last_sensor.date(), datetime.min.time())
        elif isinstance(last_sensor, date) and not isinstance(last_sensor, datetime):
            last_sensor = datetime.combine(last_sensor, datetime.min.time())
        return last_date, last_sensor
    finally:
        cur.close()


def append_dataframe(conn, table: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    upload = df.copy()
    upload.columns = [str(c).upper() for c in upload.columns]

    try:
        from snowflake.connector.pandas_tools import write_pandas
    except ImportError as exc:
        raise RuntimeError(
            "pandas support for Snowflake is required (snowflake-connector-python with pandas)"
        ) from exc

    cfg = snowflake_config()
    success, _nchunks, nrows, _ = write_pandas(
        conn,
        upload,
        table_name=table.upper(),
        database=cfg["database"],
        schema=cfg["schema"],
        quote_identifiers=True,
        auto_create_table=False,
        use_logical_type=True,
    )
    if not success:
        raise RuntimeError(f"write_pandas failed for table {table}")
    return int(nrows)


def append_dataframe_chunked(
    conn,
    table: str,
    df: pd.DataFrame,
    *,
    chunk_size: int = 50_000,
) -> int:
    if df is None or df.empty:
        return 0
    total = 0
    for start in range(0, len(df), chunk_size):
        total += append_dataframe(conn, table, df.iloc[start : start + chunk_size])
    return total


def save_registry(conn, organization_id: str, payload: dict) -> None:
    state_json = json.dumps(payload, default=str)
    sql = f"""
        MERGE INTO {qualified_table("SYNTH_PIPELINE_STATE_V2")} AS t
        USING (SELECT %s AS ORGANIZATION_ID, %s AS STATE_JSON) AS s
        ON t."ORGANIZATION_ID" = s.ORGANIZATION_ID
        WHEN MATCHED THEN UPDATE SET
            t."STATE_JSON" = s.STATE_JSON,
            t."UPDATED_AT" = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN INSERT ("ORGANIZATION_ID", "STATE_JSON")
            VALUES (s.ORGANIZATION_ID, s.STATE_JSON)
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, (organization_id, state_json))
    finally:
        cur.close()
    logger.info("Saved V2 pipeline state for org %s", organization_id)


def load_registry(conn, organization_id: str) -> Optional[dict]:
    sql = f"""
        SELECT "STATE_JSON"
        FROM {qualified_table("SYNTH_PIPELINE_STATE_V2")}
        WHERE "ORGANIZATION_ID" = %s
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, (organization_id,))
        row = cur.fetchone()
        if row and row[0]:
            return json.loads(row[0])
    finally:
        cur.close()
    return None


def clear_organization_data(conn, organization_id: str) -> dict:
    """Delete all RED_BUFF V2 synthetic rows for an organization."""
    counts = {}
    cur = conn.cursor()
    try:
        for table in TABLE_NAMES:
            sql = f'DELETE FROM {qualified_table(table)} WHERE "ORGANIZATION_ID" = %s'
            cur.execute(sql, (organization_id,))
            counts[table] = cur.rowcount
    finally:
        cur.close()
    logger.info("Cleared RED_BUFF V2 Snowflake data for org %s: %s", organization_id, counts)
    return counts
