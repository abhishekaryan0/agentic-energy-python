"""Snowflake connectivity and append helpers for RED_BUFF synthetic data."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = Path(__file__).resolve().parent / "red_bluff_pipeline_registry.json"
_LEGACY_REGISTRY_PATH = Path(__file__).resolve().parent / "red_bluff_well_registry.json"


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
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "SENSOR_DATA"),
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
    """
    Build Snowflake connector kwargs (same PAT/password rules as transformations).

    mode:
      auto — SNOWFLAKE_TOKEN (PAT) if set, else password (+ optional passcode)
      token — force programmatic access token
      password — force password (+ SNOWFLAKE_PASSCODE if MFA TOTP is on)
    """
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
        # Snowflake PAT: pass token as password (default authenticator).
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
    """Open a Snowflake connection; retries with password if PAT is expired."""
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
            "  1. Regenerate SNOWFLAKE_TOKEN in Snowflake (User → Programmatic Access Tokens), or\n"
            "  2. Run with a fresh 6-digit code:\n"
            "     SNOWFLAKE_PASSCODE=123456 python RED_BUFF/run_incremental_snowflake.py"
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
    db = cfg["database"]
    schema = cfg["schema"]
    return f'"{db}"."{schema}"."{table.upper()}"'


def ensure_core_tables(conn) -> None:
    """Create database, schema, and ML sensor tables (Snowflake-only path for RED_BUFF)."""
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

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {qual}.WELL_DAILY_PRODUCTION (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                DATE DATE NOT NULL,
                OIL_VOLUME FLOAT,
                GAS_VOLUME FLOAT,
                WATER_VOLUME FLOAT,
                BOE FLOAT,
                LIFT_TYPE VARCHAR(50),
                STATUS VARCHAR(50),
                NOTES VARCHAR(500),
                TEST_OIL FLOAT,
                TEST_WATER FLOAT,
                TEST_GAS FLOAT,
                FACILITY_ID VARCHAR(100),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                UNIQUE (WELL_ID, DATE)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {qual}.WELL_SENSOR_READINGS (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                LIFT_TYPE VARCHAR(50),
                WELL_STATUS VARCHAR(50),
                STROKES_PER_MINUTE FLOAT,
                PUMP_FILLAGE FLOAT,
                SURFACE_STROKE_LENGTH FLOAT,
                DOWNHOLE_GROSS_STROKE FLOAT,
                DOWNHOLE_NET_STROKE FLOAT,
                MOTOR_TEMP FLOAT,
                MOTOR_CURRENT FLOAT,
                DISCHARGE_PRESSURE FLOAT,
                VIBRATION_X FLOAT,
                DRIVE_FREQUENCY FLOAT,
                TUBING_PRESSURE FLOAT,
                CASING_PRESSURE FLOAT,
                INJECTION_RATE FLOAT,
                INJECTION_PRESSURE FLOAT,
                WELLHEAD_TEMP FLOAT,
                PUMP_INTAKE_PRESSURE FLOAT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
        )
    finally:
        cur.close()
    logger.info("Core Snowflake sensor tables ensured")


def ensure_schema(conn=None) -> None:
    """Create all Snowflake tables used by the RED_BUFF pipeline."""
    own = conn is None
    if own:
        conn = connect()
    try:
        ensure_core_tables(conn)
        ensure_synth_tables(conn)
    finally:
        if own:
            conn.close()


def ensure_synth_tables(conn) -> None:
    """Create Red Bluff synthetic master / invoice tables if missing."""
    ddl = [
        f"""
        CREATE TABLE IF NOT EXISTS {qualified_table("SYNTH_ASSETS")} (
            ORGANIZATION_ID VARCHAR(36) NOT NULL,
            ASSET_ID VARCHAR(100) NOT NULL,
            ASSET_TYPE VARCHAR(50),
            LIFT_TYPE VARCHAR(50),
            LEASE VARCHAR(50),
            BASIN VARCHAR(100),
            SUB_BASIN VARCHAR(100),
            STATE VARCHAR(10),
            COUNTY VARCHAR(100),
            REGION VARCHAR(100),
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {qualified_table("SYNTH_PERSONNEL")} (
            ORGANIZATION_ID VARCHAR(36) NOT NULL,
            ASSET_ID VARCHAR(100) NOT NULL,
            ROLE_KEY VARCHAR(50) NOT NULL,
            NAME VARCHAR(200),
            ROLE VARCHAR(100),
            EMAIL VARCHAR(200),
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID, ROLE_KEY)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {qualified_table("SYNTH_ASSET_DAILY_HISTORY")} (
            ORGANIZATION_ID VARCHAR(36) NOT NULL,
            ASSET_ID VARCHAR(100) NOT NULL,
            DATE DATE NOT NULL,
            ASSET_TYPE VARCHAR(50),
            LIFT_TYPE VARCHAR(50),
            RUN_STATUS VARCHAR(50),
            HISTORY_JSON VARCHAR,
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID, DATE)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {qualified_table("SYNTH_PIPELINE_STATE")} (
            ORGANIZATION_ID VARCHAR(36) NOT NULL,
            STATE_JSON VARCHAR,
            UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {qualified_table("SYNTH_INVOICES")} (
            ORGANIZATION_ID VARCHAR(36) NOT NULL,
            INVOICE_ID VARCHAR(50) NOT NULL,
            INVOICE_DATE DATE,
            ASSET_ID VARCHAR(100),
            CATEGORY VARCHAR(50),
            GL_CODE VARCHAR(20),
            SERVICE_DESCRIPTION VARCHAR(500),
            VENDOR VARCHAR(200),
            TOTAL_USD FLOAT,
            INVOICE_JSON VARCHAR,
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, INVOICE_ID)
        )
        """,
    ]
    cur = conn.cursor()
    try:
        for stmt in ddl:
            cur.execute(stmt)
    finally:
        cur.close()
    logger.info("Synthetic master tables ensured in Snowflake")


def synth_assets_exist(conn, organization_id: str) -> bool:
    sql = f"""
        SELECT COUNT(*) FROM {qualified_table("SYNTH_ASSETS")}
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
    """Return (max daily production date, max sensor timestamp) for an organization."""
    prod_sql = f"""
        SELECT MAX("DATE") AS MAX_DATE
        FROM {qualified_table("WELL_DAILY_PRODUCTION")}
        WHERE "ORGANIZATION_ID" = %s
    """
    sensor_sql = f"""
        SELECT MAX("TIMESTAMP") AS MAX_TS
        FROM {qualified_table("WELL_SENSOR_READINGS")}
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
        last_ts = sensor_row[0] if sensor_row and sensor_row[0] else None
        return last_date, last_ts
    finally:
        cur.close()


def append_dataframe(conn, table: str, df: pd.DataFrame) -> int:
    """Append rows to a Snowflake table. Column names must be uppercase in SF."""
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


def load_registry(conn, organization_id: str) -> Optional[dict]:
    """Load pipeline state from Snowflake (optional one-time migration from local JSON)."""
    import json

    sql = f"""
        SELECT "STATE_JSON"
        FROM {qualified_table("SYNTH_PIPELINE_STATE")}
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

    if _REGISTRY_PATH.is_file():
        payload = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        save_registry(conn, organization_id, payload)
        logger.info("Migrated local registry into Snowflake")
        return payload
    if _LEGACY_REGISTRY_PATH.is_file():
        legacy = json.loads(_LEGACY_REGISTRY_PATH.read_text(encoding="utf-8"))
        payload = {"wells": legacy.get("wells", []), "master_initialized": False}
        save_registry(conn, organization_id, payload)
        return payload
    return None


def save_registry(conn, organization_id: str, payload: dict) -> None:
    """Persist pipeline state to Snowflake only."""
    import json

    state_json = json.dumps(payload, default=str)
    sql = f"""
        MERGE INTO {qualified_table("SYNTH_PIPELINE_STATE")} AS t
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
    logger.info("Saved pipeline state to Snowflake for org %s", organization_id)


def clear_local_registry() -> None:
    for path in (_REGISTRY_PATH, _LEGACY_REGISTRY_PATH):
        if path.is_file():
            path.unlink()
            logger.info("Removed local registry file %s", path)


def clear_organization_data(conn, organization_id: str) -> dict:
    """Delete all Red Bluff synthetic rows for an organization."""
    tables = [
        "SYNTH_INVOICES",
        "SYNTH_ASSET_DAILY_HISTORY",
        "WELL_SENSOR_READINGS",
        "WELL_DAILY_PRODUCTION",
        "SYNTH_PERSONNEL",
        "SYNTH_ASSETS",
        "SYNTH_PIPELINE_STATE",
    ]
    counts = {}
    cur = conn.cursor()
    try:
        for table in tables:
            sql = f'DELETE FROM {qualified_table(table)} WHERE "ORGANIZATION_ID" = %s'
            cur.execute(sql, (organization_id,))
            counts[table] = cur.rowcount
    finally:
        cur.close()
    clear_local_registry()
    logger.info("Cleared Snowflake data for org %s: %s", organization_id, counts)
    return counts
