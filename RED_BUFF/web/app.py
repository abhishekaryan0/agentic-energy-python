"""FastAPI UI for Red Bluff Snowflake data generation."""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_WEB_DIR = Path(__file__).resolve().parent
_RED_BUFF_DIR = _WEB_DIR.parent
_PROJECT_ROOT = _RED_BUFF_DIR.parent

for p in (_PROJECT_ROOT, _RED_BUFF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline import generation_preview, reset_organization_data, run_red_buff_pipeline
from snowflake_io import connect, get_watermarks, qualified_table, snowflake_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Red Bluff Data Studio", version="1.0.0")
app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _org_id() -> str:
    oid = os.getenv("ORGANIZATION_ID", "").strip()
    if not oid:
        raise HTTPException(500, "ORGANIZATION_ID is not set in .env")
    return oid


def fetch_data_summary(organization_id: str) -> Dict[str, Any]:
    """Aggregate row counts and date ranges from Snowflake."""
    conn = connect()
    cur = conn.cursor()
    try:
        def _one(sql: str, params=()):
            cur.execute(sql, params)
            return cur.fetchone()

        prod = _one(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT "WELL_ID"),
                   MIN("DATE"), MAX("DATE")
            FROM {qualified_table("WELL_DAILY_PRODUCTION")}
            WHERE "ORGANIZATION_ID" = %s
            """,
            (organization_id,),
        )
        sensor = _one(
            f"""
            SELECT COUNT(*), MIN("TIMESTAMP"), MAX("TIMESTAMP")
            FROM {qualified_table("WELL_SENSOR_READINGS")}
            WHERE "ORGANIZATION_ID" = %s
            """,
            (organization_id,),
        )
        history = _one(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT "ASSET_ID"),
                   MIN("DATE"), MAX("DATE")
            FROM {qualified_table("SYNTH_ASSET_DAILY_HISTORY")}
            WHERE "ORGANIZATION_ID" = %s
            """,
            (organization_id,),
        )
        invoices = _one(
            f"""
            SELECT COUNT(*), MIN("INVOICE_DATE"), MAX("INVOICE_DATE")
            FROM {qualified_table("SYNTH_INVOICES")}
            WHERE "ORGANIZATION_ID" = %s
            """,
            (organization_id,),
        )
        assets = _one(
            f"""
            SELECT COUNT(*) FROM {qualified_table("SYNTH_ASSETS")}
            WHERE "ORGANIZATION_ID" = %s
            """,
            (organization_id,),
        )
        by_well = []
        cur.execute(
            f"""
            SELECT "WELL_ID", COUNT(*) AS n, MIN("DATE"), MAX("DATE")
            FROM {qualified_table("WELL_DAILY_PRODUCTION")}
            WHERE "ORGANIZATION_ID" = %s
            GROUP BY "WELL_ID"
            ORDER BY "WELL_ID"
            LIMIT 20
            """,
            (organization_id,),
        )
        for row in cur.fetchall():
            by_well.append(
                {
                    "well_id": row[0],
                    "daily_rows": int(row[1]),
                    "min_date": str(row[2]) if row[2] else None,
                    "max_date": str(row[3]) if row[3] else None,
                }
            )

        last_daily, last_sensor = get_watermarks(conn, organization_id)
        gen_preview = generation_preview(last_daily)

        def _fmt_dt(v):
            if v is None:
                return None
            return v.isoformat() if hasattr(v, "isoformat") else str(v)

        return {
            "organization_id": organization_id,
            "database": snowflake_config().get("database"),
            "schema": snowflake_config().get("schema"),
            "watermark_daily": _fmt_dt(last_daily),
            "watermark_sensor": _fmt_dt(last_sensor),
            "generation": gen_preview,
            "production": {
                "rows": int(prod[0] or 0),
                "wells": int(prod[1] or 0),
                "min_date": _fmt_dt(prod[2]),
                "max_date": _fmt_dt(prod[3]),
            },
            "sensor_readings": {
                "rows": int(sensor[0] or 0),
                "min_timestamp": _fmt_dt(sensor[1]),
                "max_timestamp": _fmt_dt(sensor[2]),
                "ratio_to_daily": round((sensor[0] or 0) / max(prod[0] or 1, 1), 2),
            },
            "master_history": {
                "rows": int(history[0] or 0),
                "assets": int(history[1] or 0),
                "min_date": _fmt_dt(history[2]),
                "max_date": _fmt_dt(history[3]),
            },
            "invoices": {
                "rows": int(invoices[0] or 0),
                "min_date": _fmt_dt(invoices[1]),
                "max_date": _fmt_dt(invoices[2]),
            },
            "asset_catalog": {"rows": int(assets[0] or 0)},
            "sample_wells": by_well,
        }
    finally:
        cur.close()
        conn.close()


def _fmt_window(dt) -> str:
    if dt is None:
        return "—"
    if hasattr(dt, "date"):
        return dt.date().isoformat()
    return str(dt)[:10]


def _run_job(job_id: str, organization_id: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.utcnow().isoformat() + "Z"

    try:
        result = run_red_buff_pipeline(
            organization_id=organization_id,
            ensure_tables=True,
        )
        summary = fetch_data_summary(organization_id)
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "status": "completed",
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                    "run": {
                        "window_start": _fmt_window(result["window_start"]),
                        "window_end": _fmt_window(result["window_end"]),
                        "is_full_backfill": result.get("is_full_backfill", False),
                        "pending_days": result.get("pending_days"),
                        "last_daily": result.get("last_daily"),
                        "daily_rows": result["daily_rows"],
                        "sensor_rows": result["sensor_rows"],
                        "master_history_rows": result["master_history_rows"],
                        "invoice_rows": result["invoice_rows"],
                        "message": result.get("message"),
                    },
                    "summary": summary,
                }
            )
    except Exception as exc:
        logger.exception("Generation failed")
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "status": "failed",
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                    "error": str(exc),
                }
            )


@app.get("/")
async def index():
    return FileResponse(_WEB_DIR / "static" / "index.html")


@app.get("/api/health")
async def health():
    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        return {"ok": True, "snowflake": "connected"}
    except Exception as exc:
        return {"ok": False, "snowflake": "error", "detail": str(exc)}


@app.get("/api/stats")
async def stats():
    try:
        return fetch_data_summary(_org_id())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/generate")
async def generate():
    """Append missing days through today (full backfill if empty)."""
    org_id = _org_id()
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, org_id),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/reset")
async def reset():
    """Delete all synthetic Snowflake data for this organization."""
    org_id = _org_id()
    try:
        cleared = reset_organization_data(org_id, ensure_tables=True)
        return {"ok": True, "cleared": cleared}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job
