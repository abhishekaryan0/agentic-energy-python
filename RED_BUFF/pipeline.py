"""
Full Red Bluff synthetic pipeline → Snowflake only.

Generate button behavior:
  - No data yet: full history per well (spud → today), like the original scripts.
  - Has data: append from (last daily date + 1) → today only.
"""

from __future__ import annotations

import importlib.util
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_RED_BUFF_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RED_BUFF_DIR.parent

import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_RED_BUFF_DIR) not in sys.path:
    sys.path.insert(0, str(_RED_BUFF_DIR))

from column_maps import (
    assets_catalog_to_dataframe,
    history_to_daily_production,
    history_to_sensor_readings,
    invoices_to_dataframe,
    master_history_to_dataframe,
    personnel_to_dataframe,
)
from invoice_gen import generate_supporting_invoices, generate_well_invoices
from master_assets import (
    SUPPORTING_ASSET_DEFS,
    apply_clean_well_history,
    build_supporting_asset_catalog,
    compute_lease_starts,
    dates_in_window,
    generate_supporting_history,
    get_lease_location,
    get_personnel,
    select_clean_well_ids,
)
from snowflake_io import (
    append_dataframe,
    clear_organization_data,
    connect,
    ensure_schema,
    get_watermarks,
    load_registry,
    save_registry,
    synth_assets_exist,
)

logger = logging.getLogger(__name__)

_NUM_WELLS = 60
_EARLIEST_SIM_DATE = datetime(2012, 1, 1)


def _load_twin_engine():
    legacy_path = _RED_BUFF_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("red_buff_twin_engine", legacy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def _filter_history(history: list, start: datetime, end: datetime) -> list:
    return [
        row
        for row in history
        if start <= _parse_date(row["date"]) <= end
    ]


def _today() -> datetime:
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _simulation_cap(config_end: datetime) -> datetime:
    return max(config_end, _today())


def _compute_window_to_today(
    last_daily_date,
    config_end: datetime,
) -> tuple[datetime, datetime, bool]:
    """
    Return (window_start, window_end, is_full_backfill).

    full backfill: each well loads from its spud date → today.
    incremental: last_daily + 1 → today.
    """
    window_end = _simulation_cap(config_end)
    if last_daily_date is None:
        return _EARLIEST_SIM_DATE, window_end, True

    last_dt = datetime.combine(last_daily_date, datetime.min.time())
    window_start = last_dt + timedelta(days=1)
    return window_start, window_end, False


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def generation_preview(last_daily_date) -> dict:
    """
    Describe what the next Generate click will do (append only; never deletes).
    Example: last_daily=16th, today=19th → pending_days=3 (17th–19th).
    """
    today = _today().date()
    last_d = _as_date(last_daily_date)

    if last_d is None:
        return {
            "mode": "full_backfill",
            "last_daily": None,
            "today": today.isoformat(),
            "pending_days": None,
            "window_start": None,
            "window_end": today.isoformat(),
            "message": "No data yet — first run loads full history.",
        }

    pending = (today - last_d).days
    next_d = last_d + timedelta(days=1)

    if pending <= 0:
        return {
            "mode": "incremental",
            "last_daily": last_d.isoformat(),
            "today": today.isoformat(),
            "pending_days": 0,
            "window_start": None,
            "window_end": today.isoformat(),
            "message": f"Up to date through {last_d.isoformat()}.",
        }

    return {
        "mode": "incremental",
        "last_daily": last_d.isoformat(),
        "today": today.isoformat(),
        "pending_days": pending,
        "window_start": next_d.isoformat(),
        "window_end": today.isoformat(),
        "message": f"Incremental: {pending} day(s) ({next_d.isoformat()} → {today.isoformat()}).",
    }


def _init_registry(engine, wells: List[dict]) -> dict:
    for w in wells:
        w["spud_date"] = _parse_date(w["spud_date"])
    clean_ids = select_clean_well_ids(wells)
    lease_starts = compute_lease_starts(wells)
    supporting = build_supporting_asset_catalog(lease_starts)
    return {
        "wells": [
            {
                **w,
                "spud_date": w["spud_date"].strftime("%Y-%m-%d")
                if isinstance(w["spud_date"], datetime)
                else w["spud_date"],
            }
            for w in wells
        ],
        "clean_well_ids": clean_ids,
        "lease_starts": lease_starts,
        "supporting_assets": supporting,
        "master_initialized": True,
    }


def _build_asset_catalog(wells: List[dict], supporting: List[dict]) -> List[dict]:
    catalog = []
    for w in wells:
        lease = w["lease"]
        catalog.append(
            {
                "asset_id": w["asset_id"],
                "asset_type": "Well",
                "lift_type": w["lift"],
                "lease": lease,
                "location": get_lease_location(lease),
            }
        )
    for s in supporting:
        lease = s["lease"]
        catalog.append(
            {
                "asset_id": s["asset_id"],
                "asset_type": s["asset_type"],
                "lift_type": None,
                "lease": lease,
                "location": get_lease_location(lease),
            }
        )
    return catalog


def _seed_master_metadata(conn, organization_id: str, catalog: List[dict]) -> None:
    if synth_assets_exist(conn, organization_id):
        logger.info("Asset catalog already in Snowflake — skipping metadata seed")
        return

    assets_df = assets_catalog_to_dataframe(catalog, organization_id)
    append_dataframe(conn, "SYNTH_ASSETS", assets_df)

    personnel_frames = []
    for item in catalog:
        aid = item["asset_id"]
        lease = item["lease"]
        personnel_frames.append(personnel_to_dataframe(aid, get_personnel(lease), organization_id))

    if personnel_frames:
        append_dataframe(conn, "SYNTH_PERSONNEL", pd.concat(personnel_frames, ignore_index=True))

    logger.info("Seeded %d assets and personnel into Snowflake", len(catalog))


def reset_organization_data(organization_id: str, ensure_tables: bool = True) -> dict:
    """Delete all synthetic Snowflake data for this org and local registry."""
    conn = connect()
    try:
        if ensure_tables:
            ensure_schema(conn)
        return clear_organization_data(conn, organization_id)
    finally:
        conn.close()


def run_red_buff_pipeline(
    organization_id: str,
    ensure_tables: bool = True,
) -> dict:
    """Generate missing days through today (full backfill if empty)."""
    engine = _load_twin_engine()
    config_end = engine.END_DATE

    conn = connect()
    try:
        if ensure_tables:
            ensure_schema(conn)

        last_daily, _last_sensor = get_watermarks(conn, organization_id)
        preview = generation_preview(last_daily)
        window_start, window_end, is_full = _compute_window_to_today(last_daily, config_end)

        if not is_full and window_start > window_end:
            return {
                "window_start": window_start,
                "window_end": window_end,
                "simulation_end_date": engine.END_DATE,
                "is_full_backfill": False,
                "pending_days": 0,
                "last_daily": preview.get("last_daily"),
                "daily_rows": 0,
                "sensor_rows": 0,
                "master_history_rows": 0,
                "invoice_rows": 0,
                "message": preview["message"],
            }

        sim_end = max(config_end, window_end)
        engine.END_DATE = sim_end

        mode = "full backfill (spud → today)" if is_full else "incremental (last date + 1 → today)"
        logger.info(
            "Pipeline %s: %s → %s (watermark=%s)",
            mode,
            window_start.date() if not is_full else "per-well spud",
            window_end.date(),
            last_daily,
        )

        registry = load_registry(conn, organization_id)
        if registry is None or not registry.get("wells"):
            logger.info("Initializing pipeline registry (%d wells)", _NUM_WELLS)
            wells = engine.build_wells()
            registry = _init_registry(engine, wells)
            save_registry(conn, organization_id, registry)

        wells = registry["wells"]
        for w in wells:
            w["spud_date"] = _parse_date(w["spud_date"])

        clean_ids = set(registry.get("clean_well_ids") or [])
        supporting = registry.get("supporting_assets") or []
        lease_starts = registry.get("lease_starts") or compute_lease_starts(wells)

        catalog = _build_asset_catalog(wells, supporting)
        _seed_master_metadata(conn, organization_id, catalog)

        daily_frames = []
        sensor_frames = []
        master_history_frames = []
        invoice_batches: List[dict] = []

        for i, w in enumerate(wells):
            spud = _parse_date(w["spud_date"])
            chunk_start = spud if is_full else max(spud, window_start)
            if chunk_start > window_end:
                continue

            aid = w["asset_id"]
            lift = w["lift"]
            logger.info("[%d/%d] Well %s", i + 1, len(wells), aid)

            history = engine.simulate_well(
                aid,
                lift,
                spud,
                w["base_rate"],
                w["gross_base"],
                w["net_base"],
            )
            chunk = _filter_history(history, chunk_start, window_end)
            if not chunk:
                continue

            if aid in clean_ids:
                chunk = apply_clean_well_history(chunk, lift)

            daily_frames.append(history_to_daily_production(chunk, w, organization_id))
            sensor_frames.append(history_to_sensor_readings(chunk, w, organization_id))
            master_history_frames.append(
                master_history_to_dataframe(aid, "Well", lift, chunk, organization_id)
            )
            invoice_batches.extend(
                generate_well_invoices(
                    aid,
                    lift,
                    chunk,
                    skip_workover=(aid in clean_ids),
                )
            )

        for s in supporting:
            aid = s["asset_id"]
            asset_type = s["asset_type"]
            lease = s["lease"]
            hist_start = _parse_date(s.get("history_start") or lease_starts[lease])
            sup_start = hist_start if is_full else max(hist_start, window_start)
            date_strs = dates_in_window(sup_start, engine.END_DATE, sup_start, window_end)
            if not date_strs:
                continue

            builder_name = next(b for t, _suffix, b in SUPPORTING_ASSET_DEFS if t == asset_type)
            logger.info("Supporting asset %s (%d days)", aid, len(date_strs))
            chunk = generate_supporting_history(asset_type, builder_name, date_strs)
            master_history_frames.append(
                master_history_to_dataframe(aid, asset_type, None, chunk, organization_id)
            )
            invoice_batches.extend(generate_supporting_invoices(aid, asset_type, chunk))

        result = {
            "window_start": window_start,
            "window_end": window_end,
            "simulation_end_date": engine.END_DATE,
            "is_full_backfill": is_full,
            "pending_days": preview.get("pending_days"),
            "last_daily": preview.get("last_daily"),
            "daily_rows": 0,
            "sensor_rows": 0,
            "master_history_rows": 0,
            "invoice_rows": 0,
            "message": None,
        }

        if daily_frames:
            result["daily_rows"] = append_dataframe(
                conn, "WELL_DAILY_PRODUCTION", pd.concat(daily_frames, ignore_index=True)
            )
            result["sensor_rows"] = append_dataframe(
                conn, "WELL_SENSOR_READINGS", pd.concat(sensor_frames, ignore_index=True)
            )

        if master_history_frames:
            result["master_history_rows"] = append_dataframe(
                conn,
                "SYNTH_ASSET_DAILY_HISTORY",
                pd.concat(master_history_frames, ignore_index=True),
            )

        if invoice_batches:
            inv_df = invoices_to_dataframe(invoice_batches, organization_id)
            result["invoice_rows"] = append_dataframe(conn, "SYNTH_INVOICES", inv_df)

        if not daily_frames and not master_history_frames:
            result["message"] = "No rows generated for this window."

        logger.info(
            "Pipeline complete: daily=%s sensor=%s history=%s invoices=%s",
            result["daily_rows"],
            result["sensor_rows"],
            result["master_history_rows"],
            result["invoice_rows"],
        )
        return result
    finally:
        conn.close()
