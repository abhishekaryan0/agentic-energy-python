"""
Full South Texas Energy Partners synthetic pipeline → Snowflake.

Schema: AT_RISK_ASSETS.SOUTH_TEXAS_SENSOR_DATA (isolated from RED_BUFF / APEX).

Modes:
  - Default: incremental generation through today (like RED_BUFF).
  - --from-master: bulk load South_Texas_Energy_Partners_Master.json into Snowflake.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_STEP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _STEP_DIR.parent

import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_STEP_DIR) not in sys.path:
    sys.path.insert(0, str(_STEP_DIR))

from column_maps import (
    asset_record_to_catalog_row,
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
    get_lease_area,
    get_lease_location,
    get_personnel,
    select_clean_well_ids,
)
from config import LEGACY_ORGANIZATION_LABEL, NUM_WELLS, STEP_ORGANIZATION_ID
from snowflake_io import (
    append_dataframe,
    append_dataframe_chunked,
    clear_organization_data,
    connect,
    ensure_schema,
    get_watermarks,
    load_registry,
    save_registry,
    snowflake_config,
    synth_assets_exist,
)

logger = logging.getLogger(__name__)

_EARLIEST_SIM_DATE = datetime(2013, 1, 1)
_MASTER_JSON = _STEP_DIR / "South_Texas_Energy_Partners_Master.json"


def _load_twin_engine():
    legacy_path = _STEP_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("step_twin_engine", legacy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def _filter_history(history: list, start: datetime, end: datetime) -> list:
    return [row for row in history if start <= _parse_date(row["date"]) <= end]


def _today() -> datetime:
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _simulation_cap(config_end: datetime) -> datetime:
    return max(config_end, _today())


def _compute_window_to_today(last_daily_date, config_end: datetime) -> tuple[datetime, datetime, bool]:
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
                "area": get_lease_area(lease),
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
                "area": get_lease_area(lease),
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
    """Delete synthetic Snowflake rows for this org (and legacy label if different)."""
    conn = connect()
    try:
        if ensure_tables:
            ensure_schema(conn)
        totals = clear_organization_data(conn, organization_id)
        if organization_id != LEGACY_ORGANIZATION_LABEL:
            legacy = clear_organization_data(conn, LEGACY_ORGANIZATION_LABEL)
            for table, count in legacy.items():
                totals[table] = totals.get(table, 0) + count
        return totals
    finally:
        conn.close()


def load_master_json(
    master_path: Optional[Path] = None,
) -> dict:
    path = master_path or _MASTER_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Master JSON not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def master_to_dataframes(data: dict, organization_id: str) -> dict:
    """Flatten master JSON into Snowflake-ready dataframes."""
    assets = data.get("assets", {})
    assets_rows = []
    personnel_frames = []
    daily_frames = []
    sensor_frames = []
    history_frames = []
    invoice_batches = []

    for asset_id, asset in assets.items():
        assets_rows.append(asset_record_to_catalog_row(asset_id, asset, organization_id))

        personnel_frames.append(
            personnel_to_dataframe(
                asset_id, asset.get("assigned_personnel", {}), organization_id
            )
        )

        asset_type = asset.get("type")
        lift = asset.get("lift")
        history = asset.get("history") or []

        if asset_type == "Well" and lift:
            well = {"asset_id": asset_id, "lease": asset_id.split("-")[0], "lift": lift}
            daily_frames.append(history_to_daily_production(history, well, organization_id))
            sensor_frames.append(history_to_sensor_readings(history, well, organization_id))

        if history:
            history_frames.append(
                master_history_to_dataframe(
                    asset_id, asset_type, lift, history, organization_id
                )
            )

        for inv in asset.get("financial_ledger") or []:
            invoice_batches.append({**inv, "asset_id": asset_id})

    result = {
        "assets": pd.DataFrame(assets_rows) if assets_rows else pd.DataFrame(),
        "personnel": pd.concat(personnel_frames, ignore_index=True)
        if personnel_frames
        else pd.DataFrame(),
        "daily": pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        "sensor": pd.concat(sensor_frames, ignore_index=True) if sensor_frames else pd.DataFrame(),
        "history": pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame(),
        "invoices": invoices_to_dataframe(invoice_batches, organization_id)
        if invoice_batches
        else pd.DataFrame(),
    }
    return result


def load_master_to_snowflake(
    organization_id: str,
    *,
    master_path: Optional[Path] = None,
    ensure_tables: bool = True,
    chunk_size: int = 50_000,
) -> dict:
    """Bulk load the full master JSON into Snowflake (SOUTH_TEXAS_SENSOR_DATA schema)."""
    data = load_master_json(master_path)
    frames = master_to_dataframes(data, organization_id)

    conn = connect()
    try:
        if ensure_tables:
            ensure_schema(conn)

        cfg = snowflake_config()
        logger.info(
            "Loading master JSON into %s.%s for org %s",
            cfg["database"],
            cfg["schema"],
            organization_id,
        )

        result = {
            "mode": "from_master",
            "schema": cfg["schema"],
            "daily_rows": 0,
            "sensor_rows": 0,
            "asset_rows": 0,
            "personnel_rows": 0,
            "master_history_rows": 0,
            "invoice_rows": 0,
            "message": None,
        }

        logger.info("Loading assets (%d rows)...", len(frames["assets"]))
        result["asset_rows"] = append_dataframe(conn, "SYNTH_ASSETS", frames["assets"])
        logger.info("Loading personnel (%d rows)...", len(frames["personnel"]))
        result["personnel_rows"] = append_dataframe(conn, "SYNTH_PERSONNEL", frames["personnel"])
        logger.info("Loading daily production (%d rows)...", len(frames["daily"]))
        result["daily_rows"] = append_dataframe_chunked(
            conn, "WELL_DAILY_PRODUCTION", frames["daily"], chunk_size=chunk_size
        )
        logger.info("Loading sensor readings (%d rows)...", len(frames["sensor"]))
        result["sensor_rows"] = append_dataframe_chunked(
            conn, "WELL_SENSOR_READINGS", frames["sensor"], chunk_size=chunk_size
        )
        logger.info("Loading master history (%d rows)...", len(frames["history"]))
        result["master_history_rows"] = append_dataframe_chunked(
            conn, "SYNTH_ASSET_DAILY_HISTORY", frames["history"], chunk_size=chunk_size
        )
        logger.info("Loading invoices (%d rows)...", len(frames["invoices"]))
        result["invoice_rows"] = append_dataframe_chunked(
            conn, "SYNTH_INVOICES", frames["invoices"], chunk_size=chunk_size
        )
        logger.info(
            "Master load complete: daily=%s sensor=%s history=%s invoices=%s",
            result["daily_rows"],
            result["sensor_rows"],
            result["master_history_rows"],
            result["invoice_rows"],
        )
        return result
    finally:
        conn.close()


def run_step_pipeline(
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
            logger.info("Initializing pipeline registry (%d wells)", NUM_WELLS)
            lease_ownership = engine.assign_lease_ownership()
            wells = engine.build_wells(lease_ownership)
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

        for i, w in enumerate(wells):
            spud = _parse_date(w["spud_date"])
            chunk_start = spud if is_full else max(spud, window_start)
            if chunk_start > window_end:
                continue

            aid = w["asset_id"]
            lift = w["lift"]
            wi = (w.get("ownership") or {}).get("working_interest", 1.0)
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

            result["daily_rows"] += append_dataframe(
                conn,
                "WELL_DAILY_PRODUCTION",
                history_to_daily_production(chunk, w, organization_id),
            )
            result["sensor_rows"] += append_dataframe(
                conn,
                "WELL_SENSOR_READINGS",
                history_to_sensor_readings(chunk, w, organization_id),
            )
            result["master_history_rows"] += append_dataframe(
                conn,
                "SYNTH_ASSET_DAILY_HISTORY",
                master_history_to_dataframe(aid, "Well", lift, chunk, organization_id),
            )
            inv_df = invoices_to_dataframe(
                generate_well_invoices(
                    aid,
                    lift,
                    chunk,
                    skip_workover=(aid in clean_ids),
                    working_interest=wi,
                ),
                organization_id,
            )
            result["invoice_rows"] += append_dataframe(conn, "SYNTH_INVOICES", inv_df)

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
            result["master_history_rows"] += append_dataframe(
                conn,
                "SYNTH_ASSET_DAILY_HISTORY",
                master_history_to_dataframe(aid, asset_type, None, chunk, organization_id),
            )
            inv_df = invoices_to_dataframe(
                generate_supporting_invoices(aid, asset_type, chunk),
                organization_id,
            )
            result["invoice_rows"] += append_dataframe(conn, "SYNTH_INVOICES", inv_df)

        if not result["daily_rows"] and not result["master_history_rows"]:
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
