"""
RED_BUFF V2 synthetic pipeline → Snowflake (*_V2 tables).

Same pattern as v1 RED_BUFF/pipeline.py:
  - Default: generate with local generate_digital_twin.py → write *_V2 tables
  - organization_id (RED_BUFF_ORGANIZATION_ID) on every row

Uses local generate_digital_twin.py (RBNR V2 logic). Does not use RBNR_V2_Master_build.py.
"""

from __future__ import annotations

import importlib.util
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .config import LEGACY_ORGANIZATION_LABEL, NUM_WELLS, RED_BUFF_ORGANIZATION_ID
from .invoice_gen import (
    generate_supporting_invoices,
    generate_well_invoices,
    generate_well_revenue,
)
from .master_assets import (
    SUPPORTING_ASSET_DEFS,
    build_supporting_asset_catalog,
    compute_lease_starts,
    generate_supporting_history,
    get_personnel,
    select_clean_well_ids,
)
from .snowflake_io import (
    append_dataframe,
    clear_organization_data,
    connect,
    ensure_schema,
    get_watermarks,
    load_registry,
    save_registry,
    snowflake_config,
    synth_assets_exist,
)
from .tables import (
    SynthAssetDailyHistoryV2,
    SynthAssetsV2,
    SynthInvoicesV2,
    SynthPersonnelV2,
    SynthRevenueV2,
    WellDailyProductionV2,
    WellSensorReadingsV2,
)

logger = logging.getLogger(__name__)

_V2_DIR = Path(__file__).resolve().parent
_EARLIEST_SIM_DATE = datetime(2008, 1, 1)


def _load_twin_engine():
    """Load local generate_digital_twin.py (same pattern as v1)."""
    path = _V2_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("red_buff_v2_twin_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d")


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
    return last_dt + timedelta(days=1), window_end, False


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _parse_date(value).date()


def generation_preview(last_daily_date) -> dict:
    today = _today().date()
    last_d = _as_date(last_daily_date)
    if last_d is None:
        return {
            "mode": "full_backfill",
            "last_daily": None,
            "today": today.isoformat(),
            "pending_days": None,
            "message": "No V2 data yet — first run loads full history.",
        }
    pending = (today - last_d).days
    if pending <= 0:
        return {
            "mode": "incremental",
            "last_daily": last_d.isoformat(),
            "today": today.isoformat(),
            "pending_days": 0,
            "message": f"Up to date through {last_d.isoformat()}.",
        }
    next_d = last_d + timedelta(days=1)
    return {
        "mode": "incremental",
        "last_daily": last_d.isoformat(),
        "today": today.isoformat(),
        "pending_days": pending,
        "message": f"Incremental: {pending} day(s) ({next_d.isoformat()} → {today.isoformat()}).",
    }


def _stamp_org(rows: List[dict], organization_id: str) -> List[dict]:
    out = []
    for row in rows:
        r = dict(row)
        r["organization_id"] = organization_id
        out.append(r)
    return out


def _init_registry(engine) -> dict:
    wells = engine.build_wells()
    lease_starts = compute_lease_starts(wells)
    supporting = build_supporting_asset_catalog(lease_starts, twin=engine)
    return {
        "version": "v2",
        "wells": wells,
        "clean_well_ids": select_clean_well_ids(wells),
        "lease_starts": lease_starts,
        "supporting_assets": supporting,
        "master_initialized": True,
    }


def _seed_master_metadata(conn, organization_id: str, registry: dict, engine) -> None:
    if synth_assets_exist(conn, organization_id):
        logger.info("SYNTH_ASSETS_V2 already seeded — skipping")
        return

    assets_payload = {}
    for w in registry["wells"]:
        lease = w["lease"]
        info = engine.LEASES[lease]
        assets_payload[w["asset_id"]] = {
            "type": "Well",
            "lift": w["lift"],
            "area": w.get("area") or info["sub_basin"],
            "lease_type": w.get("lease_type") or info["lease_type"],
            "working_interest": w.get("working_interest", info["wi"]),
            "net_revenue_interest": w.get("net_revenue_interest", info["nri"]),
            "wellbore_profile": w.get("wellbore_profile"),
            "lateral_length_ft": w.get("lateral_length_ft"),
            "location": engine.make_loc(info),
            "assigned_personnel": get_personnel(lease, twin=engine),
        }
    for s in registry["supporting_assets"]:
        lease = s["lease"]
        info = engine.LEASES[lease]
        assets_payload[s["asset_id"]] = {
            "type": s["asset_type"],
            "lift": None,
            "area": s.get("area") or info["sub_basin"],
            "lease_type": s.get("lease_type") or info["lease_type"],
            "working_interest": s.get("working_interest", info["wi"]),
            "net_revenue_interest": s.get("net_revenue_interest"),
            "location": engine.make_loc(info),
            "assigned_personnel": get_personnel(lease, twin=engine),
            "pump_type": s.get("swd_pump") if s["asset_type"] == "SWD Well" else None,
        }

    append_dataframe(
        conn, "SYNTH_ASSETS_V2", SynthAssetsV2.assets_to_dataframe(assets_payload, organization_id)
    )
    append_dataframe(
        conn,
        "SYNTH_PERSONNEL_V2",
        SynthPersonnelV2.assets_personnel_to_dataframe(assets_payload, organization_id),
    )
    logger.info("Seeded %d assets + personnel (*_V2)", len(assets_payload))


def reset_organization_data(organization_id: str, ensure_tables: bool = True) -> dict:
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


def run_step_v2_pipeline(
    organization_id: str = RED_BUFF_ORGANIZATION_ID,
    ensure_tables: bool = True,
    *,
    well_offset: int = 0,
    well_limit: Optional[int] = None,
    force_full_backfill: bool = False,
    include_supporting: Optional[bool] = None,
) -> dict:
    """Generate missing V2 days through today → Snowflake *_V2 (same as v1 run_step_pipeline).

    Lambda full-load helpers:
      well_offset / well_limit — process a slice of wells (avoids 15m timeout)
      force_full_backfill — ignore watermark; generate spud→end for the slice
      include_supporting — default: only on the last well slice
    """
    engine = _load_twin_engine()
    # stamp() / invoices use tenant UUID, not the legacy label
    engine.ORGANIZATION_ID = organization_id
    config_end = engine.END_DATE

    conn = connect()
    try:
        if ensure_tables:
            ensure_schema(conn)

        last_daily, _ = get_watermarks(conn, organization_id)
        preview = generation_preview(last_daily)
        window_start, window_end, is_full = _compute_window_to_today(last_daily, config_end)

        if force_full_backfill:
            is_full = True
            window_start = _EARLIEST_SIM_DATE
            window_end = _simulation_cap(config_end)

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
                "revenue_rows": 0,
                "well_offset": well_offset,
                "wells_processed": 0,
                "wells_total": 0,
                "message": preview["message"],
            }

        sim_end = max(config_end, window_end)
        engine.END_DATE = sim_end

        mode = "full backfill (spud → today)" if is_full else "incremental (last date + 1 → today)"
        logger.info(
            "RED_BUFF V2 %s: %s → %s (watermark=%s, org=%s, offset=%s, limit=%s)",
            mode,
            window_start.date() if not is_full else "per-well spud",
            window_end.date(),
            last_daily,
            organization_id,
            well_offset,
            well_limit,
        )

        registry = load_registry(conn, organization_id)
        if registry is None or not registry.get("wells"):
            logger.info("Initializing V2 registry (%d wells)", NUM_WELLS)
            registry = _init_registry(engine)
            save_registry(conn, organization_id, registry)

        wells = registry["wells"]
        for w in wells:
            w["spud_date"] = _parse_date(w["spud_date"])

        supporting = registry.get("supporting_assets") or []
        lease_starts = registry.get("lease_starts") or compute_lease_starts(
            [{**w, "spud_date": w["spud_date"].strftime("%Y-%m-%d")} for w in wells]
        )
        _seed_master_metadata(conn, organization_id, registry, engine)

        end = len(wells) if well_limit is None else min(len(wells), well_offset + max(0, well_limit))
        start = max(0, min(well_offset, len(wells)))
        well_batch = wells[start:end]
        do_supporting = (
            include_supporting
            if include_supporting is not None
            else (end >= len(wells) and (is_full or force_full_backfill or start == 0))
        )
        # Incremental daily runs always refresh supporting for the window.
        if not is_full and not force_full_backfill:
            do_supporting = True

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
            "revenue_rows": 0,
            "well_offset": start,
            "wells_processed": 0,
            "wells_total": len(wells),
            "next_well_offset": end if end < len(wells) else None,
            "supporting_loaded": False,
            "message": None,
        }

        wells_only: Dict[str, dict] = {}

        for i, w in enumerate(well_batch):
            global_i = start + i
            spud = _parse_date(w["spud_date"])
            chunk_start = spud if is_full else max(spud, window_start)
            if chunk_start > window_end:
                continue

            aid = w["asset_id"]
            lift = w["lift"]
            is_clean = bool(w.get("is_clean"))
            wi = float(w.get("working_interest") or 1.0)
            nri = float(w.get("net_revenue_interest") or wi * 0.75)
            area = w.get("area") or engine.LEASES[w["lease"]]["area"]
            logger.info("[%d/%d] Well %s (%s)", global_i + 1, len(wells), aid, lift)

            lateral_ft = int(w.get("lateral_length_ft") or 0)
            history, shutin_ev, _pump = engine.simulate_well(
                aid, lift, spud, is_clean, lateral_ft=lateral_ft
            )
            wells_only[aid] = {"type": "Well", "lift": lift, "history": history}
            chunk = _stamp_org(_filter_history(history, chunk_start, window_end), organization_id)
            if not chunk:
                continue

            result["daily_rows"] += append_dataframe(
                conn,
                "WELL_DAILY_PRODUCTION_V2",
                WellDailyProductionV2.history_to_dataframe(
                    chunk, asset_id=aid, lift_type=lift, organization_id=organization_id
                ),
            )
            result["sensor_rows"] += append_dataframe(
                conn,
                "WELL_SENSOR_READINGS_V2",
                WellSensorReadingsV2.history_to_dataframe(
                    chunk, asset_id=aid, lift_type=lift, organization_id=organization_id
                ),
            )
            result["master_history_rows"] += append_dataframe(
                conn,
                "SYNTH_ASSET_DAILY_HISTORY_V2",
                SynthAssetDailyHistoryV2.history_to_dataframe(
                    chunk,
                    asset_id=aid,
                    asset_type="Well",
                    lift_type=lift,
                    organization_id=organization_id,
                ),
            )

            shutin_chunk = (
                shutin_ev
                if is_full
                else [e for e in shutin_ev if chunk_start <= _parse_date(e["date"]) <= window_end]
            )
            inv = generate_well_invoices(
                aid, lift, chunk, shutin_chunk, working_interest=wi
            )
            inv = _stamp_org(inv, organization_id)
            result["invoice_rows"] += append_dataframe(
                conn, "SYNTH_INVOICES_V2", SynthInvoicesV2.invoices_to_dataframe(inv, organization_id)
            )

            if is_full:
                rev = generate_well_revenue(
                    aid, chunk, net_revenue_interest=nri
                )
                rev = _stamp_org(rev, organization_id)
                result["revenue_rows"] += append_dataframe(
                    conn,
                    "SYNTH_REVENUE_V2",
                    SynthRevenueV2.revenue_to_dataframe(rev, organization_id),
                )
            result["wells_processed"] += 1

        if do_supporting:
            # Supporting assets are production-coupled across the full lease set.
            if len(wells_only) < len(wells):
                logger.info(
                    "Regenerating all %d wells in-memory for supporting-asset coupling",
                    len(wells),
                )
                wells_only = {}
                for w in wells:
                    spud = _parse_date(w["spud_date"])
                    hist, _shutin, _pump = engine.simulate_well(
                        w["asset_id"],
                        w["lift"],
                        spud,
                        bool(w.get("is_clean")),
                        lateral_ft=int(w.get("lateral_length_ft") or 0),
                    )
                    wells_only[w["asset_id"]] = {
                        "type": "Well",
                        "lift": w["lift"],
                        "history": hist,
                    }

            if wells_only:
                all_dates = sorted(
                    {d["date"] for v in wells_only.values() for d in v["history"]}
                )
                sd = datetime.strptime(all_dates[0], "%Y-%m-%d")
                ed = datetime.strptime(all_dates[-1], "%Y-%m-%d")
                # Group supporting by lease so Tank feeds SWD.
                by_lease = {}
                for s in supporting:
                    by_lease.setdefault(s["lease"], []).append(s)
                for lease, items in by_lease.items():
                    order = {"Facility": 0, "Compressor": 1, "Tank Battery": 2, "SWD Well": 3}
                    items = sorted(items, key=lambda x: order.get(x["asset_type"], 9))
                    tank_hist = None
                    info = engine.LEASES[lease]
                    for s in items:
                        aid = s["asset_id"]
                        asset_type = s["asset_type"]
                        wi = float(s.get("working_interest") or info["wi"])
                        hist_start = _parse_date(s.get("history_start") or lease_starts[lease])
                        sup_start = hist_start if is_full else max(hist_start, window_start)
                        if sup_start > window_end:
                            continue
                        logger.info("Supporting %s (%s)", aid, asset_type)
                        hist = generate_supporting_history(
                            asset_type,
                            wells_only,
                            lease,
                            sd,
                            ed,
                            twin=engine,
                            tank_hist=tank_hist,
                        )
                        if asset_type == "Tank Battery":
                            tank_hist = hist
                        chunk = _stamp_org(
                            _filter_history(hist, sup_start, window_end), organization_id
                        )
                        if not chunk:
                            continue
                        result["master_history_rows"] += append_dataframe(
                            conn,
                            "SYNTH_ASSET_DAILY_HISTORY_V2",
                            SynthAssetDailyHistoryV2.history_to_dataframe(
                                chunk,
                                asset_id=aid,
                                asset_type=asset_type,
                                lift_type=None,
                                organization_id=organization_id,
                            ),
                        )
                        inv = generate_supporting_invoices(
                            aid, asset_type, chunk, working_interest=wi
                        )
                        inv = _stamp_org(inv, organization_id)
                        result["invoice_rows"] += append_dataframe(
                            conn,
                            "SYNTH_INVOICES_V2",
                            SynthInvoicesV2.invoices_to_dataframe(inv, organization_id),
                        )
                result["supporting_loaded"] = True

        if not result["daily_rows"] and not result["master_history_rows"]:
            result["message"] = "No rows generated for this window."

        logger.info(
            "RED_BUFF V2 complete: wells=%s/%s daily=%s sensor=%s history=%s invoices=%s revenue=%s",
            result["wells_processed"],
            result["wells_total"],
            result["daily_rows"],
            result["sensor_rows"],
            result["master_history_rows"],
            result["invoice_rows"],
            result["revenue_rows"],
        )
        return result
    finally:
        conn.close()
