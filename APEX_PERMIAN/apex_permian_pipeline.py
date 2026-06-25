"""
Apex Permian synthetic pipeline → Snowflake (same data shape as APEX_PERMIAN/generate_digital_twin.py).

Empty org: full history (spud → today) for all wells.
Has data: append from last daily date + 1 through today.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from APEX_PERMIAN.column_maps import master_df_to_daily_production, master_df_to_sensor_readings
from APEX_PERMIAN.generate_digital_twin import build_hierarchy, generate_well_timeline
from APEX_PERMIAN.snowflake_io import (
    append_dataframe,
    clear_organization_data,
    connect,
    ensure_schema,
    get_watermarks,
    load_registry,
    save_registry,
)

logger = logging.getLogger(__name__)

_EARLIEST_SIM_DATE = datetime(2014, 1, 1)


def _today() -> datetime:
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _compute_window_to_today(last_daily_date, config_end: datetime) -> tuple[datetime, datetime, bool]:
    window_end = max(config_end, _today())
    if last_daily_date is None:
        return _EARLIEST_SIM_DATE, window_end, True

    last_dt = datetime.combine(last_daily_date, datetime.min.time())
    window_start = last_dt + timedelta(days=1)
    return window_start, window_end, False


def generation_preview(last_daily_date) -> dict:
    today = _today().date()
    last_d = _as_date(last_daily_date)

    if last_d is None:
        return {
            "mode": "full_backfill",
            "last_daily": None,
            "today": today.isoformat(),
            "pending_days": None,
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
            "message": f"Up to date through {last_d.isoformat()}.",
        }

    return {
        "mode": "incremental",
        "last_daily": last_d.isoformat(),
        "today": today.isoformat(),
        "pending_days": pending,
        "message": f"Incremental: {pending} day(s) ({next_d.isoformat()} → {today.isoformat()}).",
    }


def reset_organization_data(organization_id: str, ensure_tables: bool = True) -> dict:
    conn = connect()
    try:
        if ensure_tables:
            ensure_schema(conn)
        return clear_organization_data(conn, organization_id)
    finally:
        conn.close()


def run_apex_permian_pipeline(organization_id: str, ensure_tables: bool = True) -> dict:
    import APEX_PERMIAN.generate_digital_twin as twin

    config_end = twin.END_DATE
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
                "is_full_backfill": False,
                "pending_days": 0,
                "last_daily": preview.get("last_daily"),
                "daily_rows": 0,
                "sensor_rows": 0,
                "message": preview["message"],
            }

        sim_end = max(config_end, window_end)
        twin.END_DATE = sim_end

        registry = load_registry(conn, organization_id)
        if registry is None or not registry.get("wells"):
            well_meta, fac_meta = build_hierarchy()
            for w in well_meta:
                w["spud_date"] = _parse_date(w["spud_date"])
            registry = {
                "wells": [
                    {
                        **w,
                        "spud_date": w["spud_date"].strftime("%Y-%m-%d")
                        if isinstance(w["spud_date"], datetime)
                        else w["spud_date"],
                    }
                    for w in well_meta
                ],
                "facilities": fac_meta,
                "master_initialized": True,
            }
            save_registry(conn, organization_id, registry)

        well_meta = []
        for w in registry["wells"]:
            row = dict(w)
            row["spud_date"] = _parse_date(row["spud_date"])
            well_meta.append(row)

        daily_frames: List[pd.DataFrame] = []
        sensor_frames: List[pd.DataFrame] = []

        for i, well in enumerate(well_meta):
            spud = _parse_date(well["spud_date"])
            chunk_start = spud if is_full else max(spud, window_start)
            if chunk_start > window_end:
                continue

            logger.info("[%d/%d] Well %s", i + 1, len(well_meta), well["asset_id"])
            timeline = generate_well_timeline(well)
            if timeline.empty:
                continue

            timeline["date"] = pd.to_datetime(timeline["date"])
            mask = (timeline["date"] >= pd.Timestamp(chunk_start)) & (
                timeline["date"] <= pd.Timestamp(window_end)
            )
            chunk = timeline.loc[mask]
            if chunk.empty:
                continue

            daily_frames.append(master_df_to_daily_production(chunk, well_meta, organization_id))
            sensor_frames.append(master_df_to_sensor_readings(chunk, well_meta, organization_id))

        result: Dict[str, object] = {
            "window_start": window_start,
            "window_end": window_end,
            "is_full_backfill": is_full,
            "pending_days": preview.get("pending_days"),
            "last_daily": preview.get("last_daily"),
            "daily_rows": 0,
            "sensor_rows": 0,
            "message": None,
        }

        if daily_frames:
            result["daily_rows"] = append_dataframe(
                conn, "WELL_DAILY_PRODUCTION", pd.concat(daily_frames, ignore_index=True)
            )
            result["sensor_rows"] = append_dataframe(
                conn, "WELL_SENSOR_READINGS", pd.concat(sensor_frames, ignore_index=True)
            )
        else:
            result["message"] = "No rows generated for this window."

        logger.info(
            "Apex Permian pipeline complete: daily=%s sensor=%s",
            result["daily_rows"],
            result["sensor_rows"],
        )
        return result
    finally:
        conn.close()
