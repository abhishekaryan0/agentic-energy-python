"""Map Snowflake SYNTH_* rows to scripts-compatible master + financial parquet."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from transformations.synth_schemas import (
    INVOICES_V2_COLUMNS,
    MASTER_ASSETS_COLUMNS,
    MASTER_HISTORY_COLUMNS,
)


def _float_or_none(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _date_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def _parse_history_blob(blob: Any) -> dict:
    if blob is None or (isinstance(blob, float) and pd.isna(blob)):
        return {}
    if isinstance(blob, dict):
        return blob
    try:
        return json.loads(str(blob))
    except Exception:
        return {}


def build_master_assets_df(
    assets_df: pd.DataFrame,
    personnel_df: pd.DataFrame,
    invoices_df: Optional[pd.DataFrame],
    organization_id: str,
) -> pd.DataFrame:
    """One row per asset; personnel and invoices rolled into JSON columns."""
    if assets_df.empty:
        return pd.DataFrame(columns=MASTER_ASSETS_COLUMNS)

    personnel_by_asset: Dict[str, dict] = {}
    if not personnel_df.empty:
        for asset_id, grp in personnel_df.groupby("asset_id", sort=False):
            people = {}
            for _, r in grp.iterrows():
                key = str(r.get("role_key") or r.get("role") or "contact")
                people[str(key)] = {
                    "name": r.get("name"),
                    "role": r.get("role"),
                    "email": r.get("email"),
                }
            personnel_by_asset[str(asset_id)] = people

    ledger_by_asset: Dict[str, list] = {}
    if invoices_df is not None and not invoices_df.empty:
        for asset_id, grp in invoices_df.groupby("asset_id", sort=False):
            ledger_by_asset[str(asset_id)] = [
                _parse_history_blob(r.get("invoice_json"))
                for _, r in grp.iterrows()
                if r.get("invoice_json")
            ]

    rows = []
    for _, a in assets_df.iterrows():
        asset_id = str(a["asset_id"])
        raw_asset = {
            "asset_id": asset_id,
            "type": a.get("asset_type"),
            "lift": a.get("lift_type"),
            "lease": a.get("lease"),
            "location": {
                "basin": a.get("basin"),
                "sub_basin": a.get("sub_basin"),
                "state": a.get("state"),
                "county": a.get("county"),
                "region": a.get("region"),
            },
        }
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "asset_type": a.get("asset_type"),
                "lift_type": a.get("lift_type"),
                "assigned_personnel_json": json.dumps(
                    personnel_by_asset.get(asset_id, {}), default=str
                ),
                "financial_ledger_json": json.dumps(
                    ledger_by_asset.get(asset_id, []), default=str
                )
                if ledger_by_asset.get(asset_id)
                else None,
                "raw_asset_json": json.dumps(raw_asset, default=str),
            }
        )
    return pd.DataFrame(rows, columns=MASTER_ASSETS_COLUMNS)


def _history_row_from_snowflake(
    organization_id: str,
    asset_id: str,
    asset_type: str,
    lift_type: Optional[str],
    day_date,
    run_status: Optional[str],
    history_json: Any,
) -> dict:
    h = _parse_history_blob(history_json)
    asset_type = asset_type or h.get("asset_type")
    status = run_status or h.get("run_status") or h.get("status")
    notes = h.get("notes") or h.get("failure_cause")

    row = {
        "organization_id": organization_id,
        "asset_id": asset_id,
        "asset_type": asset_type,
        "lift_type": lift_type if asset_type == "Well" else lift_type,
        "date": _date_or_none(day_date or h.get("date")),
        "status": status,
        "notes": notes,
        "oil_prod": None,
        "allocated_oil": None,
        "test_oil": None,
        "avg_tubing_pressure": None,
        "avg_casing_pressure": None,
        "avg_motor_amps": None,
        "gross_stroke_len": None,
        "net_stroke_len": None,
        "spm": None,
        "pump_fillage_pct": None,
        "freq_hz": None,
        "pump_intake_pressure": None,
        "motor_temp_f": None,
        "injection_rate_mcf": None,
        "raw_history_json": json.dumps(h if h else {"date": str(day_date)}, default=str),
    }

    if asset_type == "Facility":
        row["oil_prod"] = _float_or_none(h.get("oil_prod"))
    elif asset_type == "Well":
        row.update(
            {
                "allocated_oil": _float_or_none(
                    h.get("allocated_oil") or h.get("oil_volume")
                ),
                "test_oil": _float_or_none(h.get("test_oil")),
                "avg_tubing_pressure": _float_or_none(
                    h.get("avg_tubing_pressure") or h.get("tubing_pressure")
                ),
                "avg_casing_pressure": _float_or_none(
                    h.get("avg_casing_pressure") or h.get("casing_pressure")
                ),
                "avg_motor_amps": _float_or_none(
                    h.get("avg_motor_amps") or h.get("motor_current")
                ),
                "gross_stroke_len": _float_or_none(h.get("gross_stroke_len")),
                "net_stroke_len": _float_or_none(h.get("net_stroke_len")),
                "spm": _float_or_none(h.get("spm") or h.get("strokes_per_minute")),
                "pump_fillage_pct": _float_or_none(
                    h.get("pump_fillage_pct") or h.get("pump_fillage")
                ),
                "freq_hz": _float_or_none(h.get("freq_hz") or h.get("drive_frequency")),
                "pump_intake_pressure": _float_or_none(h.get("pump_intake_pressure")),
                "motor_temp_f": _float_or_none(h.get("motor_temp_f") or h.get("motor_temp")),
                "injection_rate_mcf": _float_or_none(h.get("injection_rate_mcf")),
            }
        )
    return row


def build_master_history_df(history_df: pd.DataFrame, organization_id: str) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=MASTER_HISTORY_COLUMNS)
    rows = []
    for _, r in history_df.iterrows():
        rows.append(
            _history_row_from_snowflake(
                organization_id,
                str(r["asset_id"]),
                str(r.get("asset_type") or ""),
                r.get("lift_type"),
                r.get("date"),
                r.get("run_status"),
                r.get("history_json"),
            )
        )
    return pd.DataFrame(rows, columns=MASTER_HISTORY_COLUMNS)


def build_invoices_v2_df(invoices_df: pd.DataFrame, organization_id: str) -> pd.DataFrame:
    if invoices_df.empty:
        return pd.DataFrame(columns=INVOICES_V2_COLUMNS)
    rows = []
    for _, r in invoices_df.iterrows():
        raw = _parse_history_blob(r.get("invoice_json"))
        rows.append(
            {
                "organization_id": organization_id,
                "invoice_version": "v2",
                "invoice_id": r.get("invoice_id"),
                "invoice_date": _date_or_none(r.get("invoice_date")),
                "status": raw.get("status"),
                "category": r.get("category"),
                "service_description": r.get("service_description"),
                "service_period_start": None,
                "service_period_end": None,
                "asset_id": r.get("asset_id"),
                "vendor_name": r.get("vendor"),
                "vendor_type": None,
                "total_usd": _float_or_none(r.get("total_usd")),
                "subtotal_usd": None,
                "gl_code": r.get("gl_code"),
                "cost_type": None,
                "capex_opex_flag": None,
                "line_items_count": None,
                "line_total_usd_sum": None,
                "raw_invoice_json": json.dumps(raw if raw else dict(r), default=str),
            }
        )
    return pd.DataFrame(rows, columns=INVOICES_V2_COLUMNS)
