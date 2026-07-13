"""Map South Texas Energy Partners digital-twin history rows to Snowflake canonical tables."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from master_assets import get_lease_area
from transformations.columns import WELL_DAILY_PRODUCTION_COLUMNS, WELL_SENSOR_READINGS_COLUMNS

SYNTH_ASSETS_COLUMNS = [
    "organization_id",
    "asset_id",
    "asset_type",
    "lift_type",
    "lease",
    "area",
    "basin",
    "sub_basin",
    "state",
    "county",
    "region",
]

SYNTH_PERSONNEL_COLUMNS = [
    "organization_id",
    "asset_id",
    "role_key",
    "name",
    "role",
    "email",
]

SYNTH_ASSET_DAILY_HISTORY_COLUMNS = [
    "organization_id",
    "asset_id",
    "date",
    "asset_type",
    "lift_type",
    "run_status",
    "history_json",
]

SYNTH_INVOICES_COLUMNS = [
    "organization_id",
    "invoice_id",
    "invoice_date",
    "asset_id",
    "category",
    "gl_code",
    "service_description",
    "vendor",
    "total_usd",
    "invoice_json",
]

_DAILY_COLS = [c for c, _ in WELL_DAILY_PRODUCTION_COLUMNS]
_SENSOR_COLS = [c for c, _ in WELL_SENSOR_READINGS_COLUMNS]


def _clean(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None


def _lease_from_asset_id(asset_id: str) -> str:
    return asset_id.split("-")[0] if "-" in asset_id else asset_id


def _facility_id(lease: str) -> str:
    return f"{lease}-FAC-01"


def _map_daily_sensors(day: dict, lift: str) -> Dict[str, Optional[float]]:
    """Extract canonical sensor fields from one STEP daily history row."""
    out: Dict[str, Optional[float]] = {}

    if lift == "Pumping Unit":
        out["surface_stroke_length"] = _clean(day.get("polished_rod_stroke_length_in"))
        out["downhole_gross_stroke"] = _clean(day.get("polished_rod_stroke_length_in"))
        out["downhole_net_stroke"] = _clean(day.get("effective_plunger_stroke_in"))
        out["strokes_per_minute"] = _clean(day.get("pumping_speed_spm"))
        out["pump_fillage"] = _clean(day.get("volumetric_efficiency_pct"))
        out["casing_pressure"] = _clean(day.get("annular_casing_pressure_psi"))
        out["tubing_pressure"] = _clean(day.get("wellhead_tubing_pressure_psi"))
        out["motor_current"] = _clean(day.get("surface_motor_amperage"))
        out["drive_frequency"] = _clean(day.get("motor_drive_frequency_hz"))
        out["pump_intake_pressure"] = _clean(day.get("downhole_pump_intake_psi"))
        out["motor_temp"] = _clean(day.get("gearbox_motor_temp_f"))
    elif lift == "ESP":
        out["tubing_pressure"] = _clean(day.get("flowing_tubing_pressure_psi"))
        out["casing_pressure"] = _clean(day.get("annular_casing_pressure_psi"))
        out["motor_temp"] = _clean(day.get("motor_winding_temp_f"))
        out["motor_current"] = _clean(day.get("motor_load_amps"))
        out["drive_frequency"] = _clean(day.get("vfd_output_frequency_hz"))
        out["pump_intake_pressure"] = _clean(day.get("pump_intake_pressure_psi"))
        out["discharge_pressure"] = _clean(day.get("pump_discharge_pressure_psi"))
        out["vibration_x"] = _clean(day.get("lateral_vibration_hz"))
    elif lift == "Gas Lift":
        out["casing_pressure"] = _clean(day.get("annular_casing_pressure_psi"))
        out["injection_pressure"] = _clean(day.get("lift_gas_injection_pressure_psi"))
        out["tubing_pressure"] = _clean(day.get("surface_tubing_pressure_psi"))
        out["injection_rate"] = _clean(day.get("lift_gas_rate_mscfd"))
        out["wellhead_temp"] = _clean(day.get("wellhead_flowing_temp_f"))
        out["motor_temp"] = _clean(day.get("compressor_inlet_temp_f"))

    return out


def history_to_daily_production(
    history: List[dict],
    well: dict,
    organization_id: str,
) -> pd.DataFrame:
    """Convert per-well daily history dicts to WELL_DAILY_PRODUCTION rows."""
    asset_id = well["asset_id"]
    lease = well.get("lease") or _lease_from_asset_id(asset_id)
    lift = well["lift"]
    facility = _facility_id(lease)

    rows = []
    for day in history:
        oil = _clean(day.get("gross_oil_production_bbl")) or 0.0
        gas = _clean(day.get("casinghead_gas_mcf")) or 0.0
        water = _clean(day.get("produced_water_volume_bbl")) or 0.0
        boe = round(oil + gas / 6.0, 2) if (oil or gas) else 0.0
        status = day.get("run_status") or "PRODUCING"

        rows.append(
            {
                "organization_id": organization_id,
                "well_id": asset_id,
                "date": pd.to_datetime(day["date"]).date(),
                "oil_volume": oil,
                "gas_volume": gas,
                "water_volume": water,
                "boe": boe,
                "lift_type": lift,
                "status": status,
                "notes": None,
                "test_oil": _clean(day.get("well_test_rate_bopd")),
                "test_water": None,
                "test_gas": None,
                "facility_id": facility,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_DAILY_COLS)
    return pd.DataFrame(rows)[_DAILY_COLS]


def history_to_sensor_readings(
    history: List[dict],
    well: dict,
    organization_id: str,
) -> pd.DataFrame:
    """Convert daily STEP history to WELL_SENSOR_READINGS rows (one per day at midnight)."""
    asset_id = well["asset_id"]
    lift = well["lift"]
    rows: List[dict] = []

    for day in history:
        ts = pd.Timestamp(day["date"])
        sensors = _map_daily_sensors(day, lift)
        well_status = day.get("run_status") or "PRODUCING"

        rec = {c: None for c in _SENSOR_COLS}
        rec["organization_id"] = organization_id
        rec["well_id"] = asset_id
        rec["timestamp"] = ts
        rec["lift_type"] = lift
        rec["well_status"] = well_status
        for col, val in sensors.items():
            if col in rec:
                rec[col] = val
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=_SENSOR_COLS)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[_SENSOR_COLS]


def _resolve_area(asset: dict, lease: str) -> str:
    area = (asset.get("area") or "").strip()
    return area or get_lease_area(lease)


def asset_record_to_catalog_row(asset_id: str, asset: dict, organization_id: str) -> dict:
    loc = asset.get("location") or {}
    lease = _lease_from_asset_id(asset_id)
    return {
        "organization_id": organization_id,
        "asset_id": asset_id,
        "asset_type": asset.get("type"),
        "lift_type": asset.get("lift"),
        "lease": lease,
        "area": _resolve_area(asset, lease),
        "basin": loc.get("basin"),
        "sub_basin": loc.get("sub_basin"),
        "state": loc.get("state"),
        "county": loc.get("county"),
        "region": loc.get("region"),
    }


def assets_catalog_to_dataframe(catalog: List[dict], organization_id: str) -> pd.DataFrame:
    rows = []
    for item in catalog:
        if "asset_type" in item or "type" in item:
            loc = item.get("location") or {}
            lease = item.get("lease") or _lease_from_asset_id(item["asset_id"])
            rows.append(
                {
                    "organization_id": organization_id,
                    "asset_id": item["asset_id"],
                    "asset_type": item.get("asset_type") or item.get("type"),
                    "lift_type": item.get("lift_type") or item.get("lift"),
                    "lease": lease,
                    "area": (item.get("area") or "").strip() or get_lease_area(lease),
                    "basin": loc.get("basin"),
                    "sub_basin": loc.get("sub_basin"),
                    "state": loc.get("state"),
                    "county": loc.get("county"),
                    "region": loc.get("region"),
                }
            )
        else:
            rows.append(item)
    if not rows:
        return pd.DataFrame(columns=SYNTH_ASSETS_COLUMNS)
    return pd.DataFrame(rows)[SYNTH_ASSETS_COLUMNS]


def personnel_to_dataframe(
    asset_id: str,
    personnel: dict,
    organization_id: str,
) -> pd.DataFrame:
    rows = []
    for role_key, person in (personnel or {}).items():
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "role_key": role_key,
                "name": person.get("name"),
                "role": person.get("role"),
                "email": person.get("email"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=SYNTH_PERSONNEL_COLUMNS)
    return pd.DataFrame(rows)[SYNTH_PERSONNEL_COLUMNS]


def master_history_to_dataframe(
    asset_id: str,
    asset_type: str,
    lift_type: Optional[str],
    history: List[dict],
    organization_id: str,
) -> pd.DataFrame:
    rows = []
    for day in history:
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "date": pd.to_datetime(day["date"]).date(),
                "asset_type": asset_type,
                "lift_type": lift_type,
                "run_status": day.get("run_status"),
                "history_json": json.dumps(day, default=str),
            }
        )
    if not rows:
        return pd.DataFrame(columns=SYNTH_ASSET_DAILY_HISTORY_COLUMNS)
    return pd.DataFrame(rows)[SYNTH_ASSET_DAILY_HISTORY_COLUMNS]


def invoices_to_dataframe(invoices: List[dict], organization_id: str) -> pd.DataFrame:
    rows = []
    for inv in invoices:
        rows.append(
            {
                "organization_id": organization_id,
                "invoice_id": inv.get("invoice_id"),
                "invoice_date": pd.to_datetime(inv.get("invoice_date")).date()
                if inv.get("invoice_date")
                else None,
                "asset_id": inv.get("asset_id"),
                "category": inv.get("category"),
                "gl_code": inv.get("gl_code"),
                "service_description": inv.get("service_description"),
                "vendor": inv.get("vendor"),
                "total_usd": inv.get("total_usd"),
                "invoice_json": json.dumps(inv, default=str),
            }
        )
    if not rows:
        return pd.DataFrame(columns=SYNTH_INVOICES_COLUMNS)
    return pd.DataFrame(rows)[SYNTH_INVOICES_COLUMNS]
