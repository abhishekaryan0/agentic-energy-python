"""SYNTH_ASSETS_V2 — asset catalog with V2 ownership / lease / pump metadata."""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pandas as pd

TABLE_NAME = "SYNTH_ASSETS_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("asset_type", "VARCHAR(50)"),
    ("lift_type", "VARCHAR(50)"),
    ("lease", "VARCHAR(50)"),
    ("area", "VARCHAR(100)"),
    ("lease_type", "VARCHAR(100)"),
    ("working_interest", "FLOAT"),
    ("net_revenue_interest", "FLOAT"),
    ("wellbore_profile", "VARCHAR(50)"),
    ("lateral_length_ft", "FLOAT"),
    ("basin", "VARCHAR(100)"),
    ("sub_basin", "VARCHAR(100)"),
    ("state", "VARCHAR(10)"),
    ("county", "VARCHAR(100)"),
    ("region", "VARCHAR(100)"),
    ("pump_type", "VARCHAR(50)"),
    ("pump_curve_json", "VARCHAR"),
    ("pump_static_specs_json", "VARCHAR"),
    ("tank_static_specs_json", "VARCHAR"),
    ("asset_json", "VARCHAR"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


def ddl(qualified: str) -> str:
    cols = ",\n            ".join(f"{name.upper()} {ctype}" for name, ctype in COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            {cols},
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID)
        )
        """


def _lease_from_asset_id(asset_id: str) -> str:
    return asset_id.split("-")[0] if "-" in asset_id else asset_id


def _json_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, default=str)


def asset_to_row(asset_id: str, asset: dict, organization_id: str) -> dict:
    loc = asset.get("location") or {}
    lease = _lease_from_asset_id(asset_id)
    # Drop heavy nested collections from the side-car JSON; they live in other tables.
    slim = {
        k: v
        for k, v in asset.items()
        if k
        not in (
            "history",
            "financial_ledger",
            "revenue_ledger",
            "assigned_personnel",
            "pump_curve",
            "pump_static_specs",
            "tank_static_specs",
        )
    }
    return {
        "organization_id": organization_id,
        "asset_id": asset_id,
        "asset_type": asset.get("type"),
        "lift_type": asset.get("lift"),
        "lease": lease,
        "area": asset.get("area"),
        "lease_type": asset.get("lease_type"),
        "working_interest": asset.get("working_interest"),
        "net_revenue_interest": asset.get("net_revenue_interest"),
        "wellbore_profile": asset.get("wellbore_profile"),
        "lateral_length_ft": asset.get("lateral_length_ft"),
        "basin": loc.get("basin"),
        "sub_basin": loc.get("sub_basin"),
        "state": loc.get("state"),
        "county": loc.get("county"),
        "region": loc.get("region"),
        "pump_type": asset.get("pump_type"),
        "pump_curve_json": _json_or_none(asset.get("pump_curve")),
        "pump_static_specs_json": _json_or_none(asset.get("pump_static_specs")),
        "tank_static_specs_json": _json_or_none(asset.get("tank_static_specs")),
        "asset_json": json.dumps(slim, default=str),
    }


def assets_to_dataframe(assets: dict, organization_id: str) -> pd.DataFrame:
    rows = [asset_to_row(aid, asset, organization_id) for aid, asset in assets.items()]
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


class SynthAssetsV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    asset_to_row = staticmethod(asset_to_row)
    assets_to_dataframe = staticmethod(assets_to_dataframe)
