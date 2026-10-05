"""SYNTH_REVENUE_V2 — monthly revenue_ledger rows (V2-only)."""

from __future__ import annotations

import json
from typing import List

import pandas as pd

TABLE_NAME = "SYNTH_REVENUE_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("revenue_id", "VARCHAR(50)"),
    ("production_month", "VARCHAR(7)"),
    ("oil_volume_bbl", "FLOAT"),
    ("gas_volume_mcf", "FLOAT"),
    ("residue_gas_mcf", "FLOAT"),
    ("ngl_volume_bbl", "FLOAT"),
    ("water_volume_bbl", "FLOAT"),
    ("realized_oil_usd_bbl", "FLOAT"),
    ("realized_gas_usd_mcf", "FLOAT"),
    ("realized_ngl_usd_bbl", "FLOAT"),
    ("gross_oil_revenue_usd", "FLOAT"),
    ("gross_gas_revenue_usd", "FLOAT"),
    ("gross_ngl_revenue_usd", "FLOAT"),
    ("gross_revenue_usd", "FLOAT"),
    ("net_revenue_interest", "FLOAT"),
    ("net_revenue_before_deducts_usd", "FLOAT"),
    ("severance_tax_usd", "FLOAT"),
    ("ad_valorem_tax_usd", "FLOAT"),
    ("gathering_processing_usd", "FLOAT"),
    ("net_revenue_usd", "FLOAT"),
    ("revenue_json", "VARCHAR"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


def ddl(qualified: str) -> str:
    cols = ",\n            ".join(f"{name.upper()} {ctype}" for name, ctype in COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            {cols},
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, REVENUE_ID)
        )
        """


def revenue_to_dataframe(rows_in: List[dict], organization_id: str) -> pd.DataFrame:
    rows = []
    for r in rows_in:
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": r.get("asset_id"),
                "revenue_id": r.get("revenue_id"),
                "production_month": r.get("production_month"),
                "oil_volume_bbl": r.get("oil_volume_bbl"),
                "gas_volume_mcf": r.get("gas_volume_mcf"),
                "residue_gas_mcf": r.get("residue_gas_mcf"),
                "ngl_volume_bbl": r.get("ngl_volume_bbl"),
                "water_volume_bbl": r.get("water_volume_bbl"),
                "realized_oil_usd_bbl": r.get("realized_oil_usd_bbl"),
                "realized_gas_usd_mcf": r.get("realized_gas_usd_mcf"),
                "realized_ngl_usd_bbl": r.get("realized_ngl_usd_bbl"),
                "gross_oil_revenue_usd": r.get("gross_oil_revenue_usd"),
                "gross_gas_revenue_usd": r.get("gross_gas_revenue_usd"),
                "gross_ngl_revenue_usd": r.get("gross_ngl_revenue_usd"),
                "gross_revenue_usd": r.get("gross_revenue_usd"),
                "net_revenue_interest": r.get("net_revenue_interest"),
                "net_revenue_before_deducts_usd": r.get("net_revenue_before_deducts_usd"),
                "severance_tax_usd": r.get("severance_tax_usd"),
                "ad_valorem_tax_usd": r.get("ad_valorem_tax_usd"),
                "gathering_processing_usd": r.get("gathering_processing_usd"),
                "net_revenue_usd": r.get("net_revenue_usd"),
                "revenue_json": json.dumps(r, default=str),
            }
        )
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


def assets_revenue_to_dataframe(assets: dict, organization_id: str) -> pd.DataFrame:
    batches = []
    for asset_id, asset in assets.items():
        for r in asset.get("revenue_ledger") or []:
            batches.append({**r, "asset_id": asset_id})
    return revenue_to_dataframe(batches, organization_id)


class SynthRevenueV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    revenue_to_dataframe = staticmethod(revenue_to_dataframe)
    assets_revenue_to_dataframe = staticmethod(assets_revenue_to_dataframe)
