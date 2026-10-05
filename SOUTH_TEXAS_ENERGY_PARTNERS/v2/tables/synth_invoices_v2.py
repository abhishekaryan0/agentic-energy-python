"""SYNTH_INVOICES_V2 — financial_ledger rows with V2 WI / gross cost fields."""

from __future__ import annotations

import json
from typing import List

import pandas as pd

TABLE_NAME = "SYNTH_INVOICES_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("invoice_id", "VARCHAR(50)"),
    ("invoice_date", "DATE"),
    ("asset_id", "VARCHAR(100)"),
    ("category", "VARCHAR(50)"),
    ("service_description", "VARCHAR(500)"),
    ("vendor", "VARCHAR(200)"),
    ("gross_cost_usd", "FLOAT"),
    ("working_interest", "FLOAT"),
    ("total_usd", "FLOAT"),
    ("invoice_json", "VARCHAR"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


def ddl(qualified: str) -> str:
    cols = ",\n            ".join(f"{name.upper()} {ctype}" for name, ctype in COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            {cols},
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, INVOICE_ID)
        )
        """


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
                "service_description": inv.get("service_description"),
                "vendor": inv.get("vendor"),
                "gross_cost_usd": inv.get("gross_cost_usd"),
                "working_interest": inv.get("working_interest"),
                "total_usd": inv.get("total_usd"),
                "invoice_json": json.dumps(inv, default=str),
            }
        )
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


def assets_invoices_to_dataframe(assets: dict, organization_id: str) -> pd.DataFrame:
    batches = []
    for asset_id, asset in assets.items():
        for inv in asset.get("financial_ledger") or []:
            batches.append({**inv, "asset_id": asset_id})
    return invoices_to_dataframe(batches, organization_id)


class SynthInvoicesV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    invoices_to_dataframe = staticmethod(invoices_to_dataframe)
    assets_invoices_to_dataframe = staticmethod(assets_invoices_to_dataframe)
