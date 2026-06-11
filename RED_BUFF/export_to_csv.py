"""
Red Bluff Resources — CSV Exporter
======================================
Flattens Red_Bluff_Resources_Master.json into 4 relational CSV files
joined by asset_id — ready for SQL, Pandas, or any tabular ML pipeline.

Pipeline order:
  1. python3 generate_digital_twin.py   → red_bluff_engineering.json
  2. python3 generate_invoices.py       → adds financial_ledger
  3. python3 build_master.py            → Red_Bluff_Resources_Master.json
  4. python3 export_to_csv.py           → THIS SCRIPT → 4 CSV files

Output files:
  assets.csv            — one row per asset (asset_id, type, lift, location)
  personnel.csv         — one row per role per asset
  production_history.csv— one row per day per well (all sensor fields)
  invoices.csv          — one row per invoice (all financial fields)
"""

import json
import pandas as pd

if __name__ == "__main__":
    print("Loading Red_Bluff_Resources_Master.json...")
    with open("Red_Bluff_Resources_Master.json") as f:
        data = json.load(f)

    assets_rows     = []
    personnel_rows  = []
    history_rows    = []
    invoice_rows    = []

    for asset_id, asset in data["assets"].items():
        # ── Assets ───────────────────────────────────────────────────────────
        loc = asset.get("location", {})
        assets_rows.append({
            "asset_id":   asset_id,
            "type":       asset.get("type"),
            "lift":       asset.get("lift"),
            "basin":      loc.get("basin"),
            "sub_basin":  loc.get("sub_basin"),
            "state":      loc.get("state"),
            "county":     loc.get("county"),
            "region":     loc.get("region"),
        })

        # ── Personnel ─────────────────────────────────────────────────────────
        for role_key, person in asset.get("assigned_personnel", {}).items():
            personnel_rows.append({
                "asset_id":  asset_id,
                "role_key":  role_key,
                "name":      person.get("name"),
                "role":      person.get("role"),
                "email":     person.get("email"),
            })

        # ── Production history ────────────────────────────────────────────────
        for day in asset.get("history", []):
            row = {"asset_id": asset_id}
            row.update(day)
            history_rows.append(row)

        # ── Invoices ──────────────────────────────────────────────────────────
        for inv in asset.get("financial_ledger", []):
            row = {"asset_id": asset_id}
            row.update(inv)
            invoice_rows.append(row)

    # Write CSVs
    pd.DataFrame(assets_rows).to_csv("assets.csv",              index=False)
    pd.DataFrame(personnel_rows).to_csv("personnel.csv",        index=False)
    pd.DataFrame(history_rows).to_csv("production_history.csv", index=False)
    pd.DataFrame(invoice_rows).to_csv("invoices.csv",           index=False)

    print(f"✓ Done — 4 CSV files written")
    print(f"  assets.csv:              {len(assets_rows):,} rows")
    print(f"  personnel.csv:           {len(personnel_rows):,} rows")
    print(f"  production_history.csv:  {len(history_rows):,} rows")
    print(f"  invoices.csv:            {len(invoice_rows):,} rows")
