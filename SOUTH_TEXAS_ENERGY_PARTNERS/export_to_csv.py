"""
South Texas Energy Partners — CSV Exporter
==========================================
Flattens South_Texas_Energy_Partners_Master.json into Snowflake-ready
CSV files aligned to Ayush's pipeline structure.

Pipeline order:
  1. python3 generate_digital_twin.py   → south_texas_engineering.json
  2. python3 generate_invoices.py       → adds financial_ledger
  3. python3 build_master.py            → South_Texas_Energy_Partners_Master.json
  4. python3 export_to_csv.py           → THIS SCRIPT → CSV files

Output files (aligned to Snowflake tables):
  WELL_SENSOR_READINGS table:
    south_texas_well_sensor_readings.csv  — daily sensor instrument readings per well
  WELL_DAILY_PRODUCTION table:
    south_texas_well_daily_production.csv — daily production volumes per well
  Supporting tables:
    south_texas_assets.csv                — one row per asset (metadata)
    south_texas_personnel.csv             — one row per role per asset
    south_texas_supporting_history.csv    — daily readings for Facility/Compressor/SWD/Tank
    south_texas_invoices.csv              — one row per invoice

Key fields per Ayush's pipeline:
  organization_id  — tenant isolation (SOUTH_TEXAS_ENERGY_PARTNERS on all rows)
  well_id          — maps to asset_id for Snowflake joins
  date             — YYYY-MM-DD format for incremental watermark queries
  timestamp        — YYYY-MM-DD 00:00:00 for Snowflake TIMESTAMP compatibility
"""

import json
import pandas as pd

try:
    from config import LEGACY_ORGANIZATION_LABEL, STEP_ORGANIZATION_ID
except ImportError:
    STEP_ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"
    LEGACY_ORGANIZATION_LABEL = "SOUTH_TEXAS_ENERGY_PARTNERS"

# ── Sensor fields per lift type ────────────────────────────────────────────────
# WELL_SENSOR_READINGS — instrument readings only
ESP_SENSORS = [
    'flowing_tubing_pressure_psi', 'annular_casing_pressure_psi',
    'motor_winding_temp_f', 'motor_load_amps', 'vfd_output_frequency_hz',
    'pump_intake_pressure_psi', 'pump_discharge_pressure_psi',
    'lateral_vibration_hz',
]
GL_SENSORS = [
    'annular_casing_pressure_psi', 'lift_gas_injection_pressure_psi',
    'surface_tubing_pressure_psi', 'lift_gas_rate_mscfd',
    'wellhead_flowing_temp_f', 'production_choke_opening_pct',
    'compressor_inlet_temp_f',
]
PU_SENSORS = [
    'polished_rod_stroke_length_in', 'effective_plunger_stroke_in',
    'pumping_speed_spm', 'volumetric_efficiency_pct',
    'annular_casing_pressure_psi', 'wellhead_tubing_pressure_psi',
    'surface_motor_amperage', 'motor_drive_frequency_hz',
    'downhole_pump_intake_psi', 'gearbox_motor_temp_f',
]
SENSOR_FIELDS = {'ESP': ESP_SENSORS, 'Gas Lift': GL_SENSORS, 'Pumping Unit': PU_SENSORS}

# WELL_DAILY_PRODUCTION — volumes and status only
PRODUCTION_FIELDS = [
    'gross_oil_production_bbl', 'well_test_rate_bopd',
    'produced_water_volume_bbl', 'casinghead_gas_mcf',
    'run_status',
]

if __name__ == "__main__":
    print("Loading South_Texas_Energy_Partners_Master.json...")
    with open("South_Texas_Energy_Partners_Master.json") as f:
        data = json.load(f)

    assets = data["assets"]

    assets_rows      = []
    personnel_rows   = []
    sensor_rows      = []
    production_rows  = []
    supporting_rows  = []
    invoice_rows     = []

    for asset_id, asset in assets.items():
        org_id = STEP_ORGANIZATION_ID
        loc    = asset.get("location", {})
        lift   = asset.get("lift")

        # ── Assets table ──────────────────────────────────────────────────────
        assets_rows.append({
            "organization_id":      org_id,
            "asset_id":             asset_id,
            "well_id":              asset_id,   # alias for Snowflake joins
            "type":                 asset.get("type"),
            "lift":                 lift,
            "area":                 asset.get("area", ""),
            "working_interest":     asset.get("working_interest"),
            "lease_type":           asset.get("lease_type"),
            "net_revenue_interest": asset.get("net_revenue_interest"),
            "basin":                loc.get("basin"),
            "sub_basin":            loc.get("sub_basin"),
            "state":                loc.get("state"),
            "county":               loc.get("county"),
            "region":               loc.get("region"),
        })

        # ── Personnel table ───────────────────────────────────────────────────
        for role_key, person in asset.get("assigned_personnel", {}).items():
            personnel_rows.append({
                "organization_id": org_id,
                "asset_id":        asset_id,
                "well_id":         asset_id,
                "role_key":        role_key,
                "name":            person.get("name"),
                "role":            person.get("role"),
                "email":           person.get("email"),
            })

        # ── History rows ──────────────────────────────────────────────────────
        for day in asset.get("history", []):
            date_str  = day["date"]
            timestamp = date_str + " 00:00:00"  # daily data — midnight timestamp
            run_status= day.get("run_status")

            if asset.get("type") == "Well" and lift in SENSOR_FIELDS:
                # WELL_SENSOR_READINGS
                sensor_row = {
                    "organization_id": org_id,
                    "well_id":         asset_id,
                    "asset_id":        asset_id,
                    "date":            date_str,
                    "timestamp":       timestamp,
                    "lift_type":       lift,
                }
                for field in SENSOR_FIELDS[lift]:
                    sensor_row[field] = day.get(field)
                sensor_rows.append(sensor_row)

                # WELL_DAILY_PRODUCTION
                prod_row = {
                    "organization_id": org_id,
                    "well_id":         asset_id,
                    "asset_id":        asset_id,
                    "date":            date_str,
                    "timestamp":       timestamp,
                    "lift_type":       lift,
                }
                for field in PRODUCTION_FIELDS:
                    prod_row[field] = day.get(field)
                production_rows.append(prod_row)

            else:
                # Supporting assets — Facility, Compressor, SWD, Tank Battery
                row = {
                    "organization_id": org_id,
                    "asset_id":        asset_id,
                    "asset_type":      asset.get("type"),
                    "date":            date_str,
                    "timestamp":       timestamp,
                }
                row.update({k: v for k, v in day.items()
                             if k not in ("date", "timestamp", "organization_id")})
                supporting_rows.append(row)

        # ── Invoices table ────────────────────────────────────────────────────
        for inv in asset.get("financial_ledger", []):
            row = {
                "organization_id": org_id,
                "asset_id":        asset_id,
                "well_id":         asset_id,
            }
            row.update(inv)
            invoice_rows.append(row)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    pd.DataFrame(assets_rows).to_csv(
        "south_texas_assets.csv", index=False)
    pd.DataFrame(personnel_rows).to_csv(
        "south_texas_personnel.csv", index=False)
    pd.DataFrame(sensor_rows).to_csv(
        "south_texas_well_sensor_readings.csv", index=False)
    pd.DataFrame(production_rows).to_csv(
        "south_texas_well_daily_production.csv", index=False)
    pd.DataFrame(supporting_rows).to_csv(
        "south_texas_supporting_history.csv", index=False)
    pd.DataFrame(invoice_rows).to_csv(
        "south_texas_invoices.csv", index=False)

    print("✓ Done — 6 CSV files written (aligned to Snowflake table structure)")
    print(f"  south_texas_assets.csv:                  {len(assets_rows):,} rows")
    print(f"  south_texas_personnel.csv:               {len(personnel_rows):,} rows")
    print(f"  south_texas_well_sensor_readings.csv:    {len(sensor_rows):,} rows")
    print(f"  south_texas_well_daily_production.csv:   {len(production_rows):,} rows")
    print(f"  south_texas_supporting_history.csv:      {len(supporting_rows):,} rows")
    print(f"  south_texas_invoices.csv:                {len(invoice_rows):,} rows")
    print(f"\n  organization_id on all rows:             {STEP_ORGANIZATION_ID}")
    print(f"  well_id alias included for Snowflake joins")
    print(f"  timestamp field included for incremental watermark queries")
