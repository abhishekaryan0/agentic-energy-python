import io
import json
import os
import sys
import random
import argparse
from datetime import datetime

import pandas as pd

# --- CONFIGURATION ---
COMPANY = "Apex Permian"

_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

_env_path = os.path.join(_sys_root, ".env")
if os.path.isfile(_env_path):
    from dotenv import load_dotenv

    load_dotenv(_env_path)


INVOICES_PARQUET_COLUMNS = [
    "organization_id",
    "invoice_version",
    "invoice_id",
    "invoice_date",
    "status",
    "category",
    "service_description",
    "service_period_start",
    "service_period_end",
    "asset_id",
    "vendor_name",
    "vendor_type",
    "total_usd",
    "subtotal_usd",
    "gl_code",
    "cost_type",
    "capex_opex_flag",
    "line_items_count",
    "line_total_usd_sum",
    "raw_invoice_json",
]


def _to_float_or_none(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_date_or_none(v):
    if not v:
        return None
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def _upload_df_to_s3_parquet(df: pd.DataFrame, bucket: str, key: str) -> None:
    import boto3

    df = df.where(pd.notnull(df), None)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def _upload_invoices_v2_to_s3(all_invoices: list, organization_id: str) -> str:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET env var is required for invoices upload")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{organization_id}/financial_data/run_ts={ts}"
    key = f"{prefix}/invoices_v2.parquet"

    records = []
    for inv in all_invoices:
        records.append(
            {
                "organization_id": organization_id,
                "invoice_version": "v2",
                "invoice_id": inv.get("invoice_id"),
                "invoice_date": _to_date_or_none(inv.get("invoice_date")),
                "status": inv.get("status"),
                "category": inv.get("category"),
                "service_description": inv.get("service_description"),
                "service_period_start": None,
                "service_period_end": None,
                "asset_id": inv.get("site_asset_info", {}).get("asset_id"),
                "vendor_name": inv.get("vendor", {}).get("vendor_name"),
                "vendor_type": None,
                "total_usd": _to_float_or_none(inv.get("total_usd")),
                "subtotal_usd": None,
                "gl_code": None,
                "cost_type": None,
                "capex_opex_flag": None,
                "line_items_count": None,
                "line_total_usd_sum": None,
                "raw_invoice_json": json.dumps(inv, default=str),
            }
        )

    df = pd.DataFrame(records, columns=INVOICES_PARQUET_COLUMNS)
    _upload_df_to_s3_parquet(df, bucket=bucket, key=key)
    return key


def _upload_master_parquet_to_s3(data: dict, organization_id: str) -> dict:
    """
    Store master history and asset metadata as parquet.
    Missing fields become NULL to keep schema stable.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET env var is required for master upload")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{organization_id}/digital_twin/master_data/run_ts={ts}"

    # -------------------
    # Assets parquet
    # -------------------
    asset_rows = []
    for asset_id, asset in (data.get("assets") or {}).items():
        asset_rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "asset_type": asset.get("type"),
                "lift_type": asset.get("lift"),
                "assigned_personnel_json": json.dumps(asset.get("assigned_personnel")) if asset.get("assigned_personnel") is not None else None,
                "financial_ledger_json": json.dumps(asset.get("financial_ledger")) if asset.get("financial_ledger") is not None else None,
                "raw_asset_json": json.dumps(asset, default=str),
            }
        )
    df_assets = pd.DataFrame(asset_rows)
    key_assets = f"{prefix}/master_assets.parquet"
    _upload_df_to_s3_parquet(df_assets, bucket=bucket, key=key_assets)

    # -------------------
    # History parquet
    # -------------------
    history_rows = []
    for asset_id, asset in (data.get("assets") or {}).items():
        asset_type = asset.get("type")
        lift_type = asset.get("lift") if asset_type == "Well" else None
        for h in asset.get("history") or []:
            base = {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "asset_type": asset_type,
                "lift_type": lift_type,
                "date": _to_date_or_none(h.get("date")),
                "status": h.get("status"),
                "notes": h.get("notes"),
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
            }

            if asset_type == "Facility":
                base["oil_prod"] = _to_float_or_none(h.get("oil_prod"))
            elif asset_type == "Well":
                base.update(
                    {
                        "allocated_oil": _to_float_or_none(h.get("allocated_oil")),
                        "test_oil": _to_float_or_none(h.get("test_oil")),
                        "avg_tubing_pressure": _to_float_or_none(h.get("avg_tubing_pressure")),
                        "avg_casing_pressure": _to_float_or_none(h.get("avg_casing_pressure")),
                        "avg_motor_amps": _to_float_or_none(h.get("avg_motor_amps")),
                        "gross_stroke_len": _to_float_or_none(h.get("gross_stroke_len")),
                        "net_stroke_len": _to_float_or_none(h.get("net_stroke_len")),
                        "spm": _to_float_or_none(h.get("spm")),
                        "pump_fillage_pct": _to_float_or_none(h.get("pump_fillage_pct")),
                        "freq_hz": _to_float_or_none(h.get("freq_hz")),
                        "pump_intake_pressure": _to_float_or_none(h.get("pump_intake_pressure")),
                        "motor_temp_f": _to_float_or_none(h.get("motor_temp_f")),
                        "injection_rate_mcf": _to_float_or_none(h.get("injection_rate_mcf")),
                        "notes": h.get("notes"),
                    }
                )

            history_rows.append(base | {"raw_history_json": json.dumps(h, default=str)})

    df_history = pd.DataFrame(history_rows)
    key_history = f"{prefix}/master_history.parquet"
    _upload_df_to_s3_parquet(df_history, bucket=bucket, key=key_history)

    return {"assets_key": key_assets, "history_key": key_history}

def generate_invoice_id():
    return f"APENR-{random.randint(2014, 2026)}-{random.randint(10000, 99999)}"

# --- 1. PERSONNEL BUILDER ---
def assign_personnel(asset_id):
    """Assigns synthetic personnel to an asset."""
    # Extract lease name to assign consistent operators (e.g., "Mustang" from "Mustang-12-1-H")
    lease = asset_id.split("-")[0] if "-" in asset_id else "Permian"
    
    return {
        "lease_manager": {
            "name": f"{lease} Operator {random.choice(['A', 'B', 'C'])}",
            "role": "Lease Operator",
            "email": f"operator.{lease.lower()}@apexpermian.com",
            "handle": f"@{lease.lower()}_ops"
        },
        "production_engineer": {
            "name": random.choice(["James Hernandez", "Sofia Patel", "Michael Chen"]),
            "role": "Production Engineer",
            "email": "engineering@apexpermian.com",
            "handle": "@prod_eng"
        }
    }

# --- 2. INVOICE GENERATORS ---
def create_invoice(asset_id, date_str, category, desc, cost, vendor):
    """Helper function to standardize invoice creation."""
    return {
        "invoice_id": generate_invoice_id(),
        "invoice_date": date_str,
        "status": random.choice(["Approved", "Approved", "Approved", "In Review"]),
        "category": category,
        "service_description": desc,
        "site_asset_info": {"asset_id": asset_id},
        "vendor": {"vendor_name": vendor},
        "total_usd": round(cost, 2)
    }

def generate_monthly_invoices(asset_id, date_str):
    """Generates 5-7 distinct recurring invoices for the end of the month."""
    invoices = []
    
    # 1. Utility/Electrical
    invoices.append(create_invoice(asset_id, date_str, "Utilities", "Monthly electrical power consumption.", random.uniform(1500, 3500), "Permian Basin Power Co."))
    
    # 2. Water Hauling/Disposal
    invoices.append(create_invoice(asset_id, date_str, "Water Management", "SWD disposal fees and trucking.", random.uniform(2000, 6000), "Blue Water Logistics"))
    
    # 3. Lease Manager / Pumper Fee
    invoices.append(create_invoice(asset_id, date_str, "Labor", "Contract pumper and lease operator allocation.", random.uniform(800, 1500), "Apex Permian Internal Allocation"))
    
    # 4. Engineering Allocation
    invoices.append(create_invoice(asset_id, date_str, "Labor", "Production engineering support allocation.", random.uniform(500, 1200), "Apex Permian Internal Allocation"))
    
    # 5. G&A Overhead
    invoices.append(create_invoice(asset_id, date_str, "G&A", "Corporate overhead and administrative allocation.", random.uniform(300, 800), "Apex Permian Internal Allocation"))
    
    # 6. Telemetry/SCADA
    invoices.append(create_invoice(asset_id, date_str, "Monitoring", "SCADA cellular telemetry and software licensing.", random.uniform(100, 250), "Oilfield Tech Solutions"))
    
    # 7. Equipment Rentals (70% chance to have a rental on site)
    if random.random() < 0.70:
        invoices.append(create_invoice(asset_id, date_str, "Rentals", "Surface equipment / generator rental.", random.uniform(400, 1200), "Permian Rental Supply"))

    return invoices

def generate_event_invoice(asset_id, start_date, end_date, note):
    """Generates a high-cost workover invoice for mechanical failures."""
    cost = random.uniform(25000.0, 75000.0) if "ESP" in note or "Shaft" in note else random.uniform(12000.0, 25000.0)
    return create_invoice(asset_id, end_date, "Workover", f"Rig Time & Labor: {note.replace('ALERT: ', '')}", cost, "High Plains Pumping & Workover LLC")

def _load_daily_production_from_s3(organization_id: str) -> pd.DataFrame:
    """Load s3://{S3_BUCKET}/{ORGANIZATION_ID}/sensor_data/daily_production.parquet into a DataFrame."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET env var is required to load daily_production.parquet")

    key = f"{organization_id}/sensor_data/daily_production.parquet"
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed; cannot load parquet from S3") from e

    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=buf)
    buf.seek(0)
    return pd.read_parquet(buf)


# --- 3. MASTER BUILDER LOGIC ---
def build_master_dataset(write_json: bool = False, input_from_s3: bool = False):
    all_invoices = []

    if input_from_s3:
        org_id = os.getenv("ORGANIZATION_ID")
        if not org_id:
            raise ValueError("ORGANIZATION_ID env var is required for master build from S3")

        print("Loading daily_production from S3 for master build...")
        daily_df = _load_daily_production_from_s3(org_id)
        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df["asset_id"] = daily_df["asset_id"].astype(str)
        daily_df["facility_id"] = daily_df["facility_id"].astype(str)

        data = {"company": COMPANY, "assets": {}}

        # Facilities
        for fid in daily_df["facility_id"].dropna().unique():
            data["assets"][fid] = {"type": "Facility", "history": []}

        # Wells + per-well lift_type
        for well_id in daily_df["asset_id"].dropna().unique():
            wdf = daily_df[daily_df["asset_id"] == well_id]
            lift_type = None
            if "lift_type" in wdf.columns:
                lift_type = str(wdf["lift_type"].dropna().iloc[0]) if not wdf["lift_type"].dropna().empty else None
            data["assets"][well_id] = {"type": "Well", "lift": lift_type, "history": []}

        # Facility-day ESD decisions:
        # in digital twin uploads, ESD is represented by setting well status to SHUT_IN for all wells on that day.
        facility_day_esd = {}
        grouped_fac = daily_df.groupby(["facility_id", "date"], dropna=False)
        for (fid, date), g in grouped_fac:
            statuses = g.get("status")
            if statuses is None:
                facility_day_esd[(fid, date)] = False
            else:
                facility_day_esd[(fid, date)] = bool((statuses == "SHUT_IN").all())

        # Build facility history + well history
        for (fid, date), g in grouped_fac:
            d_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            esd = facility_day_esd.get((fid, date), False)
            total_oil = 0.0 if esd else float(g.get("true_oil", pd.Series([0.0])).sum())

            data["assets"][fid]["history"].append(
                {"date": d_str, "oil_prod": round(total_oil, 2), "status": "ESD" if esd else "OK"}
            )

            if "true_oil" not in g.columns:
                continue

            # add each well's daily history entry
            for _, row in g.iterrows():
                well_id = str(row["asset_id"])
                allocated_oil = 0.0 if esd else _to_float_or_none(row.get("true_oil"))
                entry = {
                    "date": d_str,
                    "allocated_oil": round(allocated_oil or 0.0, 2),
                    "test_oil": _to_float_or_none(row.get("test_oil")),
                    "avg_tubing_pressure": _to_float_or_none(row.get("avg_tubing_pressure")),
                    "avg_casing_pressure": _to_float_or_none(row.get("avg_casing_pressure")),
                    "avg_motor_amps": _to_float_or_none(row.get("avg_motor_amps")),
                    "gross_stroke_len": _to_float_or_none(row.get("gross_stroke_len")),
                    "net_stroke_len": _to_float_or_none(row.get("net_stroke_len")),
                    "spm": _to_float_or_none(row.get("spm")),
                    "pump_fillage_pct": _to_float_or_none(row.get("pump_fillage_pct")),
                    "freq_hz": _to_float_or_none(row.get("freq_hz")),
                    "pump_intake_pressure": _to_float_or_none(row.get("pump_intake_pressure")),
                    "motor_temp_f": _to_float_or_none(row.get("motor_temp_f")),
                    "injection_rate_mcf": _to_float_or_none(row.get("injection_rate_mcf")),
                    "status": "SHUT_IN" if esd else row.get("status"),
                    "notes": row.get("notes"),
                }
                data["assets"][well_id]["history"].append(entry)

        # Add personnel + financial ledger (invoices) per well
        for well_id, asset_data in list(data["assets"].items()):
            if asset_data.get("type") != "Well":
                continue

            asset_data["assigned_personnel"] = assign_personnel(well_id)
            asset_invoices = []
            failure_start = None
            failure_note = ""

            # Ensure chronological order
            history = sorted(asset_data.get("history", []), key=lambda x: str(x.get("date")))

            for day in history:
                current_date = day["date"]
                status = day.get("status")
                notes = day.get("notes", "")
                notes = "" if notes is None or (isinstance(notes, float) and pd.isna(notes)) else str(notes)
                status_str = "" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status)

                if current_date.endswith("-28"):
                    asset_invoices.extend(generate_monthly_invoices(well_id, current_date))

                rand_chance = random.random()
                if rand_chance < 0.005:
                    asset_invoices.append(
                        create_invoice(
                            well_id,
                            current_date,
                            "Maintenance",
                            "Incidental roustabout work: Surface vessel valve replacement/maintenance.",
                            random.uniform(1500, 4500),
                            "WestTex Roustabout Services",
                        )
                    )
                elif rand_chance < 0.001:
                    asset_invoices.append(
                        create_invoice(
                            well_id,
                            current_date,
                            "HSE",
                            "HSE / Soil Remediation: Clean up minor leak at wellhead packing.",
                            random.uniform(5000, 15000),
                            "Permian Environmental Cleaners",
                        )
                    )

                if status_str == "FAILURE" and failure_start is None:
                    failure_start = current_date
                    failure_note = notes
                elif status_str == "OK" and failure_start is not None:
                    asset_invoices.append(generate_event_invoice(well_id, failure_start, current_date, failure_note))
                    failure_start = None

            asset_data["financial_ledger"] = asset_invoices
            all_invoices.extend(asset_invoices)

        # Optionally write local JSON outputs
        if write_json:
            print("Packaging Unified Master Dataset (local JSON)...")
            with open("Invoices_Generated_v2.json", "w") as f:
                json.dump(all_invoices, f, indent=2)
            with open("Master.json", "w") as f:
                json.dump(data, f, indent=2)
            print("Done! Master.json and Invoices_Generated_v2.json have been successfully created.")

        return data, all_invoices

    # Legacy behavior: load engineering JSON from local disk
    print("Loading Engineering Base Dataset (local JSON)...")
    try:
        with open("apex_permian_v3_1_engineering.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find apex_permian_v3_1_engineering.json. Please ensure it is in the same folder or run with --s3.")
        return None, None

    print("Enriching assets with personnel and generating granular financial ledger...")
    
    for asset_id, asset_data in data["assets"].items():
        if asset_data.get("type") != "Well":
            continue
            
        # 1. Assign Personnel to the Asset
        asset_data["assigned_personnel"] = assign_personnel(asset_id)
        
        # We will attach invoices directly to the asset in the Master file, 
        # but also keep them in the massive all_invoices list for the ledger.
        asset_invoices = []
        
        history = asset_data.get("history", [])
        failure_start = None
        failure_note = ""

        for day in history:
            current_date = day["date"]
            status = day["status"]
            notes = day.get("notes", "")

            # A. Monthly Recurring Loop (5-7 invoices on the 28th)
            if current_date.endswith("-28"):
                monthly_invs = generate_monthly_invoices(asset_id, current_date)
                asset_invoices.extend(monthly_invs)

            # B. Ad-Hoc / Exception Invoices (Random Daily Probability)
            rand_chance = random.random()
            if rand_chance < 0.005: # Approx 1.5 times a year
                asset_invoices.append(create_invoice(asset_id, current_date, "Maintenance", "Incidental roustabout work: Surface vessel valve replacement/maintenance.", random.uniform(1500, 4500), "WestTex Roustabout Services"))
            elif rand_chance < 0.001: # Very rare (approx once every 3 years)
                asset_invoices.append(create_invoice(asset_id, current_date, "HSE", "HSE / Soil Remediation: Clean up minor leak at wellhead packing.", random.uniform(5000, 15000), "Permian Environmental Cleaners"))

            # C. Physics-Driven Workovers (Failures)
            if status == "FAILURE" and failure_start is None:
                failure_start = current_date
                failure_note = notes
            elif status == "OK" and failure_start is not None:
                asset_invoices.append(generate_event_invoice(asset_id, failure_start, current_date, failure_note))
                failure_start = None
                
        # Attach the asset's specific invoices to its object in the Master dict
        asset_data["financial_ledger"] = asset_invoices
        all_invoices.extend(asset_invoices)

    # Output 1: The New Massive Invoice Ledger
    print(f"Generated {len(all_invoices)} highly granular invoices.")
    if write_json:
        with open("Invoices_Generated_v2.json", "w") as f:
            json.dump(all_invoices, f, indent=2)
        
        # Output 2: The Unified Master File
        print("Packaging Unified Master Dataset...")
        with open("Master.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print("Done! Master.json and Invoices_Generated_v2.json have been successfully created.")
    return data, all_invoices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Master.json + S3 parquet datasets")
    parser.add_argument("--s3", action="store_true", help="Upload parquet to S3")
    parser.add_argument("--json", action="store_true", help="Write local JSON outputs (Master.json + Invoices_Generated_v2.json)")
    parser.add_argument(
        "--organization-id",
        default=os.getenv("ORGANIZATION_ID"),
        help="Organization/tenant UUID. Default: ORGANIZATION_ID env.",
    )
    args = parser.parse_args()

    if args.organization_id:
        os.environ["ORGANIZATION_ID"] = args.organization_id

    data, all_invoices = build_master_dataset(write_json=args.json, input_from_s3=args.s3)

    if args.s3:
        org_id = os.getenv("ORGANIZATION_ID")
        if not org_id:
            raise ValueError("ORGANIZATION_ID env var is required for S3 upload")
        # invoices + master history/asset metadata
        _upload_invoices_v2_to_s3(all_invoices, organization_id=org_id)
        _upload_master_parquet_to_s3(data, organization_id=org_id)
        print("SUCCESS: Uploaded invoices_v2 + master parquet to S3.")