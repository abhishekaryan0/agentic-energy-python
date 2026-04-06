import io
import json
import os
import sys
import random
import argparse
import pandas as pd
from datetime import datetime

_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

_env_path = os.path.join(_sys_root, ".env")
if os.path.isfile(_env_path):
    from dotenv import load_dotenv

    load_dotenv(_env_path)

# --- CONFIGURATION & ACCOUNTING DICTIONARIES ---
# Standardized rates and codes to ensure realism
RECURRING_GL = "605110"  # Base LOE / Utilities
WORKOVER_GL = "710520"   # Workover Expense

def generate_invoice_id():
    return f"APENR-2026-00{random.randint(1000, 9999)}"

def build_recurring_invoice(asset_id, date_str):
    """Generates the monthly baseline LOE invoice for an asset."""
    base_cost = round(random.uniform(4500.0, 6500.0), 2)
    return {
        "invoice_id": generate_invoice_id(),
        "invoice_date": date_str,
        "status": "Approved",
        "category": "Base LOE",
        "service_description": "Monthly recurring expenses: Electricity, SWD hauling, and contract pumper fees.",
        "site_asset_info": {"asset_id": asset_id},
        "vendor": {
            "vendor_name": "Permian Basin Utilities & Services",
            "vendor_type": "Recurring Operations"
        },
        "cost_allocation": {
            "gl_code": RECURRING_GL,
            "cost_type": "Operating Expense",
            "capex_opex_flag": "OPEX"
        },
        "line_items": [
            {"description": "Electrical Power", "quantity": 1.0, "unit_price_usd": base_cost * 0.4, "line_total_usd": round(base_cost * 0.4, 2)},
            {"description": "Water Disposal / Hauling", "quantity": 1.0, "unit_price_usd": base_cost * 0.6, "line_total_usd": round(base_cost * 0.6, 2)}
        ],
        "subtotal_usd": base_cost,
        "total_usd": base_cost
    }

def build_event_invoice(asset_id, start_date, end_date, note):
    """Generates a high-cost invoice triggered by an engineering failure."""
    # Determine cost based on the failure note from the engineering script
    if "Parted" in note or "Stuck" in note:
        cost = round(random.uniform(15000.0, 25000.0), 2)
        desc = f"Workover Rig: {note.replace('ALERT: ', '')} Return to production."
    elif "Shaft" in note or "Short" in note:
        cost = round(random.uniform(40000.0, 60000.0), 2)
        desc = "ESP Pull and Replace: Motor/Cable failure."
    else:
        cost = round(random.uniform(5000.0, 10000.0), 2)
        desc = f"Field Repair Service: {note}"

    return {
        "invoice_id": generate_invoice_id(),
        "invoice_date": end_date,  # Billed when work is complete
        "status": "Approved",
        "category": "Work Over / Repair",
        "service_description": desc,
        "service_period_start": start_date,
        "service_period_end": end_date,
        "site_asset_info": {"asset_id": asset_id},
        "vendor": {
            "vendor_name": "High Plains Pumping & Workover LLC",
            "vendor_type": "Workover / Pump Service"
        },
        "cost_allocation": {
            "gl_code": WORKOVER_GL,
            "cost_type": "Operating Expense",
            "capex_opex_flag": "OPEX"
        },
        "line_items": [
            {"description": "Rig Time & Labor", "quantity": 1.0, "unit_price_usd": cost, "line_total_usd": cost}
        ],
        "subtotal_usd": cost,
        "total_usd": cost
    }

S3_SENSOR_DATA_PREFIX = "sensor_data"

def upload_invoices_to_s3(invoices, organization_id=None):
    organization_id = organization_id or os.getenv("ORGANIZATION_ID")
    if not organization_id:
        print("ORGANIZATION_ID not set (env or argument). Skipping S3 upload.")
        return
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        print("S3_BUCKET not set (.env). Skipping S3 upload.")
        return
    try:
        import boto3
    except ImportError:
        print("boto3 not installed. pip install boto3")
        return

    # Invoice-level flattening so it matches the same schema as invoices_v2.
    # Missing fields are stored as NULL (None), not 0.
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

    records = []
    for inv in invoices:
        line_items = inv.get("line_items") or []
        line_items_count = len(line_items) if isinstance(line_items, list) else None
        line_total_usd_values = []
        if isinstance(line_items, list):
            for item in line_items:
                if not isinstance(item, dict):
                    continue
                if item.get("line_total_usd") is not None:
                    line_total_usd_values.append(_to_float_or_none(item.get("line_total_usd")))
        line_total_usd_sum = (
            float(sum(v for v in line_total_usd_values if v is not None))
            if line_total_usd_values
            else None
        )

        records.append(
            {
                "organization_id": organization_id,
                "invoice_version": "v1",
                "invoice_id": inv.get("invoice_id"),
                "invoice_date": _to_date_or_none(inv.get("invoice_date")),
                "status": inv.get("status"),
                "category": inv.get("category"),
                "service_description": inv.get("service_description"),
                "service_period_start": _to_date_or_none(inv.get("service_period_start")),
                "service_period_end": _to_date_or_none(inv.get("service_period_end")),
                "asset_id": inv.get("site_asset_info", {}).get("asset_id"),
                "vendor_name": inv.get("vendor", {}).get("vendor_name"),
                "vendor_type": inv.get("vendor", {}).get("vendor_type"),
                "total_usd": _to_float_or_none(inv.get("total_usd")),
                "subtotal_usd": _to_float_or_none(inv.get("subtotal_usd")),
                "gl_code": inv.get("cost_allocation", {}).get("gl_code"),
                "cost_type": inv.get("cost_allocation", {}).get("cost_type"),
                "capex_opex_flag": inv.get("cost_allocation", {}).get("capex_opex_flag"),
                "line_items_count": line_items_count,
                "line_total_usd_sum": line_total_usd_sum,
                "raw_invoice_json": json.dumps(inv, default=str),
            }
        )
    
    if not records:
        print("No invoice records to upload.")
        return

    df = pd.DataFrame(records)
    # Ensure pandas NaN becomes parquet NULLs (None).
    df = df.where(pd.notnull(df), None)
    
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    s3 = boto3.client("s3")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{organization_id}/financial_data/run_ts={ts}"
    key = f"{prefix}/invoices_v1.parquet"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

    print(f"SUCCESS: Invoices parquet uploaded to s3://{bucket}/{key} ({len(df):,} rows)")


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


# --- MAIN EXECUTION LOGIC ---
def run_financial_engine(upload_s3=False, write_json=True, organization_id=None):
    all_invoices = []

    org_id = organization_id or os.getenv("ORGANIZATION_ID")

    if upload_s3 and org_id:
        print("Loading daily_production from S3 for invoice generation...")
        daily_df = _load_daily_production_from_s3(org_id)
        # Normalize types for consistent comparisons/formatting
        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df["asset_id"] = daily_df["asset_id"].astype(str)

        print("Listening for events and generating invoices...")
        for asset_id in daily_df["asset_id"].dropna().unique():
            well_hist = daily_df[daily_df["asset_id"] == asset_id].sort_values("date")
            failure_start_date = None
            failure_note = ""

            for _, day in well_hist.iterrows():
                current_date = pd.to_datetime(day["date"]).date().isoformat()
                status = day.get("status")
                status = str(status) if status is not None and not pd.isna(status) else ""
                notes = day.get("notes", "")
                notes = "" if notes is None or (isinstance(notes, float) and pd.isna(notes)) else str(notes)

                # 1) Recurring LOE on the 28th of every month
                if current_date.endswith("-28"):
                    all_invoices.append(build_recurring_invoice(asset_id, current_date))

                # 2) Failures: FAILURE -> next OK generates a workover invoice
                if status == "FAILURE" and failure_start_date is None:
                    failure_start_date = current_date
                    failure_note = notes
                elif status == "OK" and failure_start_date is not None:
                    all_invoices.append(build_event_invoice(asset_id, failure_start_date, current_date, failure_note))
                    failure_start_date = None
    else:
        # Fallback: use local engineering JSON (legacy behavior).
        print("Loading Engineering Digital Twin JSON...")
        with open("apex_permian_v3_1_engineering.json", "r") as f:
            data = json.load(f)

        print("Listening for events and generating invoices...")
        for asset_id, asset_data in data["assets"].items():
            if asset_data.get("type") != "Well":
                continue

            history = asset_data.get("history", [])
            failure_start_date = None
            failure_note = ""

            for day in history:
                current_date = day["date"]
                status = day["status"]
                notes = day.get("notes", "")

                # 1) Recurring LOE on the 28th of every month
                if current_date.endswith("-28"):
                    all_invoices.append(build_recurring_invoice(asset_id, current_date))

                # 2) Failures: FAILURE -> next OK generates a workover invoice
                if status == "FAILURE" and failure_start_date is None:
                    failure_start_date = current_date
                    failure_note = notes
                elif status == "OK" and failure_start_date is not None:
                    all_invoices.append(build_event_invoice(asset_id, failure_start_date, current_date, failure_note))
                    failure_start_date = None

    if write_json:
        # Save the output
        print(f"Generated {len(all_invoices)} highly realistic invoices.")
        with open("Invoices_Generated.json", "w") as f:
            json.dump(all_invoices, f, indent=2)
        print("Saved to Invoices_Generated.json")

    org_id = organization_id or os.getenv("ORGANIZATION_ID")
    if upload_s3:
        upload_invoices_to_s3(all_invoices, organization_id=org_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate invoices data (JSON, S3 parquet)."
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Upload invoices parquet to S3 under {ORGANIZATION_ID}/financial_data/; uses S3_BUCKET and ORGANIZATION_ID from .env",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write Invoices_Generated.json to current directory",
    )
    parser.add_argument(
        "--organization-id",
        default=os.getenv("ORGANIZATION_ID"),
        help="Organization/tenant UUID. Default: ORGANIZATION_ID env.",
    )
    args = parser.parse_args()
    if args.organization_id:
        os.environ["ORGANIZATION_ID"] = args.organization_id
    run_financial_engine(
        upload_s3=args.s3,
        write_json=args.json,
        organization_id=args.organization_id,
    )