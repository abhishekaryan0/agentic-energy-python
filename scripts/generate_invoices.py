import json
import random
from datetime import datetime

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

# --- MAIN EXECUTION LOGIC ---
def run_financial_engine():
    print("Loading Engineering Digital Twin...")
    with open("apex_permian_v3_1_engineering.json", "r") as f:
        data = json.load(f)

    all_invoices = []

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

            # 1. Check for Recurring LOE (Trigger on the 28th of every month)
            if current_date.endswith("-28"):
                all_invoices.append(build_recurring_invoice(asset_id, current_date))

            # 2. Check for Failures (Event-Driven Logic)
            if status == "FAILURE" and failure_start_date is None:
                # The well just broke! Record the date and reason.
                failure_start_date = current_date
                failure_note = notes
            
            elif status == "OK" and failure_start_date is not None:
                # The well is fixed! Generate the invoice for the downtime period.
                all_invoices.append(
                    build_event_invoice(asset_id, failure_start_date, current_date, failure_note)
                )
                # Reset the tracker
                failure_start_date = None

    # Save the output
    print(f"Generated {len(all_invoices)} highly realistic invoices.")
    with open("Invoices_Generated.json", "w") as f:
        json.dump(all_invoices, f, indent=2)
    print("Saved to Invoices_Generated.json")

if __name__ == "__main__":
    run_financial_engine()