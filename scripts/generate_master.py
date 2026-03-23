import json
import random
from datetime import datetime

# --- CONFIGURATION ---
COMPANY = "Apex Permian"

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

# --- 3. MASTER BUILDER LOGIC ---
def build_master_dataset():
    print("Loading Engineering Base Dataset...")
    try:
        with open("apex_permian_v3_1_engineering.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find apex_permian_v3_1_engineering.json. Please ensure it is in the same folder.")
        return

    all_invoices = []
    
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
    with open("Invoices_Generated_v2.json", "w") as f:
        json.dump(all_invoices, f, indent=2)
        
    # Output 2: The Unified Master File
    print("Packaging Unified Master Dataset...")
    with open("Master.json", "w") as f:
        json.dump(data, f, indent=2)
        
    print("Done! Master.json and Invoices_Generated_v2.json have been successfully created.")

if __name__ == "__main__":
    build_master_dataset()