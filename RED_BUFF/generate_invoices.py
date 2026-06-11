import json
import random

EXPENSE_LIBRARY = {
    "Pumping Unit": {
        "Monthly": [
            {"desc": "Electrical Power - Surface",              "gl": "605-110", "range": (1200, 2800),  "vendor": "Trans-Pecos Electric"},
            {"desc": "Production Technician Labor Allocation",  "gl": "605-210", "range": (800,  1200),  "vendor": "Red Bluff Internal"},
            {"desc": "Chemical Treatment - Corrosion/Scale",    "gl": "605-310", "range": (400,  900),   "vendor": "Permian Chemical Supply"},
            {"desc": "SCADA / RTU Telemetry",                   "gl": "605-410", "range": (150,  250),   "vendor": "Basin Telemetry & Controls"},
            {"desc": "G&A Overhead Allocation",                 "gl": "822-110", "range": (300,  800),   "vendor": "Red Bluff Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "Produced Water Hauling",                  "gl": "605-510", "range": (1500, 4000),  "vendor": "Reeves County Water Hauling", "prob": 0.85},
            {"desc": "Hot Oil Treatment - Paraffin",            "gl": "605-610", "range": (800,  1500),  "vendor": "Midland Hot Oil Service",      "prob": 0.30},
            {"desc": "Roustabout - Location Maintenance",       "gl": "605-710", "range": (300,  600),   "vendor": "Pecos Valley Roustabout",      "prob": 0.15},
        ],
        "Workover": {
            "desc": "Rod Job: Parted Rods / Pump R&R",
            "gl":   "710-520",
            "vendor": "Red Bluff Rig Services",
            "rig_rate": (550, 750),
            "rig_hours": (12, 24),
            "parts": (2000, 5000),
        }
    },
    "ESP": {
        "Monthly": [
            {"desc": "Electrical Power - High Draw",            "gl": "605-130", "range": (5000, 12000), "vendor": "Trans-Pecos Electric"},
            {"desc": "Production Technician Labor Allocation",  "gl": "605-210", "range": (800,  1200),  "vendor": "Red Bluff Internal"},
            {"desc": "Chemical Treatment - ESP Scale Inhibitor","gl": "605-330", "range": (1000, 2500),  "vendor": "Permian Chemical Supply"},
            {"desc": "VFD Monitoring & Controls",               "gl": "605-420", "range": (300,  600),   "vendor": "Basin Telemetry & Controls"},
            {"desc": "G&A Overhead Allocation",                 "gl": "822-110", "range": (300,  800),   "vendor": "Red Bluff Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "SWD Disposal Fee - High Volume",          "gl": "605-520", "range": (4000, 9000),  "vendor": "Reeves County Water Hauling", "prob": 0.90},
            {"desc": "VFD Calibration & Reset",                 "gl": "605-740", "range": (450,  900),   "vendor": "West Texas Electrical Services","prob": 0.25},
            {"desc": "Acid Treatment - Batch",                  "gl": "605-620", "range": (2500, 5000),  "vendor": "Permian Chemical Supply",    "prob": 0.15},
        ],
        "Workover": {
            "desc": "ESP Workover: Motor & Cable R&R",
            "gl":   "710-520",
            "vendor": "Red Bluff Rig Services",
            "rig_rate":  (550, 750),
            "rig_hours": (36, 72),
            "motor":     (45000, 70000),
            "cable":     (18000, 28000),
        }
    },
    "Gas Lift": {
        "Monthly": [
            {"desc": "Compression Fuel / Power",                "gl": "605-120", "range": (2000, 4500),  "vendor": "Trans-Pecos Electric"},
            {"desc": "Production Technician Labor Allocation",  "gl": "605-210", "range": (800,  1200),  "vendor": "Red Bluff Internal"},
            {"desc": "Chemical Injection - Continuous",         "gl": "605-310", "range": (600,  1100),  "vendor": "Permian Chemical Supply"},
            {"desc": "SCADA / RTU Telemetry",                   "gl": "605-410", "range": (150,  250),   "vendor": "Basin Telemetry & Controls"},
            {"desc": "G&A Overhead Allocation",                 "gl": "822-110", "range": (300,  800),   "vendor": "Red Bluff Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "Produced Water Hauling",                  "gl": "605-510", "range": (1500, 4000),  "vendor": "Reeves County Water Hauling", "prob": 0.85},
            {"desc": "Compressor Skid - PM Service",            "gl": "605-720", "range": (900,  1800),  "vendor": "Delaware Basin Compression",  "prob": 0.35},
            {"desc": "Freeze Protection - Methanol Injection",  "gl": "605-320", "range": (1200, 2500),  "vendor": "Permian Chemical Supply",    "prob": 0.15},
        ],
        "Workover": {
            "desc": "Slickline - Gas Lift Valve R&R",
            "gl":   "710-520",
            "vendor": "Red Bluff Rig Services",
            "rig_rate":  (400, 550),
            "rig_hours": (8,   16),
            "parts":     (3000, 7000),
        }
    }
}

def build_workover_invoice(asset_id, date_str, lift):
    lib  = EXPENSE_LIBRARY[lift]["Workover"]
    rate = random.uniform(*lib["rig_rate"])
    hrs  = random.uniform(*lib["rig_hours"])
    cost = rate * hrs
    if "motor" in lib:
        cost += random.uniform(*lib["motor"]) + random.uniform(*lib["cable"])
    elif "parts" in lib:
        cost += random.uniform(*lib["parts"])

    return {
        "invoice_id":          f"RBNR-{date_str[:4]}-{random.randint(10000,99999)}",
        "invoice_date":        date_str,
        "category":            "Workover",
        "gl_code":             lib["gl"],
        "service_description": lib["desc"],
        "vendor":              lib["vendor"],
        "total_usd":           round(cost, 2),
    }

if __name__ == "__main__":
    print("Generating v5 Financial Ledger...")
    with open("red_bluff_engineering.json") as f:
        data = json.load(f)

    invoices = []
    for asset_id, asset in data["assets"].items():
        lift       = asset["lift"]
        lib        = EXPENSE_LIBRARY[lift]
        fail_start = None

        for day in asset["history"]:
            date_str = day["date"]
            status   = day["run_status"]

            # Monthly LOE on the 28th
            if date_str.endswith("-28"):
                for exp in lib["Monthly"]:
                    invoices.append({
                        "invoice_id":          f"RBNR-{date_str[:4]}-{random.randint(10000,99999)}",
                        "invoice_date":        date_str,
                        "category":            "Monthly LOE",
                        "gl_code":             exp["gl"],
                        "service_description": exp["desc"],
                        "vendor":              exp["vendor"],
                        "total_usd":           round(random.uniform(*exp["range"]), 2),
                        "asset_id":            asset_id,
                    })
                for exp in lib["Ad_Hoc"]:
                    if random.random() < exp["prob"]:
                        invoices.append({
                            "invoice_id":          f"RBNR-{date_str[:4]}-{random.randint(10000,99999)}",
                            "invoice_date":        date_str,
                            "category":            "Ad-Hoc LOE",
                            "gl_code":             exp["gl"],
                            "service_description": exp["desc"],
                            "vendor":              exp["vendor"],
                            "total_usd":           round(random.uniform(*exp["range"]), 2),
                            "asset_id":            asset_id,
                        })

            # Workover triggered on return to PRODUCING after SHUT-IN
            if status == "SHUT-IN" and fail_start is None:
                fail_start = date_str
            elif status == "PRODUCING" and fail_start is not None:
                inv = build_workover_invoice(asset_id, date_str, lift)
                inv["asset_id"] = asset_id
                invoices.append(inv)
                fail_start = None

    print(f"  Generated {len(invoices):,} invoices across {len(data['assets'])} assets")

    # Group invoices by asset_id into financial_ledger
    invoice_map = {}
    for inv in invoices:
        aid = inv.pop("asset_id")
        if aid not in invoice_map:
            invoice_map[aid] = []
        invoice_map[aid].append(inv)

    for asset_id, asset in data["assets"].items():
        asset["financial_ledger"] = invoice_map.get(asset_id, [])

    with open("red_bluff_engineering.json", "w") as f:
        json.dump(data, f, indent=2)

    total_h = sum(len(a["history"])          for a in data["assets"].values())
    total_l = sum(len(a["financial_ledger"]) for a in data["assets"].values())
    print(f"  Daily production records: {total_h:,}")
    print(f"  Cost ledger entries:      {total_l:,}")

    with open("red_bluff_engineering.json") as f:
        lc = sum(1 for _ in f)
    print(f"  Total lines:              {lc:,}")
    print("✓ Done: red_bluff_engineering.json")
