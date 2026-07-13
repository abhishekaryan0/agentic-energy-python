import json
import random

# South Texas Energy Partners — South Texas regional vendors
# South Texas vendors: different chemical suppliers, different electrical,
# different compression, different water hauling, different rig services

EXPENSE_LIBRARY = {
    "Pumping Unit": {
        "Monthly": [
            {"desc": "Electrical Power - Surface Unit",          "gl": "605-110", "range": (900, 2400),   "vendor": "CPS Energy - South Texas"},
            {"desc": "Production Technician Labor Allocation",   "gl": "605-210", "range": (800, 1200),   "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Chemical Treatment - Scale/Corrosion",     "gl": "605-310", "range": (350, 850),    "vendor": "Monarch Chemical Services"},
            {"desc": "SCADA / RTU Telemetry",                    "gl": "605-410", "range": (150, 280),    "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation",                  "gl": "822-110", "range": (300, 800),    "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "Produced Water Hauling",                   "gl": "605-510", "range": (1200, 3500),  "vendor": "Eagle Ford Water Services",   "prob": 0.80},
            {"desc": "Paraffin Treatment - Hot Oil",             "gl": "605-610", "range": (900, 1800),   "vendor": "South Texas Hot Oil",          "prob": 0.40},
            {"desc": "Roustabout - Location Maintenance",        "gl": "605-710", "range": (300, 600),    "vendor": "Laredo Roustabout Services",   "prob": 0.20},
        ],
        "Workover": {
            "desc": "Pump R&R / Tubing Repair",
            "gl":   "710-520",
            "vendor": "Frontier Well Services",
            "rig_rate": (500, 700),
            "rig_hours": (10, 20),
            "parts": (1500, 4500),
        }
    },
    "ESP": {
        "Monthly": [
            {"desc": "Electrical Power - ESP High Draw",         "gl": "605-130", "range": (6000, 14000), "vendor": "CPS Energy - South Texas"},
            {"desc": "Production Technician Labor Allocation",   "gl": "605-210", "range": (800, 1200),   "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Chemical Treatment - ESP Scale Inhibitor", "gl": "605-330", "range": (1200, 3000),  "vendor": "Monarch Chemical Services"},
            {"desc": "VFD Monitoring & Controls",                "gl": "605-420", "range": (300, 650),    "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation",                  "gl": "822-110", "range": (300, 800),    "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "SWD Disposal Fee - High Volume",           "gl": "605-520", "range": (5000, 12000), "vendor": "Eagle Ford Water Services",   "prob": 0.90},
            {"desc": "VFD Calibration & Service",                "gl": "605-740", "range": (500, 1100),   "vendor": "South Texas Electrical LLC",  "prob": 0.25},
            {"desc": "Scale Squeeze Treatment",                  "gl": "605-620", "range": (3000, 7000),  "vendor": "Monarch Chemical Services",   "prob": 0.20},
        ],
        "Workover": {
            "desc": "ESP Workover: Motor, Pump & Cable R&R",
            "gl":   "710-520",
            "vendor": "Frontier Well Services",
            "rig_rate":  (550, 750),
            "rig_hours": (36, 72),
            "motor":     (50000, 80000),
            "cable":     (20000, 35000),
        }
    },
    "Gas Lift": {
        "Monthly": [
            {"desc": "Compression Fuel / Power",                 "gl": "605-120", "range": (2500, 6000),  "vendor": "CPS Energy - South Texas"},
            {"desc": "Production Technician Labor Allocation",   "gl": "605-210", "range": (800, 1200),   "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Chemical Injection - Paraffin Inhibitor",  "gl": "605-315", "range": (700, 1400),   "vendor": "Monarch Chemical Services"},
            {"desc": "SCADA / RTU Telemetry",                    "gl": "605-410", "range": (150, 280),    "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation",                  "gl": "822-110", "range": (300, 800),    "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "Produced Water Hauling",                   "gl": "605-510", "range": (1500, 4500),  "vendor": "Eagle Ford Water Services",   "prob": 0.85},
            {"desc": "Compressor Skid - PM Service",             "gl": "605-720", "range": (1000, 2500),  "vendor": "South Texas Compression LLC", "prob": 0.35},
            {"desc": "Paraffin Hot Oil Treatment",               "gl": "605-315", "range": (1500, 3500),  "vendor": "South Texas Hot Oil",         "prob": 0.25},
        ],
        "Workover": {
            "desc": "Slickline - Gas Lift Valve & Mandrel Service",
            "gl":   "710-520",
            "vendor": "Frontier Well Services",
            "rig_rate":  (420, 580),
            "rig_hours": (8, 18),
            "parts":     (3500, 8000),
        }
    }
}

_invoice_counters: dict = {}


def _next_invoice_id(date_str: str) -> str:
    year = date_str[:4]
    _invoice_counters[year] = _invoice_counters.get(year, 0) + 1
    return f"STEP-{year}-{_invoice_counters[year]:05d}"


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
        "invoice_id": _next_invoice_id(date_str),
        "invoice_date":        date_str,
        "category":            "Workover",
        "gl_code":             lib["gl"],
        "service_description": lib["desc"],
        "vendor":              lib["vendor"],
        "total_usd":           round(cost, 2),
    }

if __name__ == "__main__":
    print("Generating Financial Ledger...")
    _invoice_counters.clear()

    with open("south_texas_engineering.json") as f:
        data = json.load(f)

    invoices = []
    for asset_id, asset in data["assets"].items():
        lift       = asset["lift"]
        lib        = EXPENSE_LIBRARY[lift]
        fail_start = None
        wi         = asset.get("working_interest", 1.0)

        for day in asset["history"]:
            date_str = day["date"]
            status   = day["run_status"]

            if date_str.endswith("-28"):
                for exp in lib["Monthly"]:
                    gross_cost = round(random.uniform(*exp["range"]), 2)
                    invoices.append({
                        "invoice_id": _next_invoice_id(date_str),
                        "invoice_date":        date_str,
                        "category":            "Monthly LOE",
                        "gl_code":             exp["gl"],
                        "service_description": exp["desc"],
                        "vendor":              exp["vendor"],
                        "gross_cost_usd":      gross_cost,
                        "working_interest":    wi,
                        "total_usd":           round(gross_cost * wi, 2),
                        "asset_id":            asset_id,
                    })
                for exp in lib["Ad_Hoc"]:
                    if random.random() < exp["prob"]:
                        gross_cost = round(random.uniform(*exp["range"]), 2)
                        invoices.append({
                            "invoice_id": _next_invoice_id(date_str),
                            "invoice_date":        date_str,
                            "category":            "Ad-Hoc LOE",
                            "gl_code":             exp["gl"],
                            "service_description": exp["desc"],
                            "vendor":              exp["vendor"],
                            "gross_cost_usd":      gross_cost,
                            "working_interest":    wi,
                            "total_usd":           round(gross_cost * wi, 2),
                            "asset_id":            asset_id,
                        })

            if status == "SHUT-IN" and fail_start is None:
                fail_start = date_str
            elif status == "PRODUCING" and fail_start is not None:
                inv            = build_workover_invoice(asset_id, date_str, lift)
                gross_cost     = inv["total_usd"]
                inv["gross_cost_usd"]   = gross_cost
                inv["working_interest"] = wi
                inv["total_usd"]        = round(gross_cost * wi, 2)
                inv["asset_id"]         = asset_id
                invoices.append(inv)
                fail_start = None

    print(f"  Generated {len(invoices):,} invoices across {len(data['assets'])} assets")

    invoice_map = {}
    for inv in invoices:
        aid = inv.pop("asset_id")
        if aid not in invoice_map:
            invoice_map[aid] = []
        invoice_map[aid].append(inv)

    for asset_id, asset in data["assets"].items():
        asset["financial_ledger"] = invoice_map.get(asset_id, [])

    with open("south_texas_engineering.json", "w") as f:
        json.dump(data, f, indent=2)

    total_h = sum(len(a["history"])          for a in data["assets"].values())
    total_l = sum(len(a["financial_ledger"]) for a in data["assets"].values())
    print(f"  Daily production records: {total_h:,}")
    print(f"  Cost ledger entries:      {total_l:,}")
    with open("south_texas_engineering.json") as f:
        lc = sum(1 for _ in f)
    print(f"  Total lines:              {lc:,}")
    print("✓ Done: south_texas_engineering.json")
