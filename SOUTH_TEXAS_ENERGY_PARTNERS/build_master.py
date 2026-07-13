"""
South Texas Energy Partners — Master Builder
===========================================
Reads south_texas_engineering.json (with invoices already attached)
and adds supporting assets, personnel, location, and clean wells.

Pipeline order:
  1. python3 generate_digital_twin.py   → south_texas_engineering.json
  2. python3 generate_invoices.py       → attaches financial_ledger to wells
  3. python3 build_master.py            → THIS SCRIPT → South_Texas_Energy_Partners_Master.json
"""

from __future__ import annotations

import importlib.util
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from config import LEGACY_ORGANIZATION_LABEL, STEP_ORGANIZATION_ID
except ImportError:
    STEP_ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"
    LEGACY_ORGANIZATION_LABEL = "SOUTH_TEXAS_ENERGY_PARTNERS"

_STEP_DIR = Path(__file__).resolve().parent
END_DATE = datetime(2026, 6, 8)
LEASES = ["Bravo", "Maverick", "Carrizo", "Cotulla", "Pearsall"]

LEASE_ENGINEER = {
    "Bravo": "Marcus Reyes",
    "Maverick": "Sandra Ochoa",
    "Carrizo": "Marcus Reyes",
    "Cotulla": "Sandra Ochoa",
    "Pearsall": "Marcus Reyes",
}

LEASE_LOCATION = {
    "Bravo": {
        "basin": "Gulf Coast Basin",
        "sub_basin": "Eagle Ford Shale",
        "state": "TX",
        "county": "La Salle County",
        "region": "South Texas",
    },
    "Maverick": {
        "basin": "Gulf Coast Basin",
        "sub_basin": "Eagle Ford Shale",
        "state": "TX",
        "county": "Maverick County",
        "region": "South Texas",
    },
    "Carrizo": {
        "basin": "Gulf Coast Basin",
        "sub_basin": "Eagle Ford Shale",
        "state": "TX",
        "county": "Karnes County",
        "region": "South Texas",
    },
    "Cotulla": {
        "basin": "Gulf Coast Basin",
        "sub_basin": "Eagle Ford Shale",
        "state": "TX",
        "county": "Dimmit County",
        "region": "South Texas",
    },
    "Pearsall": {
        "basin": "Gulf Coast Basin",
        "sub_basin": "Eagle Ford Shale",
        "state": "TX",
        "county": "Frio County",
        "region": "South Texas",
    },
}

AREA_MAP = {
    "Bravo": "Brush Country Area",
    "Maverick": "Rio Grande Area",
    "Carrizo": "Karnes Area",
    "Cotulla": "Rio Grande Area",
    "Pearsall": "Brush Country Area",
}

CLEAN_WELL_IDS = [
    "Bravo-1H-1-H",
    "Bravo-1H-2-H",
    "Bravo-1H-3-H",
    "Maverick-1H-21-H",
    "Maverick-1H-22-V",
    "Maverick-1H-23-H",
    "Carrizo-1H-41-H",
    "Carrizo-1H-42-H",
    "Carrizo-1H-43-H",
    "Cotulla-1H-61-H",
    "Cotulla-1H-62-H",
    "Cotulla-1H-63-H",
    "Pearsall-1H-81-H",
    "Pearsall-1H-82-H",
    "Pearsall-1H-83-H",
]


def get_personnel(lease):
    return {
        "prod_technician": {
            "name": f"{lease} Production Tech",
            "role": "Production Technician",
            "email": f"prodtech.{lease.lower()}@steppartners.com",
        },
        "well_perf_engineer": {
            "name": LEASE_ENGINEER[lease],
            "role": "Well Performance Engineer",
            "email": "wellperf@steppartners.com",
        },
    }


def daily_dates(start):
    dates = []
    d = start
    while d <= END_DATE:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def nr(lo, hi, noise_pct=0.04):
    mid = (lo + hi) / 2
    noise = (hi - lo) * noise_pct
    return round(float(np.clip(np.random.normal(mid, noise), lo, hi)), 2)


def _pressure_stepdown():
    wh = nr(80, 95)
    two = round(wh - nr(5, 15), 2)
    three = round(two - nr(5, 10), 2)
    gas = round(three - nr(5, 10), 2)
    return wh, two, three, gas


def build_facility(dates):
    h = []
    oil = random.uniform(500, 3000)
    water = random.uniform(500, 5000)
    gas = random.uniform(200, 2000)
    for ds in dates:
        oil = max(50, oil + np.random.normal(0, oil * 0.02))
        water = max(50, water + np.random.normal(0, water * 0.02))
        gas = max(10, gas + np.random.normal(0, gas * 0.02))
        wh, two, three, gas_line = _pressure_stepdown()
        h.append(
            {
                "date": ds,
                "wellhead_pressure_psi": wh,
                "two_phase_separator_pressure_psi": two,
                "three_phase_separator_pressure_psi": three,
                "gas_sales_line_pressure_psi": gas_line,
                "tester_1_pressure_psi": nr(20, 250),
                "tester_1_temp_f": nr(60, 220),
                "tester_1_oil_rate_bpd": round(nr(50, 2000), 2),
                "tester_1_water_rate_bpd": round(nr(50, 3000), 2),
                "tester_1_gas_rate_mcfd": round(nr(10, 500), 2),
                "tester_2_pressure_psi": nr(20, 250),
                "tester_2_temp_f": nr(60, 220),
                "tester_2_oil_rate_bpd": round(nr(50, 2000), 2),
                "tester_2_water_rate_bpd": round(nr(50, 3000), 2),
                "tester_2_gas_rate_mcfd": round(nr(10, 500), 2),
                "tester_3_pressure_psi": nr(20, 250),
                "tester_3_temp_f": nr(60, 220),
                "tester_3_oil_rate_bpd": round(nr(50, 2000), 2),
                "tester_3_water_rate_bpd": round(nr(50, 3000), 2),
                "tester_3_gas_rate_mcfd": round(nr(10, 500), 2),
                "prior_day_oil_production_bbl": round(oil, 2),
                "prior_day_water_production_bbl": round(water, 2),
                "prior_day_gas_production_mcf": round(gas, 2),
                "sales_gas_rate_mscfd": round(nr(100, 8000), 2),
                "sales_gas_temp_f": nr(40, 120),
                "sales_gas_pressure_psi": nr(20, 150),
                "custody_transfer_rate_bblhr": round(nr(50, 800), 2),
                "custody_transfer_temp_f": nr(60, 180),
                "crude_gravity_api": nr(28, 45),
                "basic_sediment_water_pct": round(nr(0.5, 8.0), 2),
                "produced_water_transfer_rate_bpd": round(nr(100, 5000), 2),
                "water_transfer_discharge_psi": nr(50, 280),
                "produced_water_temp_f": nr(50, 150),
                "facility_oil_tank_level_ft": nr(2.0, 22.0),
                "facility_water_tank_level_ft": nr(1.0, 18.0),
                "run_status": "ONLINE",
            }
        )
    return h


def build_compressor(dates):
    h = []
    for ds in dates:
        h.append(
            {
                "date": ds,
                "compressor_suction_pressure_psi": nr(50, 200),
                "compressor_suction_temp_f": nr(40, 120),
                "compressor_discharge_pressure_psi": nr(800, 1400),
                "compressor_discharge_temp_f": nr(100, 280),
                "gas_throughput_mscfd": round(nr(500, 15000), 2),
                "compressor_engine_rpm": round(nr(900, 1200), 0),
                "fuel_gas_rate_mcfd": round(nr(20, 200), 2),
                "compressor_engine_temp_f": nr(180, 350),
                "compressor_lube_oil_pressure_psi": nr(30, 80),
                "daily_runtime_hrs": round(nr(20, 24), 1),
                "run_status": "ONLINE",
            }
        )
    return h


def build_swd(dates):
    h = []
    vol = random.uniform(500, 8000)
    cum = 0.0
    for ds in dates:
        vol = max(50, vol + np.random.normal(0, vol * 0.03))
        cum += vol
        h.append(
            {
                "date": ds,
                "disposal_rate_bpd": round(vol, 2),
                "surface_injection_pressure_psi": nr(100, 1800),
                "swdwellhead_pressure_psi": nr(50, 400),
                "daily_disposal_volume_bbl": round(vol, 2),
                "cumulative_disposal_volume_bbl": round(cum, 2),
                "run_status": "ONLINE",
            }
        )
    return h


def build_tank(dates):
    h = []
    for ds in dates:
        ot1 = nr(2.0, 22.0)
        ot2 = nr(2.0, 22.0)
        ot3 = nr(2.0, 22.0)
        wt1 = nr(1.0, 18.0)
        gb1 = nr(1.0, 15.0)
        high = 1 if max(ot1, ot2, ot3, wt1) > 22.0 else 0
        h.append(
            {
                "date": ds,
                "ot_1_level_ft": ot1,
                "ot_1_temperature_f": nr(50, 120),
                "ot_1_run_ticket_bbl": round(nr(0, 500), 2),
                "ot_2_level_ft": ot2,
                "ot_2_temperature_f": nr(50, 120),
                "ot_2_run_ticket_bbl": round(nr(0, 500), 2),
                "ot_3_level_ft": ot3,
                "ot_3_temperature_f": nr(50, 120),
                "ot_3_run_ticket_bbl": round(nr(0, 500), 2),
                "wt_1_level_ft": wt1,
                "wt_1_temperature_f": nr(50, 100),
                "wt_1_disposal_ticket_bbl": round(nr(0, 1000), 2),
                "gb_1_level_ft": gb1,
                "gb_1_temperature_f": nr(50, 100),
                "high_level_alarm": high,
                "run_status": "ONLINE",
            }
        )
    return h


EXPENSE_MAP = {
    "Facility": {
        "Monthly": [
            {"desc": "Facility Electrical Power", "gl": "605-110", "range": (800, 2500), "vendor": "CPS Energy - South Texas"},
            {"desc": "Production Technician Labor Allocation", "gl": "605-210", "range": (800, 1200), "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Chemical Treatment - Facility", "gl": "605-310", "range": (300, 800), "vendor": "Monarch Chemical Services"},
            {"desc": "SCADA / RTU Telemetry - Facility", "gl": "605-410", "range": (150, 300), "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation", "gl": "822-110", "range": (300, 800), "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "LACT Unit Calibration & Service", "gl": "605-710", "range": (500, 1500), "vendor": "South Texas Electrical LLC", "prob": 0.25},
            {"desc": "Separator / Tester Maintenance", "gl": "605-720", "range": (400, 1200), "vendor": "Laredo Roustabout Services", "prob": 0.20},
        ],
    },
    "Compressor": {
        "Monthly": [
            {"desc": "Compression Fuel Gas", "gl": "605-120", "range": (3000, 12000), "vendor": "South Texas Compression LLC"},
            {"desc": "Production Technician Labor Allocation", "gl": "605-210", "range": (800, 1200), "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Compressor Lube Oil", "gl": "605-310", "range": (400, 900), "vendor": "Monarch Chemical Services"},
            {"desc": "SCADA / RTU Telemetry - Compression", "gl": "605-410", "range": (150, 300), "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation", "gl": "822-110", "range": (300, 800), "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "Compressor Skid - PM Service", "gl": "605-720", "range": (900, 2500), "vendor": "South Texas Compression LLC", "prob": 0.35},
            {"desc": "Engine Filter & Belt Replacement", "gl": "605-720", "range": (300, 800), "vendor": "South Texas Compression LLC", "prob": 0.25},
        ],
    },
    "SWD Well": {
        "Monthly": [
            {"desc": "SWD Injection Pump - Electrical Power", "gl": "605-130", "range": (1500, 5000), "vendor": "CPS Energy - South Texas"},
            {"desc": "Production Technician Labor Allocation", "gl": "605-210", "range": (800, 1200), "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Chemical Treatment - Scale/Corrosion", "gl": "605-310", "range": (400, 900), "vendor": "Monarch Chemical Services"},
            {"desc": "SCADA / RTU Telemetry - SWD", "gl": "605-410", "range": (150, 250), "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation", "gl": "822-110", "range": (300, 800), "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "SWD Disposal Fee - Volume Overage", "gl": "605-520", "range": (1000, 4000), "vendor": "Eagle Ford Water Services", "prob": 0.30},
            {"desc": "Injection Pump Maintenance", "gl": "605-740", "range": (500, 1500), "vendor": "South Texas Electrical LLC", "prob": 0.25},
        ],
    },
    "Tank Battery": {
        "Monthly": [
            {"desc": "Produced Water Hauling - Tank Battery", "gl": "605-510", "range": (2000, 8000), "vendor": "Eagle Ford Water Services"},
            {"desc": "Production Technician Labor Allocation", "gl": "605-210", "range": (800, 1200), "vendor": "South Texas Energy Partners Internal"},
            {"desc": "Tank Battery Chemical Treatment", "gl": "605-310", "range": (300, 700), "vendor": "Monarch Chemical Services"},
            {"desc": "SCADA / RTU Telemetry - Tank Battery", "gl": "605-410", "range": (150, 250), "vendor": "Lone Star Telemetry"},
            {"desc": "G&A Overhead Allocation", "gl": "822-110", "range": (300, 800), "vendor": "South Texas Energy Partners Internal"},
        ],
        "Ad_Hoc": [
            {"desc": "Oil Haul-Off - Tank Battery", "gl": "605-510", "range": (500, 2000), "vendor": "Eagle Ford Water Services", "prob": 0.70},
            {"desc": "Tank Cleaning & Inspection", "gl": "605-710", "range": (800, 2500), "vendor": "Laredo Roustabout Services", "prob": 0.15},
        ],
    },
}


def gen_invoices(history, lib):
    invoices = []
    for day in history:
        ds = day["date"]
        if ds.endswith("-28"):
            for exp in lib["Monthly"]:
                invoices.append(
                    {
                        "invoice_id": f"STEP-{ds[:4]}-{random.randint(10000, 99999)}",
                        "invoice_date": ds,
                        "category": "Monthly LOE",
                        "gl_code": exp["gl"],
                        "service_description": exp["desc"],
                        "vendor": exp["vendor"],
                        "total_usd": round(random.uniform(*exp["range"]), 2),
                    }
                )
            for exp in lib["Ad_Hoc"]:
                if random.random() < exp["prob"]:
                    invoices.append(
                        {
                            "invoice_id": f"STEP-{ds[:4]}-{random.randint(10000, 99999)}",
                            "invoice_date": ds,
                            "category": "Ad-Hoc LOE",
                            "gl_code": exp["gl"],
                            "service_description": exp["desc"],
                            "vendor": exp["vendor"],
                            "total_usd": round(random.uniform(*exp["range"]), 2),
                        }
                    )
    return invoices


def _load_twin_module():
    path = _STEP_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("step_twin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROD_PARAMS = {
    "Pumping Unit": {
        "water_base": (50.0, 800.0),
        "gas_base": (10.0, 150.0),
        "water_noise": 0.06,
        "gas_noise": 0.05,
        "water_decline": 0.0002,
        "gas_decline": 0.0008,
        "water_rise": 0.0001,
    },
    "ESP": {
        "water_base": (200.0, 5000.0),
        "gas_base": (50.0, 500.0),
        "water_noise": 0.05,
        "gas_noise": 0.04,
        "water_decline": 0.0001,
        "gas_decline": 0.0007,
        "water_rise": 0.00015,
    },
    "Gas Lift": {
        "water_base": (100.0, 2000.0),
        "gas_base": (100.0, 2000.0),
        "water_noise": 0.07,
        "gas_noise": 0.06,
        "water_decline": 0.00015,
        "gas_decline": 0.0006,
        "water_rise": 0.00012,
    },
}


def rebuild_clean_well(original_history, lift):
    """Rebuild a well's history as completely healthy — no failures."""
    twin = _load_twin_module()
    normals = {
        k: v
        for k, v in twin.SENSOR_NORMALS[lift].items()
        if k not in ("gross_oil_production_bbl", "well_test_rate_bopd")
    }
    sensor_keys = list(normals.keys())
    pp = PROD_PARAMS[lift]
    days = len(original_history)
    t = np.arange(days)
    water_base = random.uniform(*pp["water_base"])
    gas_base = random.uniform(*pp["gas_base"])
    water_rates = np.clip(
        water_base * (1 + pp["water_rise"] * t) * np.exp(-pp["water_decline"] * t)
        * np.random.normal(1.0, pp["water_noise"], days),
        1.0,
        None,
    )
    gas_rates = np.clip(
        gas_base * np.exp(-pp["gas_decline"] * t) * np.random.normal(1.0, pp["gas_noise"], days),
        0.5,
        None,
    )

    new_history = []
    for i, day in enumerate(original_history):
        date_str = day["date"]
        oil = max(day.get("gross_oil_production_bbl", 0) or 0, 5.0)
        new_day = {
            "date": date_str,
            "gross_oil_production_bbl": oil,
            "well_test_rate_bopd": None,
            "run_status": "PRODUCING",
            "produced_water_volume_bbl": round(float(water_rates[i]), 2),
            "casinghead_gas_mcf": round(float(gas_rates[i]), 2),
        }
        if lift == "Pumping Unit":
            gb_mid = (20.0 + 600.0) / 2
            nb_mid = (20.0 + 600.0) / 2
            gross = max(1.0, float(np.random.normal(gb_mid, gb_mid * 0.01)))
            net = max(1.0, float(min(np.random.normal(nb_mid, nb_mid * 0.012), gross * 0.98)))
            new_day["polished_rod_stroke_length_in"] = round(gross, 2)
            new_day["effective_plunger_stroke_in"] = round(net, 2)
        for s in sensor_keys:
            if s in ("polished_rod_stroke_length_in", "effective_plunger_stroke_in"):
                continue
            lo, hi = normals[s]
            mid = (lo + hi) / 2
            noise = (hi - lo) * 0.04
            new_day[s] = round(float(np.clip(np.random.normal(mid, noise), lo, hi)), 2)
        new_history.append(new_day)
    return new_history


if __name__ == "__main__":
    print("Loading south_texas_engineering.json...")
    with open(_STEP_DIR / "south_texas_engineering.json") as f:
        data = json.load(f)

    assets = data["assets"]
    org_id = None
    for asset in assets.values():
        org_id = asset.get("organization_id")
        if org_id:
            break

    print("Assigning personnel, location, and area to all wells...")
    for aid, asset in assets.items():
        if asset.get("type") != "Well":
            continue
        lease = aid.split("-")[0]
        asset["assigned_personnel"] = get_personnel(lease)
        asset["location"] = LEASE_LOCATION[lease]
        asset["area"] = AREA_MAP.get(lease, "")

    lease_starts = {}
    for lease in LEASES:
        earliest = min(
            datetime.strptime(a["history"][0]["date"], "%Y-%m-%d")
            for k, a in assets.items()
            if k.startswith(lease) and a.get("type") == "Well"
        )
        lease_starts[lease] = earliest
        print(f"  {lease}: earliest well = {earliest.date()}")

    print("\nRebuilding clean wells...")
    for aid in CLEAN_WELL_IDS:
        if aid not in assets:
            continue
        lift = assets[aid]["lift"]
        assets[aid]["history"] = rebuild_clean_well(assets[aid]["history"], lift)
        assets[aid]["financial_ledger"] = [
            i for i in assets[aid].get("financial_ledger", []) if i.get("category") != "Workover"
        ]
        print(f"  ✓ {aid} ({lift})")

    print("\nAdding supporting assets...")
    builders = [
        ("Facility", "FAC-01", build_facility),
        ("Compressor", "COMP-01", build_compressor),
        ("SWD Well", "SWD-01", build_swd),
        ("Tank Battery", "TANK-01", build_tank),
    ]
    for lease in LEASES:
        dates = daily_dates(lease_starts[lease])
        ownership = next(
            (
                a
                for k, a in assets.items()
                if k.startswith(lease) and a.get("type") == "Well"
            ),
            {},
        )
        for atype, suffix, builder in builders:
            aid = f"{lease}-{suffix}"
            history = builder(dates)
            invoices = gen_invoices(history, EXPENSE_MAP[atype])
            assets[aid] = {
                "type": atype,
                "lift": None,
                "organization_id": STEP_ORGANIZATION_ID,
                "area": AREA_MAP.get(lease, ""),
                "working_interest": ownership.get("working_interest", 1.0),
                "lease_type": ownership.get("lease_type"),
                "net_revenue_interest": 0.75 if atype == "SWD Well" else None,
                "location": LEASE_LOCATION[lease],
                "assigned_personnel": get_personnel(lease),
                "history": history,
                "financial_ledger": invoices,
            }
            print(f"  {aid:<30} {len(history):,} days  {len(invoices):,} invoices")

    out_path = _STEP_DIR / "South_Texas_Energy_Partners_Master.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    total_h = sum(len(a["history"]) for a in assets.values())
    total_l = sum(len(a.get("financial_ledger", [])) for a in assets.values())
    print(f"\n✓ Done — {len(assets)} assets, {total_h:,} daily rows, {total_l:,} invoices")
    print(f"  Output: {out_path.name}")
