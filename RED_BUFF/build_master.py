"""
Red Bluff Resources — Master Builder
========================================
Reads red_bluff_engineering.json (with invoices already attached)
and adds:
  - Supporting assets (Facility, Compressor, SWD Well, Tank Battery) per lease
  - Personnel (Well Performance Engineer + Production Technician) per lease
  - Location metadata (basin, sub_basin, state, county, region) per lease
  - 12 clean wells (3 per lease, mixed lift types) with no failure events

Pipeline order:
  1. python3 generate_digital_twin.py   → red_bluff_engineering.json
  2. python3 generate_invoices.py       → attaches financial_ledger to wells
  3. python3 build_master.py            → THIS SCRIPT → Red_Bluff_Resources_Master.json

Output: Red_Bluff_Resources_Master.json
"""

import json
import numpy as np
import random
from datetime import datetime, timedelta

END_DATE = datetime(2026, 5, 7)
LEASES   = ["Chaparral", "Mesquite", "Caliche", "Llano"]

# ─── LEASE CONFIG ─────────────────────────────────────────────────────────────
LEASE_ENGINEER = {
    "Chaparral": "Derek Tatum",
    "Mesquite":  "Kristen Albright",
    "Caliche":   "Derek Tatum",
    "Llano":     "Kristen Albright",
}

LEASE_LOCATION = {
    "Chaparral": {"basin":"Permian Basin","sub_basin":"Delaware Basin","state":"TX","county":"Reeves County","region":"West Texas"},
    "Mesquite":  {"basin":"Permian Basin","sub_basin":"Delaware Basin","state":"TX","county":"Loving County","region":"West Texas"},
    "Caliche":   {"basin":"Permian Basin","sub_basin":"Midland Basin","state":"TX","county":"Midland County","region":"West Texas"},
    "Llano":     {"basin":"Permian Basin","sub_basin":"Midland Basin","state":"TX","county":"Ector County","region":"West Texas"},
}

def get_personnel(lease):
    return {
        "prod_technician": {
            "name":  f"{lease} Production Tech",
            "role":  "Production Technician",
            "email": f"prodtech.{lease.lower()}@redbluffresources.com"
        },
        "well_perf_engineer": {
            "name":  LEASE_ENGINEER[lease],
            "role":  "Well Performance Engineer",
            "email": "wellperf@redbluffresources.com"
        }
    }

def daily_dates(start):
    dates = []; d = start
    while d <= END_DATE:
        dates.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    return dates

def nr(lo, hi, noise_pct=0.04):
    mid = (lo + hi) / 2; noise = (hi - lo) * noise_pct
    return round(float(np.clip(np.random.normal(mid, noise), lo, hi)), 2)

# ─── SUPPORTING ASSET GENERATORS ─────────────────────────────────────────────
def build_facility(dates):
    h = []; oil = random.uniform(500,3000); water = random.uniform(500,5000); gas = random.uniform(200,2000)
    for ds in dates:
        oil   = max(50, oil   + np.random.normal(0, oil*0.02))
        water = max(50, water + np.random.normal(0, water*0.02))
        gas   = max(10, gas   + np.random.normal(0, gas*0.02))
        h.append({"date":ds,"tester_pressure_psi":nr(20,250),"tester_temp_f":nr(60,220),
                  "tester_oil_rate_bpd":round(nr(50,2000),2),"tester_water_rate_bpd":round(nr(50,3000),2),
                  "tester_gas_rate_mcfd":round(nr(10,500),2),"yesterday_oil_bbl":round(oil,2),
                  "yesterday_water_bbl":round(water,2),"yesterday_gas_mcf":round(gas,2),
                  "gas_sales_rate_mscfd":round(nr(100,8000),2),"gas_sales_temp_f":nr(40,120),
                  "gas_sales_pressure_psi":nr(20,150),"lact_flow_rate_bblhr":round(nr(50,800),2),
                  "lact_temp_f":nr(60,180),"lact_density_api":nr(28,42),"lact_bsw_pct":round(nr(0.5,8.0),2),
                  "water_transfer_rate_bpd":round(nr(100,5000),2),"water_discharge_psi":nr(50,280),
                  "water_temp_f":nr(50,150),"oil_tank_level_ft":nr(2.0,22.0),
                  "water_tank_level_ft":nr(1.0,18.0),"run_status":"ONLINE"})
    return h

def build_compressor(dates):
    h = []
    for ds in dates:
        h.append({"date":ds,"suction_pressure_psi":nr(50,200),"suction_temp_f":nr(40,120),
                  "discharge_pressure_psi":nr(800,1400),"discharge_temp_f":nr(100,280),
                  "throughput_mscfd":round(nr(500,15000),2),"engine_rpm":round(nr(900,1200),0),
                  "fuel_gas_consumption_mcfd":round(nr(20,200),2),"engine_temp_f":nr(180,350),
                  "lube_oil_pressure_psi":nr(30,80),"runtime_hrs":round(nr(20,24),1),"run_status":"ONLINE"})
    return h

def build_swd(dates):
    h = []; vol = random.uniform(500,8000); cum = 0
    for ds in dates:
        vol = max(50, vol + np.random.normal(0, vol*0.03)); cum += vol
        h.append({"date":ds,"injection_rate_bpd":round(vol,2),"injection_pressure_psi":nr(100,1800),
                  "wellhead_pressure_psi":nr(50,400),"tubing_pressure_psi":nr(80,600),
                  "casing_pressure_psi":nr(20,200),"injection_temp_f":nr(60,160),
                  "daily_volume_injected_bbl":round(vol,2),"cumulative_volume_bbl":round(cum,2),
                  "pump_discharge_pressure_psi":nr(150,2000),"pump_speed_hz":nr(40,60),"run_status":"ONLINE"})
    return h

def build_tank(dates):
    h = []; oil = random.uniform(5.0,18.0); water = random.uniform(3.0,15.0)
    for ds in dates:
        oil   = float(np.clip(oil   + np.random.normal(0,0.8), 1.0, 28.0))
        water = float(np.clip(water + np.random.normal(0,1.0), 1.0, 28.0))
        h.append({"date":ds,"oil_tank_level_ft":round(oil,2),"water_tank_level_ft":round(water,2),
                  "oil_tank_temp_f":nr(50,120),"water_tank_temp_f":nr(50,100),
                  "oil_hauled_today_bbl":round(nr(0,500),2),"water_hauled_today_bbl":round(nr(0,1000),2),
                  "high_tank_alarm":1 if oil>26.0 or water>26.0 else 0,"run_status":"ONLINE"})
    return h

# ─── SUPPORTING ASSET INVOICE LIBRARIES ──────────────────────────────────────
EXPENSE_MAP = {
    "Facility": {
        "Monthly":[
            {"desc":"Facility Electrical Power","gl":"605-110","range":(800,2500),"vendor":"Trans-Pecos Electric"},
            {"desc":"Production Technician Labor Allocation","gl":"605-210","range":(800,1200),"vendor":"Red Bluff Internal"},
            {"desc":"Chemical Treatment - Facility","gl":"605-310","range":(300,800),"vendor":"Permian Chemical Supply"},
            {"desc":"SCADA / RTU Telemetry - Facility","gl":"605-410","range":(150,300),"vendor":"Basin Telemetry & Controls"},
            {"desc":"G&A Overhead Allocation","gl":"822-110","range":(300,800),"vendor":"Red Bluff Internal"},
        ],
        "Ad_Hoc":[
            {"desc":"LACT Unit Calibration & Service","gl":"605-710","range":(500,1500),"vendor":"West Texas Electrical Services","prob":0.25},
            {"desc":"Separator / Tester Maintenance","gl":"605-720","range":(400,1200),"vendor":"Pecos Valley Roustabout","prob":0.20},
            {"desc":"Roustabout - Facility Maintenance","gl":"605-710","range":(300,700),"vendor":"Pecos Valley Roustabout","prob":0.30},
        ]
    },
    "Compressor": {
        "Monthly":[
            {"desc":"Compression Fuel Gas","gl":"605-120","range":(3000,12000),"vendor":"Red Bluff Internal"},
            {"desc":"Production Technician Labor Allocation","gl":"605-210","range":(800,1200),"vendor":"Red Bluff Internal"},
            {"desc":"Compressor Lube Oil","gl":"605-310","range":(400,900),"vendor":"Permian Chemical Supply"},
            {"desc":"SCADA / RTU Telemetry - Compression","gl":"605-410","range":(150,300),"vendor":"Basin Telemetry & Controls"},
            {"desc":"G&A Overhead Allocation","gl":"822-110","range":(300,800),"vendor":"Red Bluff Internal"},
        ],
        "Ad_Hoc":[
            {"desc":"Compressor Skid - PM Service","gl":"605-720","range":(900,2500),"vendor":"Delaware Basin Compression","prob":0.35},
            {"desc":"Engine Filter & Belt Replacement","gl":"605-720","range":(300,800),"vendor":"Delaware Basin Compression","prob":0.25},
            {"desc":"Compressor Valve Overhaul","gl":"605-720","range":(2000,6000),"vendor":"Delaware Basin Compression","prob":0.10},
            {"desc":"Freeze Protection - Methanol Injection","gl":"605-320","range":(800,2000),"vendor":"Permian Chemical Supply","prob":0.15},
        ]
    },
    "SWD Well": {
        "Monthly":[
            {"desc":"SWD Injection Pump - Electrical Power","gl":"605-130","range":(1500,5000),"vendor":"Trans-Pecos Electric"},
            {"desc":"Production Technician Labor Allocation","gl":"605-210","range":(800,1200),"vendor":"Red Bluff Internal"},
            {"desc":"Chemical Treatment - Scale/Corrosion","gl":"605-310","range":(400,900),"vendor":"Permian Chemical Supply"},
            {"desc":"SCADA / RTU Telemetry - SWD","gl":"605-410","range":(150,250),"vendor":"Basin Telemetry & Controls"},
            {"desc":"G&A Overhead Allocation","gl":"822-110","range":(300,800),"vendor":"Red Bluff Internal"},
        ],
        "Ad_Hoc":[
            {"desc":"SWD Disposal Fee - Volume Overage","gl":"605-520","range":(1000,4000),"vendor":"Reeves County Water Hauling","prob":0.30},
            {"desc":"Injection Pump Maintenance","gl":"605-740","range":(500,1500),"vendor":"West Texas Electrical Services","prob":0.25},
            {"desc":"Wellbore Pressure Test - Regulatory","gl":"605-710","range":(800,2000),"vendor":"Red Bluff Rig Services","prob":0.10},
            {"desc":"Acid Treatment - SWD Wellbore","gl":"605-620","range":(2000,5000),"vendor":"Permian Chemical Supply","prob":0.08},
        ]
    },
    "Tank Battery": {
        "Monthly":[
            {"desc":"Produced Water Hauling - Tank Battery","gl":"605-510","range":(2000,8000),"vendor":"Reeves County Water Hauling"},
            {"desc":"Production Technician Labor Allocation","gl":"605-210","range":(800,1200),"vendor":"Red Bluff Internal"},
            {"desc":"Tank Battery Chemical Treatment","gl":"605-310","range":(300,700),"vendor":"Permian Chemical Supply"},
            {"desc":"SCADA / RTU Telemetry - Tank Battery","gl":"605-410","range":(150,250),"vendor":"Basin Telemetry & Controls"},
            {"desc":"G&A Overhead Allocation","gl":"822-110","range":(300,800),"vendor":"Red Bluff Internal"},
        ],
        "Ad_Hoc":[
            {"desc":"Oil Haul-Off - Tank Battery","gl":"605-510","range":(500,2000),"vendor":"Reeves County Water Hauling","prob":0.70},
            {"desc":"Tank Cleaning & Inspection","gl":"605-710","range":(800,2500),"vendor":"Pecos Valley Roustabout","prob":0.15},
            {"desc":"Tank Vent / Thief Hatch Repair","gl":"605-710","range":(200,600),"vendor":"Pecos Valley Roustabout","prob":0.20},
            {"desc":"Hot Oil Treatment - Tank Battery","gl":"605-610","range":(600,1500),"vendor":"Midland Hot Oil Service","prob":0.20},
        ]
    },
}

def gen_invoices(history, lib):
    invoices = []
    for day in history:
        ds = day['date']
        if ds.endswith("-28"):
            for exp in lib["Monthly"]:
                invoices.append({"invoice_id":f"RBNR-{ds[:4]}-{random.randint(10000,99999)}",
                                  "invoice_date":ds,"category":"Monthly LOE","gl_code":exp["gl"],
                                  "service_description":exp["desc"],"vendor":exp["vendor"],
                                  "total_usd":round(random.uniform(*exp["range"]),2)})
            for exp in lib["Ad_Hoc"]:
                if random.random() < exp["prob"]:
                    invoices.append({"invoice_id":f"RBNR-{ds[:4]}-{random.randint(10000,99999)}",
                                      "invoice_date":ds,"category":"Ad-Hoc LOE","gl_code":exp["gl"],
                                      "service_description":exp["desc"],"vendor":exp["vendor"],
                                      "total_usd":round(random.uniform(*exp["range"]),2)})
    return invoices

# ─── CLEAN WELL REBUILD ───────────────────────────────────────────────────────
SENSOR_NORMALS_CLEAN = {
    "Pumping Unit": {"gross_stroke_len":(120.0,168.0),"net_stroke_len":(100.0,148.0),
                     "strokes_per_min":(0.5,12.0),"pump_fillage_pct":(0.0,100.0),
                     "casing_pressure_psi":(50.0,5000.0),"tubing_pressure_psi":(50.0,150.0),
                     "motor_current_amps":(10.0,200.0),"freq_hz":(57.0,62.0),
                     "pump_intake_pressure":(0.0,5000.0),"motor_temp_f":(100.0,160.0)},
    "ESP":          {"avg_tubing_pressure":(200.0,500.0),"avg_casing_pressure":(20.0,200.0),
                     "motor_temp_f":(150.0,250.0),"motor_current_amps":(10.0,200.0),
                     "freq_hz":(0.0,100.0),"pump_intake_pressure_psi":(0.0,5000.0),
                     "pump_discharge_pressure_psi":(0.0,5000.0),"vibration_hz":(0.0,1.25)},
    "Gas Lift":     {"avg_casing_pressure":(50.0,5000.0),"casing_injection_pressure_psi":(800.0,1200.0),
                     "tubing_pressure_psi":(50.0,150.0),"injection_rate_mscfd":(0.0,50000.0),
                     "wellhead_temp_f":(10.0,400.0),"choke_position_pct":(0.0,100.0),
                     "motor_temp_f":(80.0,130.0)},
}

PROD_PARAMS = {
    "Pumping Unit": {"water_base":(50.0,800.0),"gas_base":(10.0,150.0),"water_noise":0.06,"gas_noise":0.05,"water_decline":0.0002,"gas_decline":0.0008,"water_rise":0.0001},
    "ESP":          {"water_base":(200.0,5000.0),"gas_base":(50.0,500.0),"water_noise":0.05,"gas_noise":0.04,"water_decline":0.0001,"gas_decline":0.0007,"water_rise":0.00015},
    "Gas Lift":     {"water_base":(100.0,2000.0),"gas_base":(100.0,2000.0),"water_noise":0.07,"gas_noise":0.06,"water_decline":0.00015,"gas_decline":0.0006,"water_rise":0.00012},
}

def rebuild_clean_well(original_history, lift):
    """Rebuild a well's history as completely healthy — no failures, no deterioration."""
    normals     = SENSOR_NORMALS_CLEAN[lift]
    sensor_keys = list(normals.keys())
    test_day    = next((datetime.strptime(d['date'],'%Y-%m-%d').day
                        for d in original_history if d.get('test_oil') is not None),
                       random.randint(5,25))

    pp          = PROD_PARAMS[lift]
    days        = len(original_history)
    t           = np.arange(days)
    water_base  = random.uniform(*pp["water_base"])
    gas_base    = random.uniform(*pp["gas_base"])
    water_rates = np.clip(water_base*(1+pp["water_rise"]*t)*np.exp(-pp["water_decline"]*t)*
                          np.random.normal(1.0,pp["water_noise"],days), 1.0, None)
    gas_rates   = np.clip(gas_base*np.exp(-pp["gas_decline"]*t)*
                          np.random.normal(1.0,pp["gas_noise"],days), 0.5, None)

    new_history = []
    for i, day in enumerate(original_history):
        date_str  = day['date']
        allocated = max(day['allocated_oil'], 5.0)  # ensure no zero oil
        is_test   = datetime.strptime(date_str,'%Y-%m-%d').day == test_day
        new_day   = {
            "date":            date_str,
            "allocated_oil":   allocated,
            "test_oil":        round(allocated*(1+random.choice([1,-1])*random.uniform(0.05,0.12)),2) if is_test else None,
        }
        if lift == "Pumping Unit":
            gb_mid = (120.0+168.0)/2; nb_mid = (100.0+148.0)/2
            gross  = max(1.0, float(np.random.normal(gb_mid, gb_mid*0.01)))
            net    = max(1.0, float(min(np.random.normal(nb_mid, nb_mid*0.012), gross*0.98)))
            new_day["gross_stroke_len"] = round(gross,2)
            new_day["net_stroke_len"]   = round(net,2)
        for s in sensor_keys:
            if s in ('gross_stroke_len','net_stroke_len'): continue
            new_day[s] = nr(*normals[s])
        new_day["run_status"]    = "PRODUCING"
        new_day["failure_cause"] = ""
        new_day["allocated_water"] = round(float(water_rates[i]),2)
        new_day["allocated_gas"]   = round(float(gas_rates[i]),2)
        new_history.append(new_day)
    return new_history

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading engineering + invoice data...")
    with open("red_bluff_engineering.json") as f:
        data = json.load(f)

    assets = data["assets"]

    # ── Assign personnel and location to all wells ────────────────────────────
    print("Assigning personnel and location to all wells...")
    for aid, asset in assets.items():
        lease = aid.split('-')[0]
        asset["assigned_personnel"] = get_personnel(lease)
        asset["location"]           = LEASE_LOCATION[lease]

    # ── Find earliest well per lease for supporting asset start dates ─────────
    lease_starts = {}
    for lease in LEASES:
        earliest = min(datetime.strptime(a["history"][0]["date"],'%Y-%m-%d')
                       for k,a in assets.items() if k.startswith(lease) and a["type"]=="Well")
        lease_starts[lease] = earliest
        print(f"  {lease}: earliest well = {earliest.date()}")

    # ── Select 12 clean wells — 3 per lease, lowest failure counts ───────────
    print("\nSelecting 12 clean wells (3 per lease, mixed lift types)...")
    for lease in LEASES:
        lease_wells = [(k,v) for k,v in assets.items()
                       if k.startswith(lease) and v["type"]=="Well"]
        ranked = sorted(lease_wells,
                        key=lambda x: sum(1 for d in x[1]["history"] if d["run_status"]=="SHUT-IN"))
        chosen = {}
        for aid, asset in ranked:
            lift = asset["lift"]
            if lift not in chosen:
                chosen[lift] = aid
            if len(chosen) == 3:
                break
        for aid in chosen.values():
            assets[aid]["history"] = rebuild_clean_well(assets[aid]["history"], assets[aid]["lift"])
            assets[aid]["financial_ledger"] = [i for i in assets[aid]["financial_ledger"]
                                                if i["category"] != "Workover"]
            print(f"  ✓ {aid} ({assets[aid]['lift']}) — rebuilt as clean")

    # ── Add 16 supporting assets (4 per lease) ────────────────────────────────
    print("\nAdding supporting assets...")
    BUILDERS = [("Facility","FAC-01",build_facility),
                ("Compressor","COMP-01",build_compressor),
                ("SWD Well","SWD-01",build_swd),
                ("Tank Battery","TANK-01",build_tank)]

    for lease in LEASES:
        dates = daily_dates(lease_starts[lease])
        for atype, suffix, builder in BUILDERS:
            aid      = f"{lease}-{suffix}"
            history  = builder(dates)
            invoices = gen_invoices(history, EXPENSE_MAP[atype])
            assets[aid] = {
                "type":               atype,
                "lift":               None,
                "location":           LEASE_LOCATION[lease],
                "assigned_personnel": get_personnel(lease),
                "history":            history,
                "financial_ledger":   invoices,
            }
            print(f"  {aid:<30} {len(history):,} days  {len(invoices):,} invoices")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_h = sum(len(a["history"]) for a in assets.values())
    total_l = sum(len(a["financial_ledger"]) for a in assets.values())
    print(f"\n✓ Done")
    print(f"  Total assets:    {len(assets)}")
    print(f"  Total daily rows:{total_h:,}")
    print(f"  Total invoices:  {total_l:,}")

    with open("Red_Bluff_Resources_Master.json","w") as f:
        json.dump(data, f, indent=2)
    print(f"  Output:          Red_Bluff_Resources_Master.json")
