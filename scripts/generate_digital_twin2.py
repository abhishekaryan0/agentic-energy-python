"""
APEX PERMIAN SYNTHETIC DATASET GENERATOR

This script generates the engineering base dataset:
    apex_permian_v3_1_engineering.json

Notes:
- This script produces the engineering / operational digital twin only.
- Personnel-enriched and invoice-enriched files are maintained separately.
- The packaged master file (Master.json) is the unified deliverable that
  combines the engineering structure with personnel and invoice objects.
- A standalone invoice extract is also maintained separately as Invoices.json.
"""

import json
import warnings
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
START_DATE = datetime(2014, 1, 1)
END_DATE = datetime(2024, 6, 1)
TOTAL_DAYS = (END_DATE - START_DATE).days

# --- 1. FAILURE LIBRARY (KPI IMPACT) ---
FAILURE_MODES = {
    "Rod Pump": {
        "Early": ["Polished Rod Clamp Slip", "Misaligned Unit"],
        "Mid": ["Parted Rods", "Stuck Pump", "Gearbox Failure"],
        "Late": ["Hole in Tubing", "Casing Leak", "Unit Structural Fatigue"]
    },
    "ESP": {
        "Early": ["Splicing Failure", "Improper Cooling"],
        "Mid": ["Broken Shaft", "Cable Short", "Vibration Trip"],
        "Late": ["Pump Stage Wear", "Scale Deposition"]
    },
    "Gas Lift": {
        "Early": ["Valve Installation Error"],
        "Mid": ["Compressor Trip", "Injection Line Freeze"],
        "Late": ["Scale Restriction", "Liquid Loading"]
    }
}

# --- 2. ASSET BUILDER ---
def build_hierarchy():
    well_registry = []
    facility_registry = {} 

    lifts = ["Rod Pump", "ESP", "Gas Lift"]
    orients = ["Horizontal", "Vertical"]

    definitions = []
    for lift in lifts:
        for orient in orients:
            for _ in range(5):
                definitions.append({"lift": lift, "orient": orient})
    random.shuffle(definitions)

    status_pool = ["Happy Path"] * 12 + ["Failure"] * 9 + ["Degrading"] * 9
    random.shuffle(status_pool)

    leases = ["Mustang", "Bronco", "Ranger", "Pinto"]

    for i in range(30):
        defn = definitions[i]
        lease = leases[i % len(leases)]
        sect = [12, 22, 14, 4][i % 4]

        fac_id = f"{lease}-{defn['orient']}-Battery"

        age_days = random.randint(365, TOTAL_DAYS - 30)
        spud_date = END_DATE - timedelta(days=age_days)

        if fac_id not in facility_registry:
            facility_registry[fac_id] = spud_date
        elif spud_date < facility_registry[fac_id]:
            facility_registry[fac_id] = spud_date

        exist = len([w for w in well_registry if w["lease"] == lease]) + 1
        w_name = f"{lease}-{sect}-{exist}-{defn['orient'][0]}"

        well_registry.append({
            "asset_id": w_name,
            "lease": lease,
            "facility_id": fac_id,
            "lift_type": defn["lift"],
            "orientation": defn["orient"],
            "spud_date": spud_date,
            "current_status_goal": status_pool[i],
            "base_rate": 800 if defn["orient"] == "Horizontal" else 150
        })

    return well_registry, facility_registry

# --- 3. PHYSICS ENGINE (The Engineering KPIs) ---
def generate_well_timeline(well):
    dates = pd.date_range(well["spud_date"], END_DATE, freq="D")
    days = len(dates)
    if days < 1:
        return pd.DataFrame()

    # A. Base Production (Decline)
    t = np.arange(days)
    decline = 0.0006 if well["orientation"] == "Horizontal" else 0.0002
    rates = well["base_rate"] * np.exp(-decline * t)
    rates = rates * np.random.normal(1.0, 0.05, days)  # Noise

    # B. Lift-Specific Parameters (KPIs)
    lift = well["lift_type"]

    # Common Defaults
    gross_stroke = [None] * days
    net_stroke = [None] * days
    spm = [None] * days
    fillage = [None] * days
    freq_hz = [None] * days
    motor_amps = [None] * days
    pip = [None] * days
    motor_temp = [None] * days
    inj_rate = [None] * days
    inj_press = [None] * days

    # Common Sensors
    tp = (rates * 0.1) + 200  # Base Tubing Pressure
    cp = (rates * 0.05) + 50  # Base Casing Pressure

    # --- ENGINEERING LOGIC ---
    if lift == "Rod Pump":
        gross_base = 144 if well["orientation"] == "Horizontal" else 100
        gross_stroke = np.full(days, gross_base)
        net_stroke = gross_stroke * np.random.uniform(0.85, 0.90, days)
        spm = np.random.normal(8.5, 0.2, days)
        fillage = np.random.normal(92, 3, days)
        fillage = np.clip(fillage, 0, 100)
        motor_amps = (rates / 10) + 20 + np.random.normal(0, 2, days)
        pip = 300 + (rates * 0.3) + np.random.normal(0, 15, days)

    elif lift == "ESP":
        freq_hz = np.random.normal(58, 1, days)
        freq_hz = np.clip(freq_hz, 0, 65)
        motor_amps = (rates / 8) + 30 + np.random.normal(0, 3, days)
        pip = 500 + (rates * 0.2) + np.random.normal(0, 10, days)
        motor_temp = 170 + (motor_amps * 0.5) + np.random.normal(0, 2, days)

    elif lift == "Gas Lift":
        inj_rate = np.random.normal(600, 20, days)
        inj_press = np.random.normal(900, 15, days)
        cp = inj_press
        tp = inj_press * 0.3

    # C. Failure Logic
    num_failures = max(1, int(days / random.randint(180, 450)))
    fail_indices = []
    if days > 60:
        fail_indices = sorted(random.sample(range(10, days - 30), k=num_failures))

    status_col = ["OK"] * days
    notes_col = [""] * days

    for idx in fail_indices:
        mode = random.choice(FAILURE_MODES[lift]["Mid"])
        downtime = random.randint(3, 10)
        end = min(idx + downtime, days)

        rates[idx:end] = 0
        status_col[idx:end] = ["FAILURE"] * (end - idx)

        if lift == "Rod Pump":
            if "Parted" in mode:
                motor_amps[idx:end] *= 0.4
                fillage[idx:end] = 0
                net_stroke[idx:end] = 0
                pip[idx:end] += 200
            elif "Stuck" in mode:
                motor_amps[idx:end] *= 2.0
                spm[idx:end] = 0

        elif lift == "ESP":
            if "Shaft" in mode:
                motor_amps[idx:end] = 20
                pip[idx:end] = 2500
            elif "Short" in mode:
                motor_amps[idx:end] = 0
                freq_hz[idx:end] = 0

        elif lift == "Gas Lift":
            if "Trip" in mode:
                inj_rate[idx:end] = 0
                cp[idx:end] *= 0.5

        notes_col[idx] = f"ALERT: {mode}."
        if end < days:
            notes_col[end] = "Workover Complete."

    # D. End of Life Goal
    last_30 = slice(-30, None)
    if well["current_status_goal"] == "Degrading":
        decay = np.linspace(1.0, 0.7, 30)
        rates[last_30] *= decay
        if lift == "ESP":
            pip[last_30] *= decay
            motor_temp[last_30] *= 1.1
        elif lift == "Rod Pump":
            fillage[last_30] *= decay
            pip[last_30] *= decay

    # E. Test Logic
    test_oil = [None] * days
    test_water = [None] * days
    test_gas = [None] * days
    curr = random.randint(0, 20)
    while curr < days:
        if rates[curr] > 0:
            test_oil[curr] = round(rates[curr], 2)
            test_water[curr] = round(rates[curr] * 2.5, 2)
            test_gas[curr] = round(rates[curr] * 1.5, 2)
        curr += random.randint(10, 25)

    return pd.DataFrame({
        "date": dates,
        "asset_id": well["asset_id"],
        "facility_id": well["facility_id"],
        "true_oil": rates,
        "true_water": rates * 2.5,
        "true_gas": rates * 1.5,
        "test_oil": test_oil,
        "test_water": test_water,
        "test_gas": test_gas,
        "avg_tubing_pressure": tp,
        "avg_casing_pressure": cp,
        "spm": spm,
        "gross_stroke_len": gross_stroke,
        "net_stroke_len": net_stroke,
        "pump_fillage_pct": fillage,
        "avg_motor_amps": motor_amps,
        "freq_hz": freq_hz,
        "pump_intake_pressure": pip,
        "motor_temp_f": motor_temp,
        "injection_rate_mcf": inj_rate,
        "status": status_col,
        "notes": notes_col
    })

# --- 4. MASTER SIMULATION ---
def run_simulation():
    print("Generating Engineering-Grade Digital Twin base dataset (v3.1)...")
    well_meta, fac_meta = build_hierarchy()

    all_dfs = []
    for w in well_meta:
        all_dfs.append(generate_well_timeline(w))

    # Filter out any empty or all-NA DataFrames before concatenation to
    # avoid FutureWarning from pandas about concat with empty/all-NA entries.
    non_empty_dfs = [
        df for df in all_dfs
        if isinstance(df, pd.DataFrame) and not df.empty and not df.isna().all().all()
    ]
    if not non_empty_dfs:
        raise ValueError("No non-empty well timelines were generated; cannot build master_df.")

    # Pandas emits a FutureWarning about concatenation with empty/all-NA entries;
    # at this point we've filtered those out, so safely silence this specific warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        master_df = pd.concat(non_empty_dfs).sort_values(["date", "facility_id"])

    output = {"company": "Apex Permian", "assets": {}}
    for w in well_meta:
        output["assets"][w["asset_id"]] = {
            "type": "Well",
            "lift": w["lift_type"],
            "history": []
        }
    for fid in fac_meta:
        output["assets"][fid] = {"type": "Facility", "history": []}

    for fid in master_df["facility_id"].unique():
        fac_df = master_df[master_df["facility_id"] == fid]
        for date, day_data in fac_df.groupby("date"):
            d_str = date.strftime("%Y-%m-%d")

            esd = False
            if random.random() < 0.0005:
                esd = True

            total_oil = day_data["true_oil"].sum() if not esd else 0

            output["assets"][fid]["history"].append({
                "date": d_str,
                "oil_prod": round(total_oil, 2),
                "status": "ESD" if esd else "OK"
            })

            sum_true = day_data["true_oil"].sum()
            for _, row in day_data.iterrows():
                alloc_oil = 0
                if not esd and sum_true > 0:
                    alloc_oil = total_oil * (row["true_oil"] / sum_true)

                def clean(val):
                    return None if pd.isna(val) else round(val, 2)

                output["assets"][row["asset_id"]]["history"].append({
                    "date": d_str,
                    "allocated_oil": round(alloc_oil, 2),
                    "test_oil": row["test_oil"],
                    "avg_tubing_pressure": clean(row["avg_tubing_pressure"]),
                    "avg_casing_pressure": clean(row["avg_casing_pressure"]),
                    "avg_motor_amps": clean(row["avg_motor_amps"]),
                    "gross_stroke_len": clean(row["gross_stroke_len"]),
                    "net_stroke_len": clean(row["net_stroke_len"]),
                    "spm": clean(row["spm"]),
                    "pump_fillage_pct": clean(row["pump_fillage_pct"]),
                    "freq_hz": clean(row["freq_hz"]),
                    "pump_intake_pressure": clean(row["pump_intake_pressure"]),
                    "motor_temp_f": clean(row["motor_temp_f"]),
                    "injection_rate_mcf": clean(row["injection_rate_mcf"]),
                    "status": "SHUT_IN" if esd else row["status"],
                    "notes": row["notes"]
                })

    with open("apex_permian_v3_1_engineering.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(
        "Done: apex_permian_v3_1_engineering.json\n"
        "Note: Master.json is maintained separately as the unified packaged dataset, "
        "and Invoices.json is maintained separately as the invoice-only extract."
    )

if __name__ == "__main__":
    run_simulation()