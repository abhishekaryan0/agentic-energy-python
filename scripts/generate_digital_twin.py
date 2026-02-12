import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Same Snowflake / .env setup as generate_sample_data.py
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

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
    
    status_pool = ["Happy Path"]*12 + ["Failure"]*9 + ["Degrading"]*9
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
    if days < 1: return pd.DataFrame()

    # A. Base Production (Decline)
    t = np.arange(days)
    decline = 0.0006 if well["orientation"] == "Horizontal" else 0.0002
    rates = well["base_rate"] * np.exp(-decline * t)
    rates = rates * np.random.normal(1.0, 0.05, days) # Noise
    
    # B. Lift-Specific Parameters (KPIs)
    lift = well["lift_type"]
    
    # Common Defaults
    gross_stroke = [None]*days
    net_stroke = [None]*days
    spm = [None]*days
    fillage = [None]*days
    freq_hz = [None]*days
    motor_amps = [None]*days
    pip = [None]*days
    motor_temp = [None]*days
    inj_rate = [None]*days
    inj_press = [None]*days
    
    # Common Sensors
    tp = (rates * 0.1) + 200 # Base Tubing Press
    cp = (rates * 0.05) + 50 # Base Casing Press
    
    # --- ENGINEERING LOGIC ---
    if lift == "Rod Pump":
        # Gross Stroke (Surface)
        gross_base = 144 if well["orientation"] == "Horizontal" else 100
        gross_stroke = np.full(days, gross_base)
        
        # Net Stroke (Downhole)
        net_stroke = gross_stroke * np.random.uniform(0.85, 0.90, days)
        
        # Speed
        spm = np.random.normal(8.5, 0.2, days)
        
        # Fillage
        fillage = np.random.normal(92, 3, days)
        fillage = np.clip(fillage, 0, 100)
        
        # Amps
        motor_amps = (rates / 10) + 20 + np.random.normal(0, 2, days)
        
        # PIP (Added for Rod Pump)
        # Low PIP = Starvation risk. High PIP = Inefficient pumping.
        pip = 300 + (rates * 0.3) + np.random.normal(0, 15, days)

    elif lift == "ESP":
        # Frequency
        freq_hz = np.random.normal(58, 1, days)
        freq_hz = np.clip(freq_hz, 0, 65)
        
        # Amps
        motor_amps = (rates / 8) + 30 + np.random.normal(0, 3, days)
        
        # PIP
        pip = 500 + (rates * 0.2) + np.random.normal(0, 10, days)
        
        # Motor Temp
        motor_temp = 170 + (motor_amps * 0.5) + np.random.normal(0, 2, days)

    elif lift == "Gas Lift":
        # Injection Rate
        inj_rate = np.random.normal(600, 20, days)
        
        # Injection Pressure
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
                pip[idx:end] += 200 # Fluid level rises (pump not moving fluid)
            elif "Stuck" in mode:
                motor_amps[idx:end] *= 2.0
                spm[idx:end] = 0
                
        elif lift == "ESP":
            if "Shaft" in mode:
                motor_amps[idx:end] = 20
                pip[idx:end] = 2500 # Max static P
            elif "Short" in mode:
                motor_amps[idx:end] = 0
                freq_hz[idx:end] = 0
                
        elif lift == "Gas Lift":
            if "Trip" in mode:
                inj_rate[idx:end] = 0
                cp[idx:end] *= 0.5

        notes_col[idx] = f"ALERT: {mode}."
        if end < days: notes_col[end] = "Workover Complete."

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
            # As rate drops, PIP usually drops (starvation) OR rises (pump wear)
            # Modeling starvation here:
            pip[last_30] *= decay 

    # E. Test Logic
    test_oil = [None]*days
    test_water = [None]*days
    test_gas = [None]*days
    curr = random.randint(0, 20)
    while curr < days:
        if rates[curr] > 0:
            test_oil[curr] = round(rates[curr], 2)
            test_water[curr] = round(rates[curr]*2.5, 2)
            test_gas[curr] = round(rates[curr]*1.5, 2)
        curr += random.randint(10, 25)

    return pd.DataFrame({
        "date": dates,
        "asset_id": well["asset_id"],
        "facility_id": well["facility_id"],
        "lift_type": well["lift_type"],
        "true_oil": rates,
        "true_water": rates*2.5,
        "true_gas": rates*1.5,
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
        "pump_intake_pressure": pip, # SHARED for RP and ESP
        "motor_temp_f": motor_temp,
        "injection_rate_mcf": inj_rate,
        "status": status_col,
        "notes": notes_col
    })

# --- 4. SNOWFLAKE UPLOAD (same tables as generate_sample_data.py) ---
# Map digital twin column names -> well_sensor_readings field names (snake_case)
TWIN_TO_SENSOR_COL = {
    "avg_tubing_pressure": "tubing_pressure",
    "avg_casing_pressure": "casing_pressure",
    "spm": "strokes_per_minute",
    "pump_fillage_pct": "pump_fillage",
    "avg_motor_amps": "motor_current",
    "pump_intake_pressure": "pump_intake_pressure",
    "motor_temp_f": "motor_temp",
    "freq_hz": "freq_hz",  # use as-is if DB has it; else try "frequency"
    "injection_rate_mcf": "injection_rate",
}

def _clean_val(val):
    if pd.isna(val) or val is None:
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None

def upload_to_snowflake(master_df, well_meta):
    """Insert digital twin data into well_daily_production and well_sensor_readings."""
    try:
        from snowflake.connector import connect as snowflake_connect
        from config.sensor_config import get_all_sensor_columns
    except Exception as e:
        print(f"Cannot upload to Snowflake: {e}")
        return

    SNOWFLAKE_CONFIG = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
    }
    if not SNOWFLAKE_CONFIG.get("user") or not SNOWFLAKE_CONFIG.get("account"):
        print("Snowflake credentials not set (.env). Skipping upload.")
        return

    try:
        all_columns = get_all_sensor_columns()
    except Exception:
        all_columns = [
            "well_id", "timestamp", "lift_type",
            "strokes_per_minute", "torque", "polish_rod_load", "pump_fillage", "tubing_pressure",
            "motor_temp", "motor_current", "discharge_pressure", "pump_intake_pressure", "motor_voltage",
            "injection_rate", "injection_temperature", "bottomhole_pressure", "injection_pressure", "cycle_time",
            "surface_pressure", "casing_pressure", "wellhead_temp",
        ]

    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    try:
        print("Clearing previous data in well_daily_production and well_sensor_readings...")
        cursor.execute("DELETE FROM well_daily_production")
        cursor.execute("DELETE FROM well_sensor_readings")

        # --- Daily production (same schema as generate_sample_data) ---
        daily_tuples = []
        for _, row in master_df.iterrows():
            daily_tuples.append((
                row["asset_id"],
                row["date"].strftime("%Y-%m-%d"),
                _clean_val(row["true_oil"]) or 0,
                _clean_val(row["true_gas"]) or 0,
                _clean_val(row["true_water"]) or 0,
                row["lift_type"],
            ))
        cursor.executemany("""
            INSERT INTO well_daily_production
            (well_id, date, oil_volume, gas_volume, water_volume, lift_type)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, daily_tuples)
        print(f"  Inserted {len(daily_tuples):,} daily production records.")

        # --- Hourly sensor readings: 24 rows per well per day (daily value repeated) ---
        # Build mapping: twin col -> list of possible DB col names
        twin_to_db = {}
        for twin_col, db_col in TWIN_TO_SENSOR_COL.items():
            if db_col in all_columns:
                twin_to_db[twin_col] = db_col
            elif twin_col == "freq_hz" and "frequency" in all_columns:
                twin_to_db[twin_col] = "frequency"

        sensor_rows = []
        for _, row in master_df.iterrows():
            base_ts = pd.Timestamp(row["date"])
            for hour in range(24):
                ts = (base_ts + pd.Timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")
                rec = {"well_id": row["asset_id"], "timestamp": ts, "lift_type": row["lift_type"]}
                for col in all_columns:
                    if col in rec:
                        continue
                    rec[col] = None
                for twin_col, db_col in twin_to_db.items():
                    if twin_col in row.index:
                        val = _clean_val(row[twin_col])
                        if val is not None:
                            rec[db_col] = val
                if row["lift_type"] == "Gas Lift" and "injection_pressure" in all_columns and rec.get("injection_pressure") is None:
                    rec["injection_pressure"] = _clean_val(row.get("avg_casing_pressure"))
                sensor_rows.append(rec)

        batch_size = 5000
        placeholders = ", ".join(["%s"] * len(all_columns))
        for i in range(0, len(sensor_rows), batch_size):
            batch = sensor_rows[i : i + batch_size]
            data_tuples = [tuple(r.get(c) for c in all_columns) for r in batch]
            cursor.executemany(
                f"INSERT INTO well_sensor_readings ({', '.join(all_columns)}) VALUES ({placeholders})",
                data_tuples,
            )
            print(f"  Sensor batch {i // batch_size + 1}: {min(i + batch_size, len(sensor_rows)):,} / {len(sensor_rows):,}")
        conn.commit()
        print("SUCCESS: Digital twin data uploaded to Snowflake (same tables as generate_sample_data).")
    except Exception as e:
        conn.rollback()
        print(f"ERROR during Snowflake upload: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


# --- 5. MASTER SIMULATION ---
def run_simulation(upload_snowflake=False, write_json=True):
    print("Generating Engineering-Grade Digital Twin (v3.1)...")
    well_meta, fac_meta = build_hierarchy()
    
    all_dfs = []
    for w in well_meta:
        all_dfs.append(generate_well_timeline(w))
    master_df = pd.concat(all_dfs).sort_values(["date", "facility_id"])
    
    output = {"company": "Apex Permian", "assets": {}}
    for w in well_meta:
        output["assets"][w["asset_id"]] = {"type": "Well", "lift": w["lift_type"], "history": []}
    for fid in fac_meta:
        output["assets"][fid] = {"type": "Facility", "history": []}
    
    for fid in master_df["facility_id"].unique():
        fac_df = master_df[master_df["facility_id"] == fid]
        for date, day_data in fac_df.groupby("date"):
            d_str = date.strftime("%Y-%m-%d")
            
            esd = False
            if random.random() < 0.0005: esd = True
            
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
                    alloc_oil = total_oil * (row["true_oil"]/sum_true)
                
                def clean(val): return None if pd.isna(val) else round(val, 2)
                
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

    if write_json:
        with open("apex_permian_v3_1_engineering.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print("Done: apex_permian_v3_1_engineering.json")

    if upload_snowflake:
        upload_to_snowflake(master_df, well_meta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate digital twin data (JSON and/or Snowflake).")
    parser.add_argument("--snowflake", action="store_true", help="Upload to Snowflake (same tables as generate_sample_data)")
    parser.add_argument("--no-json", action="store_true", help="Skip writing apex_permian_v3_1_engineering.json")
    args = parser.parse_args()
    run_simulation(upload_snowflake=args.snowflake, write_json=not args.no_json)