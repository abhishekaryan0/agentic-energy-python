#!/usr/bin/env python3
"""
Update or delete one well's data for a date range.

Modes:
  - add      : Append N days of new data after the well's last record date.
  - replace  : Delete data in [start_date, end_date] for the well, then insert
               new data for that range (optionally with anomaly types).
  - delete   : Delete data in [start_date, end_date] for the well only.

# Replace last 14 days for Well_001_RodPump with anomaly data (e.g. DATA_MISMATCH + FLUID_POUND)
python scripts/update_well_data_range.py --well Well_001_RodPump --action replace \
  --start 2025-01-29 --end 2025-02-11 \
  --anomalies DATA_MISMATCH,FLUID_POUND

# Well_002: create data that will 100% trigger an anomaly (use a range ending recently)
# DATA_MISMATCH = zero production but sensors active → anomaly review will detect it every time
python scripts/update_well_data_range.py --well Well_002 --action replace \
  --start 2025-02-01 --end 2025-02-12 \
  --anomalies DATA_MISMATCH

# Then run:  python -m anomaly_review.service Well_002

When generating data, pressure and fillage are kept stable (with small jitter) unless
you pass PRESSURE_INSTABILITY or FLUID_POUND, so the anomaly you pass is the one that fires.
MODEL_DIVERGENCE uses a 20% production drop in the last 5 days. If you still get
PATTERN_SHIFT/other: delete the cached model (e.g. models/Well_002_RodPump/) and re-run
anomaly detection so the model retrains and sees the drop.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from snowflake.connector import connect as snowflake_connect

# Load .env from script directory
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.isfile(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

# Project root for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE'),
}

# Default initial rates by lift type (oil, gas, water) for decline curve
DEFAULT_RATES = {
    'Rod Pump': (300, 600, 400),
    'ESP': (1200, 2000, 1800),
    'Gas Lift': (800, 1500, 1200),
}

# Fallback sensor ranges if sensor_config not available
FALLBACK_RANGES = {
    'Rod Pump': {
        'strokes_per_minute': {'min': 0.5, 'max': 12},
        'torque': {'min': 100000, 'max': 1824000},
        'polish_rod_load': {'min': 0, 'max': 125000},
        'pump_fillage': {'min': 0, 'max': 100},
        'tubing_pressure': {'min': 0, 'max': 3000},
    },
    'ESP': {
        'motor_temp': {'min': 50, 'max': 1500},
        'motor_current': {'min': 0, 'max': 200},
        'discharge_pressure': {'min': 0, 'max': 5000},
        'pump_intake_pressure': {'min': 0, 'max': 5000},
        'motor_voltage': {'min': 0, 'max': 5000},
    },
    'Gas Lift': {
        'injection_rate': {'min': 0, 'max': 2000},
        'injection_temperature': {'min': 0, 'max': 1000},
        'bottomhole_pressure': {'min': 50, 'max': 5000},
        'injection_pressure': {'min': 100, 'max': 5000},
        'cycle_time': {'min': 5, 'max': 180},
    },
}

COMMON_SENSORS = [
    ('surface_pressure', 500, 150, 0, 2000),
    ('casing_pressure', 600, 200, 0, 2500),
    ('wellhead_temp', 40, 10, 20, 80),
    ('tubing_pressure', 800, 100, 0, 3000),
]

# Anomaly types we can inject (code, description, applies to)
ANOMALY_OPTIONS = [
    ('DATA_MISMATCH', 'Production: zero flow but sensors active (pump running)', ['Rod Pump', 'ESP']),
    ('MODEL_DIVERGENCE', 'Production: significant drop vs expected (e.g. 50% of normal)', ['Rod Pump', 'ESP', 'Gas Lift']),
    ('ENERGY_DIV', 'Energy: high amps but low oil (energy/production divergence)', ['Rod Pump', 'ESP']),
    ('FLUID_POUND', 'Sensor: low pump fillage (Rod Pump only)', ['Rod Pump']),
    ('PRESSURE_INSTABILITY', 'Sensor: high pressure variance', ['Rod Pump', 'ESP', 'Gas Lift']),
]


def get_conn():
    return snowflake_connect(**SNOWFLAKE_CONFIG)


def _field_to_pascal(col):
    return ''.join(w.capitalize() for w in col.split('_'))


def _pascal_to_snake(s):
    return ''.join('_' + c.lower() if c.isupper() else c for c in s).lstrip('_').replace('__', '_')


def get_sensor_ranges():
    try:
        from sensor_config import get_sensor_ranges, get_lift_types
        return {lt: get_sensor_ranges(lt) for lt in get_lift_types().keys()}
    except Exception:
        return FALLBACK_RANGES


def get_all_sensor_columns():
    try:
        from sensor_config import get_all_sensor_columns
        return get_all_sensor_columns()
    except Exception:
        return [
            'well_id', 'timestamp', 'lift_type',
            'strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure',
            'motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage',
            'injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time',
            'surface_pressure', 'casing_pressure', 'wellhead_temp',
        ]


def generate_decline(days, qi_oil, qi_gas, qi_water):
    Di_annual, b, D_term_annual, t_switch = 0.35, 0.75, 0.06, 365
    Di = Di_annual / 365.0
    D_term = D_term_annual / 365.0
    t = np.arange(days)
    q_oil = np.zeros(days)
    mask_hyp = t < t_switch
    q_oil[mask_hyp] = qi_oil / ((1 + b * Di * t[mask_hyp]) ** (1 / b))
    q_switch = qi_oil / ((1 + b * Di * t_switch) ** (1 / b))
    mask_exp = t >= t_switch
    q_oil[mask_exp] = q_switch * np.exp(-D_term * (t[mask_exp] - t_switch))
    q_gas = q_oil * (qi_gas / qi_oil) if qi_oil > 0 else q_oil
    q_water = q_oil * (qi_water / qi_oil) if qi_oil > 0 else q_oil
    return q_oil, q_gas, q_water


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------


def list_wells(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
    rows = cur.fetchall()
    cur.close()
    return [r[0] for r in rows] if rows else []


def get_well_info(conn, well_id):
    """Return (lift_type, last_prod_date, last_sensor_ts) or (None, None, None)."""
    cur = conn.cursor()
    lift_type, last_prod, last_ts = None, None, None
    cur.execute(
        "SELECT lift_type, MAX(date) FROM well_daily_production WHERE well_id = %s GROUP BY lift_type",
        (well_id,)
    )
    row = cur.fetchone()
    if row:
        lift_type, last_prod = row[0], row[1]
    cur.execute(
        "SELECT MAX(timestamp) FROM well_sensor_readings WHERE well_id = %s",
        (well_id,)
    )
    row = cur.fetchone()
    if row and row[0]:
        last_ts = row[0]
    cur.close()
    return lift_type, last_prod, last_ts


def get_baseline_production(conn, well_id, before_date, days=7):
    """Average oil/gas/water for the last `days` days before `before_date`."""
    cur = conn.cursor()
    cur.execute("""
        SELECT AVG(oil_volume), AVG(gas_volume), AVG(water_volume)
        FROM well_daily_production
        WHERE well_id = %s AND date < %s AND date >= %s
    """, (well_id, before_date, before_date - timedelta(days=days)))
    row = cur.fetchone()
    cur.close()
    if row and row[0] is not None:
        return (float(row[0]), float(row[1]), float(row[2]))
    return None


def delete_well_range(conn, well_id, start_date, end_date):
    """Delete well_daily_production and well_sensor_readings for well in [start_date, end_date]."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM well_daily_production WHERE well_id = %s AND date >= %s AND date <= %s",
        (well_id, start_date, end_date)
    )
    prod_deleted = cur.rowcount
    # Sensor: timestamp can be datetime; use date range
    cur.execute("""
        DELETE FROM well_sensor_readings
        WHERE well_id = %s AND DATE(timestamp) >= %s AND DATE(timestamp) <= %s
    """, (well_id, start_date, end_date))
    sensor_deleted = cur.rowcount
    conn.commit()
    cur.close()
    return prod_deleted, sensor_deleted


# -----------------------------------------------------------------------------
# Data generation for a date range
# -----------------------------------------------------------------------------


def generate_well_data_for_range(well_id, lift_type, start_date, end_date, sensor_ranges, anomaly_types=None, baseline_rates=None):
    """
    Generate daily production and hourly sensor DataFrames for one well
    for the given date range. Optionally apply anomaly_types (list of codes).
    baseline_rates: optional (oil_avg, gas_avg, water_avg) to extend from existing level (e.g. for add).
    """
    anomaly_types = anomaly_types or []
    start_d = start_date if isinstance(start_date, date) else start_date.date()
    end_d = end_date if isinstance(end_date, date) else end_date.date()
    dates_daily = pd.date_range(start=start_d, end=end_d, freq='D', inclusive='both')
    n_days = len(dates_daily)
    if n_days == 0:
        return None, None

    # Hourly timestamps
    dates_hf = pd.date_range(start=start_d, end=end_d, freq='h', inclusive='both')
    n_hours = len(dates_hf)

    # Production: use baseline (e.g. last 7d avg) for continuity, or decline curve
    if baseline_rates and len(baseline_rates) == 3 and all(baseline_rates):
        qi_o, qi_g, qi_w = baseline_rates
        qo = np.maximum(0, np.random.normal(qi_o, 0.05 * qi_o, n_days))
        qg = np.maximum(0, np.random.normal(qi_g, 0.05 * qi_g, n_days))
        qw = np.maximum(0, np.random.normal(qi_w, 0.05 * qi_w, n_days))
    else:
        qi_o, qi_g, qi_w = DEFAULT_RATES.get(lift_type, (500, 800, 500))
        qo, qg, qw = generate_decline(n_days, qi_o, qi_g, qi_w)
        qo *= np.random.normal(1, 0.05, n_days)
        qg *= np.random.normal(1, 0.05, n_days)
        qw *= np.random.normal(1, 0.05, n_days)

    daily_df = pd.DataFrame({
        'WellID': well_id,
        'Date': dates_daily,
        'Oil_Volume': qo,
        'Gas_Volume': qg,
        'Water_Volume': qw,
        'Lift_Type': lift_type,
    })

    # Hourly sensors
    ranges = sensor_ranges.get(lift_type, {})
    w_hf = pd.DataFrame({'WellID': well_id, 'Timestamp': dates_hf})
    for field_name, rng in ranges.items():
        mn, mx = float(rng.get('min', 0)), float(rng.get('max', 1))
        if mn >= mx:
            mx = mn + 1
        mean, std = (mn + mx) / 2, (mx - mn) / 6
        col = _field_to_pascal(field_name)
        w_hf[col] = np.clip(np.random.normal(mean, max(std, 0.01), n_hours), mn, mx)
    for field, mean, std, minv, maxv in COMMON_SENSORS:
        if field not in ranges and _field_to_pascal(field) not in w_hf.columns:
            w_hf[_field_to_pascal(field)] = np.clip(np.random.normal(mean, std, n_hours), minv, maxv)

    # ---- Anomaly injection ----
    def _date_set(series, n):
        return set(d.date() for d in series[-n:]) if n <= len(series) else set()
    last_3_dates = _date_set(dates_daily, 3)
    last_5_dates = _date_set(dates_daily, 5)
    last_14_dates = _date_set(dates_daily, 14)

    if 'DATA_MISMATCH' in anomaly_types:
        mask = pd.to_datetime(daily_df['Date']).dt.date.isin(last_5_dates)
        daily_df.loc[mask, ['Oil_Volume', 'Gas_Volume', 'Water_Volume']] = 0.0

    if 'MODEL_DIVERGENCE' in anomaly_types and n_days >= 5:
        # Strong drop (0.2x) so actual is well below model prediction; checker needs >25% divergence.
        # If you still get PATTERN_SHIFT/other: delete cached model (models/Well_XXX/) and re-run detection.
        mask = pd.to_datetime(daily_df['Date']).dt.date.isin(last_5_dates)
        daily_df.loc[mask, 'Oil_Volume'] *= 0.2
        daily_df.loc[mask, 'Gas_Volume'] *= 0.2
        daily_df.loc[mask, 'Water_Volume'] *= 0.2

    if 'FLUID_POUND' in anomaly_types and lift_type == 'Rod Pump':
        fillage_col = 'PumpFillage' if 'PumpFillage' in w_hf.columns else 'pump_fillage'
        if fillage_col not in w_hf.columns:
            fillage_col = next((c for c in w_hf.columns if 'fillage' in c.lower()), None)
        if fillage_col is not None and n_hours >= 24 * 3:
            cutoff = dates_hf[-1] - timedelta(days=3)
            w_hf.loc[w_hf['Timestamp'] >= cutoff, fillage_col] = 65.0

    if 'ENERGY_DIV' in anomaly_types:
        # High amps, low oil in last 5 days: boost motor_current, cut production (if not already zero)
        if 'DATA_MISMATCH' not in anomaly_types and n_days >= 5:
            mask = pd.to_datetime(daily_df['Date']).dt.date.isin(last_5_dates)
            daily_df.loc[mask, 'Oil_Volume'] *= 0.2
            daily_df.loc[mask, 'Gas_Volume'] *= 0.2
            daily_df.loc[mask, 'Water_Volume'] *= 0.2
        if 'motor_current' in w_hf.columns or 'MotorCurrent' in w_hf.columns:
            mc_col = 'MotorCurrent' if 'MotorCurrent' in w_hf.columns else 'motor_current'
            if mc_col not in w_hf.columns:
                mc_col = next((c for c in w_hf.columns if 'current' in c.lower()), None)
            if mc_col is not None and n_hours >= 24 * 5:
                cutoff = dates_hf[-1] - timedelta(days=5)
                # Scale up amps by 1.5x in last 5 days
                w_hf.loc[w_hf['Timestamp'] >= cutoff, mc_col] *= 1.5

    if 'PRESSURE_INSTABILITY' in anomaly_types:
        tp_col = 'TubingPressure' if 'TubingPressure' in w_hf.columns else 'tubing_pressure'
        if tp_col not in w_hf.columns:
            tp_col = next((c for c in w_hf.columns if 'pressure' in c.lower() and 'tubing' in c.lower()), None)
        if tp_col is None:
            for c in w_hf.columns:
                if 'tubing' in c.lower():
                    tp_col = c
                    break
        if tp_col is not None and n_hours >= 24 * 3:
            cutoff = dates_hf[-1] - timedelta(days=3)
            # Add high variance: multiply by random 0.5–1.5
            idx = w_hf['Timestamp'] >= cutoff
            w_hf.loc[idx, tp_col] *= np.random.uniform(0.5, 1.5, idx.sum())

    # When NOT injecting PRESSURE_INSTABILITY, keep pressure stable but with small jitter so:
    # (1) flatline detection (std=0 in last 24h) does not mask our data, (2) CV stays < 18%,
    # (3) values stay below median*1.25 so PATTERN_SHIFT "pressure spike" does not fire.
    if 'PRESSURE_INSTABILITY' not in anomaly_types:
        tp_col = 'TubingPressure' if 'TubingPressure' in w_hf.columns else next(
            (c for c in w_hf.columns if 'tubing' in c.lower() and 'pressure' in c.lower()), None
        )
        if tp_col is not None:
            # Use low percentile so when merged with rest of well, we don't count as "spike"
            base = float(np.percentile(w_hf[tp_col].values, 30))
            w_hf[tp_col] = np.clip(base + np.random.normal(0, 5, n_hours), base - 15, base + 15)
        cp_col = 'CasingPressure' if 'CasingPressure' in w_hf.columns else next(
            (c for c in w_hf.columns if 'casing' in c.lower() and 'pressure' in c.lower()), None
        )
        if cp_col is not None:
            base = float(np.percentile(w_hf[cp_col].values, 30))
            w_hf[cp_col] = np.clip(base + np.random.normal(0, 5, n_hours), base - 15, base + 15)
        # Keep fillage stable and high when not FLUID_POUND so PATTERN_SHIFT "fillage drop" does not fire
        if 'FLUID_POUND' not in anomaly_types:
            fill_col = 'PumpFillage' if 'PumpFillage' in w_hf.columns else next(
                (c for c in w_hf.columns if 'fillage' in c.lower()), None
            )
            if fill_col is not None:
                w_hf[fill_col] = np.clip(88 + np.random.normal(0, 2, n_hours), 82, 98)

    return daily_df, w_hf


def insert_well_data(conn, well_id, lift_type, daily_df, hf_df, all_columns):
    """Insert daily and hourly data into Snowflake."""
    cur = conn.cursor()

    daily_tuples = [
        (well_id, row['Date'].strftime('%Y-%m-%d'),
         float(row['Oil_Volume']), float(row['Gas_Volume']),
         float(row['Water_Volume']), lift_type)
        for _, row in daily_df.iterrows()
    ]
    cur.executemany("""
        INSERT INTO well_daily_production (well_id, date, oil_volume, gas_volume, water_volume, lift_type)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, daily_tuples)
    prod_inserted = len(daily_tuples)

    hf_df = hf_df.copy()
    hf_df['Timestamp'] = hf_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    column_mapping = {'WellID': 'well_id', 'Timestamp': 'timestamp'}
    for c in hf_df.columns:
        if c not in column_mapping:
            column_mapping[c] = _pascal_to_snake(c).replace(' ', '_')
    hf_df = hf_df.rename(columns=column_mapping)
    hf_df['lift_type'] = lift_type
    for col in all_columns:
        if col not in hf_df.columns:
            hf_df[col] = None
    hf_df = hf_df[all_columns].replace({np.nan: None, np.inf: None, -np.inf: None})

    batch_size = 5000
    total = len(hf_df)
    placeholders = ', '.join(['%s'] * len(all_columns))
    for i in range(0, total, batch_size):
        batch = hf_df.iloc[i:i + batch_size]
        data_tuples = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        cur.executemany(f"""
            INSERT INTO well_sensor_readings ({', '.join(all_columns)})
            VALUES ({placeholders})
        """, data_tuples)
    conn.commit()
    cur.close()
    return prod_inserted, total


# -----------------------------------------------------------------------------
# Interactive prompts
# -----------------------------------------------------------------------------


def prompt_well_id(conn):
    wells = list_wells(conn)
    if not wells:
        print("No wells found in well_daily_production. Create data first (e.g. generate_sample_data.py).")
        return None
    print("Wells in database:")
    for i, w in enumerate(wells, 1):
        print(f"  {i}. {w}")
    print(f"  0. Enter well_id manually")
    try:
        choice = input("Choice (number or well_id): ").strip()
    except EOFError:
        return None
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice)
        if idx == 0:
            return input("Well ID: ").strip() or None
        if 1 <= idx <= len(wells):
            return wells[idx - 1]
        return None
    return choice if choice in wells else choice


def prompt_dates_for_replace_delete():
    while True:
        start_s = input("Start date (YYYY-MM-DD): ").strip()
        if not start_s:
            return None, None
        end_s = input("End date (YYYY-MM-DD): ").strip()
        if not end_s:
            return None, None
        try:
            start_d = datetime.strptime(start_s, '%Y-%m-%d').date()
            end_d = datetime.strptime(end_s, '%Y-%m-%d').date()
            if start_d <= end_d:
                return start_d, end_d
        except ValueError:
            pass
        print("Invalid dates. Use YYYY-MM-DD and ensure start <= end.")


def prompt_anomaly_types(lift_type):
    print("\nAnomaly types to inject (select by number, comma-separated; 0 = none / healthy only):")
    valid = []
    for i, (code, desc, types) in enumerate(ANOMALY_OPTIONS, 1):
        if lift_type in types:
            print(f"  {i}. {code}: {desc}")
            valid.append(code)
    print("  0. No anomalies (healthy data only)")
    try:
        raw = input("Choice(s), e.g. 1,3,4 or 0: ").strip()
    except EOFError:
        return []
    if not raw or raw == '0':
        return []
    chosen = []
    for part in raw.replace(',', ' ').split():
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(ANOMALY_OPTIONS):
                code = ANOMALY_OPTIONS[idx - 1][0]
                if code in valid and code not in chosen:
                    chosen.append(code)
    return chosen


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_add(conn, well_id, days, anomaly_types=None):
    lift_type, last_prod, last_ts = get_well_info(conn, well_id)
    if not lift_type:
        print(f"Well {well_id} has no production record. Use 'replace' with explicit start/end to create data.")
        return
    # Start after last date
    if last_prod:
        start_d = last_prod + timedelta(days=1)
    else:
        start_d = date.today() - timedelta(days=days)
    end_d = start_d + timedelta(days=days - 1)
    if end_d > date.today():
        end_d = date.today()
    print(f"Adding data from {start_d} to {end_d} ({lift_type})")
    baseline_rates = get_baseline_production(conn, well_id, start_d, days=7) if last_prod else None
    sensor_ranges = get_sensor_ranges()
    daily_df, hf_df = generate_well_data_for_range(
        well_id, lift_type, start_d, end_d, sensor_ranges, anomaly_types, baseline_rates=baseline_rates
    )
    if daily_df is None or daily_df.empty:
        print("No dates in range. Nothing to insert.")
        return
    all_columns = get_all_sensor_columns()
    prod_n, sensor_n = insert_well_data(conn, well_id, lift_type, daily_df, hf_df, all_columns)
    print(f"Inserted: {prod_n} daily, {sensor_n} hourly records.")


def run_replace(conn, well_id, start_date, end_date, anomaly_types=None):
    lift_type, _, _ = get_well_info(conn, well_id)
    if not lift_type:
        # Try to infer from well_id name
        if 'RodPump' in well_id or 'Rod Pump' in well_id:
            lift_type = 'Rod Pump'
        elif 'ESP' in well_id:
            lift_type = 'ESP'
        elif 'GasLift' in well_id or 'Gas Lift' in well_id:
            lift_type = 'Gas Lift'
        else:
            # Default to Rod Pump so Well_002 etc. work non-interactively (anomaly will still fire)
            lift_type = 'Rod Pump'
            print(f"Well {well_id} has no existing data; using lift_type={lift_type}. Override by passing a well ID like Well_002_RodPump if needed.")
    prod_del, sensor_del = delete_well_range(conn, well_id, start_date, end_date)
    print(f"Deleted: {prod_del} daily, {sensor_del} hourly records in range.")
    print(f"Generating new data for {start_date} to {end_date} ({lift_type})...")
    sensor_ranges = get_sensor_ranges()
    daily_df, hf_df = generate_well_data_for_range(well_id, lift_type, start_date, end_date, sensor_ranges, anomaly_types)
    if daily_df is None or daily_df.empty:
        print("No dates in range. Nothing to insert.")
        return
    all_columns = get_all_sensor_columns()
    prod_n, sensor_n = insert_well_data(conn, well_id, lift_type, daily_df, hf_df, all_columns)
    print(f"Inserted: {prod_n} daily, {sensor_n} hourly records.")


def run_delete(conn, well_id, start_date, end_date):
    prod_del, sensor_del = delete_well_range(conn, well_id, start_date, end_date)
    print(f"Deleted: {prod_del} daily, {sensor_del} hourly records.")


def main():
    parser = argparse.ArgumentParser(description="Update or delete one well's data for a date range.")
    parser.add_argument("--well", "-w", help="Well ID (optional; will prompt if not set)")
    parser.add_argument("--action", "-a", choices=["add", "replace", "delete"], required=True,
                        help="add = append N days after last record; replace = replace range; delete = delete range")
    parser.add_argument("--days", "-d", type=int, help="For add: number of days to append")
    parser.add_argument("--start", "-s", help="Start date YYYY-MM-DD (for replace/delete)")
    parser.add_argument("--end", "-e", help="End date YYYY-MM-DD (for replace/delete)")
    parser.add_argument("--anomalies", "-n", help="Comma-separated anomaly codes (e.g. DATA_MISMATCH,MODEL_DIVERGENCE). Only for add/replace. Omit to be prompted.")
    args = parser.parse_args()

    if not SNOWFLAKE_CONFIG.get('user') or not SNOWFLAKE_CONFIG.get('password'):
        print("Set SNOWFLAKE_* in .env or environment.")
        return 1

    conn = get_conn()
    well_id = args.well
    if not well_id:
        well_id = prompt_well_id(conn)
    if not well_id:
        print("No well selected.")
        return 1

    action = args.action
    anomaly_types = None
    if args.anomalies:
        anomaly_types = [x.strip() for x in args.anomalies.split(',') if x.strip()]

    if action == "add":
        days = args.days
        if days is None:
            try:
                days = int(input("Number of days to add: ").strip())
            except (ValueError, EOFError):
                print("Invalid days.")
                return 1
        if days < 1:
            print("Days must be >= 1.")
            return 1
        if anomaly_types is None:
            lift_type, _, _ = get_well_info(conn, well_id)
            if lift_type:
                anomaly_types = prompt_anomaly_types(lift_type)
        run_add(conn, well_id, days, anomaly_types)

    elif action in ("replace", "delete"):
        start_date, end_date = args.start, args.end
        if not start_date or not end_date:
            start_date, end_date = prompt_dates_for_replace_delete()
        if not start_date or not end_date:
            print("Start and end date required for replace/delete.")
            return 1
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date() if isinstance(end_date, str) else end_date
        if action == "replace":
            if anomaly_types is None:
                lift_type, _, _ = get_well_info(conn, well_id)
                if not lift_type:
                    if 'RodPump' in well_id:
                        lift_type = 'Rod Pump'
                    elif 'ESP' in well_id:
                        lift_type = 'ESP'
                    elif 'GasLift' in well_id:
                        lift_type = 'Gas Lift'
                    else:
                        lift_type = 'Rod Pump'
                anomaly_types = prompt_anomaly_types(lift_type)
            run_replace(conn, well_id, start_date, end_date, anomaly_types)
        else:
            run_delete(conn, well_id, start_date, end_date)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
