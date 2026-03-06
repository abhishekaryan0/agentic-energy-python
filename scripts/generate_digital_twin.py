import io
import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Same Snowflake / .env setup as generate_sample_data.py
# Same Snowflake / .env setup as generate_sample_data.py
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

_env_path = os.path.join(_sys_root, ".env")
if os.path.isfile(_env_path):
    from dotenv import load_dotenv

    load_dotenv(_env_path)

# --- CONFIGURATION ---
START_DATE = datetime(2014, 1, 1)
END_DATE = datetime(2024, 6, 1)
TOTAL_DAYS = (END_DATE - START_DATE).days

# --- 1. FAILURE LIBRARY (KPI IMPACT) ---
FAILURE_MODES = {
    "Rod Pump": {
        "Early": ["Polished Rod Clamp Slip", "Misaligned Unit"],
        "Mid": ["Parted Rods", "Stuck Pump", "Gearbox Failure"],
        "Late": ["Hole in Tubing", "Casing Leak", "Unit Structural Fatigue"],
    },
    "ESP": {
        "Early": ["Splicing Failure", "Improper Cooling"],
        "Mid": ["Broken Shaft", "Cable Short", "Vibration Trip"],
        "Late": ["Pump Stage Wear", "Scale Deposition"],
    },
    "Gas Lift": {
        "Early": ["Valve Installation Error"],
        "Mid": ["Compressor Trip", "Injection Line Freeze"],
        "Late": ["Scale Restriction", "Liquid Loading"],
    },
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

        well_registry.append(
            {
                "asset_id": w_name,
                "lease": lease,
                "facility_id": fac_id,
                "lift_type": defn["lift"],
                "orientation": defn["orient"],
                "spud_date": spud_date,
                "current_status_goal": status_pool[i],
                "base_rate": 800 if defn["orient"] == "Horizontal" else 150,
            }
        )

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
    tp = (rates * 0.1) + 200  # Base Tubing Press
    cp = (rates * 0.05) + 50  # Base Casing Press

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
                pip[idx:end] += 200  # Fluid level rises (pump not moving fluid)
            elif "Stuck" in mode:
                motor_amps[idx:end] *= 2.0
                spm[idx:end] = 0

        elif lift == "ESP":
            if "Shaft" in mode:
                motor_amps[idx:end] = 20
                pip[idx:end] = 2500  # Max static P
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
            # As rate drops, PIP usually drops (starvation) OR rises (pump wear)
            # Modeling starvation here:
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

    return pd.DataFrame(
        {
            "date": dates,
            "asset_id": well["asset_id"],
            "facility_id": well["facility_id"],
            "lift_type": well["lift_type"],
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
            "pump_intake_pressure": pip,  # SHARED for RP and ESP
            "motor_temp_f": motor_temp,
            "injection_rate_mcf": inj_rate,
            "status": status_col,
            "notes": notes_col,
        }
    )


# --- 4a. S3 UPLOAD (tenant-wise parquet, same bucket as deploy) ---
# Bucket layout: only under organization_id. No top-level models/.
#   s3://BUCKET/{organization_id}/model/   <- single model (or model artifacts) per org
#   s3://BUCKET/{organization_id}/sensor_data/  <- parquet (this script)
# Writes under sensor_data: daily_production.parquet, well_metadata.parquet, well_sensor_readings.parquet
S3_SENSOR_DATA_PREFIX = "sensor_data"  # under {organization_id}/
HOURLY_NOISE_PCT = 0.01  # ±1% per hour so 24h has small variance (same as Snowflake path)


def _uniform_production_df(master_df):
    """Return a copy with uniform dtypes and column names for parquet."""
    df = master_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


def _uniform_well_metadata(well_meta):
    """Convert well_meta list of dicts to a uniform DataFrame for parquet."""
    rows = []
    for w in well_meta:
        rows.append({
            "asset_id": w["asset_id"],
            "lease": w["lease"],
            "facility_id": w.get("facility_id"),
            "lift_type": w["lift_type"],
            "orientation": w.get("orientation"),
            "spud_date": w["spud_date"].strftime("%Y-%m-%d") if w.get("spud_date") else None,
            "current_status_goal": w.get("current_status_goal"),
            "base_rate": float(w.get("base_rate", 0)),
        })
    return pd.DataFrame(rows)


def _build_well_sensor_readings_df(master_df, organization_id):
    """Build hourly well_sensor_readings from daily master_df (same logic as Snowflake path).
    Returns a DataFrame with canonical sensor columns for parquet.
    """
    # Canonical columns for S3 parquet (aligned with TWIN_TO_SENSOR_COL + ids)
    sensor_cols = [
        "organization_id", "well_id", "timestamp", "lift_type",
        "tubing_pressure", "casing_pressure", "strokes_per_minute", "pump_fillage",
        "motor_current", "pump_intake_pressure", "motor_temp", "freq_hz",
        "injection_rate", "surface_stroke_length", "downhole_net_stroke",
        "injection_pressure", "well_status",
    ]
    twin_to_sensor = {
        "avg_tubing_pressure": "tubing_pressure",
        "avg_casing_pressure": "casing_pressure",
        "spm": "strokes_per_minute",
        "pump_fillage_pct": "pump_fillage",
        "avg_motor_amps": "motor_current",
        "pump_intake_pressure": "pump_intake_pressure",
        "motor_temp_f": "motor_temp",
        "freq_hz": "freq_hz",
        "injection_rate_mcf": "injection_rate",
        "gross_stroke_len": "surface_stroke_length",
        "net_stroke_len": "downhole_net_stroke",
    }
    rows = []
    for _, row in master_df.iterrows():
        base_ts = pd.Timestamp(row["date"])
        for hour in range(24):
            ts = base_ts + pd.Timedelta(hours=hour)
            rec = {c: None for c in sensor_cols}
            rec["organization_id"] = organization_id
            rec["well_id"] = row["asset_id"]
            rec["timestamp"] = ts
            rec["lift_type"] = row["lift_type"]
            rec["well_status"] = str(row["status"]) if row.get("status") else None
            for twin_col, out_col in twin_to_sensor.items():
                if twin_col in row.index:
                    val = _clean_val(row[twin_col])
                    if val is not None:
                        noise = 1.0 + np.random.uniform(-HOURLY_NOISE_PCT, HOURLY_NOISE_PCT)
                        rec[out_col] = round(val * noise, 2)
            if row["lift_type"] == "Gas Lift" and rec.get("injection_pressure") is None:
                rec["injection_pressure"] = _clean_val(row.get("avg_casing_pressure"))
            rows.append(rec)
    df = pd.DataFrame(rows, columns=sensor_cols)
    # Uniform dtypes for parquet
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["tubing_pressure", "casing_pressure", "strokes_per_minute", "pump_fillage",
                "motor_current", "pump_intake_pressure", "motor_temp", "freq_hz",
                "injection_rate", "surface_stroke_length", "downhole_net_stroke", "injection_pressure"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df


def upload_to_s3(master_df, well_meta, organization_id=None):
    """Write digital twin data to S3 under org's sensor_data folder (same bucket as deploy).
    Bucket has only org-level folders: {organization_id}/model/ (one model per org), {organization_id}/sensor_data/.
    Writes: s3://BUCKET/{organization_id}/sensor_data/daily_production.parquet, well_metadata.parquet, well_sensor_readings.parquet
    """
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

    # Under bucket: only {organization_id}/model/ and {organization_id}/sensor_data/
    sensor_prefix = f"{organization_id}/{S3_SENSOR_DATA_PREFIX}"

    prod_df = _uniform_production_df(master_df)
    meta_df = _uniform_well_metadata(well_meta)
    sensor_df = _build_well_sensor_readings_df(master_df, organization_id)

    buf_prod = io.BytesIO()
    buf_meta = io.BytesIO()
    buf_sensor = io.BytesIO()
    prod_df.to_parquet(buf_prod, index=False)
    buf_prod.seek(0)
    meta_df.to_parquet(buf_meta, index=False)
    buf_meta.seek(0)
    sensor_df.to_parquet(buf_sensor, index=False)
    buf_sensor.seek(0)

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=f"{sensor_prefix}/daily_production.parquet", Body=buf_prod.getvalue())
    s3.put_object(Bucket=bucket, Key=f"{sensor_prefix}/well_metadata.parquet", Body=buf_meta.getvalue())
    s3.put_object(Bucket=bucket, Key=f"{sensor_prefix}/well_sensor_readings.parquet", Body=buf_sensor.getvalue())

    print(f"SUCCESS: Sensor data parquet uploaded to s3://{bucket}/{sensor_prefix}/")
    print(f"  - daily_production.parquet ({len(prod_df):,} rows)")
    print(f"  - well_metadata.parquet ({len(meta_df):,} rows)")
    print(f"  - well_sensor_readings.parquet ({len(sensor_df):,} rows)")


# --- 4b. SNOWFLAKE UPLOAD (same tables as generate_sample_data.py) ---
# BOE = Barrel of Oil Equivalent. Client formula: BOE = (bbl oil) + (mcf gas / 6)
# Example: 20 bbl oil + 372 mcf gas -> BOE = 20 + 62 = 82
def compute_boe(oil_bbl, gas_mcf):
    """BOE = oil (bbl) + gas (mcf) / 6."""
    try:
        return round(float(oil_bbl or 0) + float(gas_mcf or 0) / 6.0, 2)
    except (TypeError, ValueError):
        return None


# Twin data we have but do NOT insert (no matching DB column or not yet used):
# - facility_id (prod table has no facility_id)
# - test_oil, test_water, test_gas (periodic test data; prod table has no test columns)
# - notes (free text; could go to a separate table or well_sensor_readings if added)
# We now insert: boe (prod), well_status + stroke lengths (sensor) when columns exist.

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
    "gross_stroke_len": "surface_stroke_length",  # Rod Pump surface stroke
    "net_stroke_len": "downhole_net_stroke",  # Rod Pump downhole net stroke
}


def _clean_val(val):
    if pd.isna(val) or val is None:
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None


def _get_table_columns(cursor, database, schema, table_name):
    """Return list of column names (lowercase) for the table, in order."""
    cursor.execute(
        """
        SELECT LOWER(COLUMN_NAME) AS col
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE LOWER(TABLE_CATALOG) = LOWER(%s)
          AND LOWER(TABLE_SCHEMA) = LOWER(%s)
          AND LOWER(TABLE_NAME) = LOWER(%s)
        ORDER BY ORDINAL_POSITION
    """,
        (database, schema, table_name),
    )
    return [row[0] for row in cursor.fetchall()]


def upload_to_snowflake(master_df, well_meta, organization_id=None):
    """Insert digital twin data into well_daily_production and well_sensor_readings.
    Uses only columns that exist in the DB so it works when not all columns are present.
    organization_id: UUID from your organizations Postgres; can also set env ORGANIZATION_ID.
    """
    organization_id = organization_id or os.getenv("ORGANIZATION_ID")
    if not organization_id:
        print("ORGANIZATION_ID not set (env or argument). Skipping Snowflake upload.")
        return
    try:
        from snowflake.connector import connect as snowflake_connect
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

    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()
    db = SNOWFLAKE_CONFIG.get("database", "AT_RISK_ASSETS")
    schema = SNOWFLAKE_CONFIG.get("schema", "SENSOR_DATA")

    try:
        # Use actual DB columns so we don't require columns that don't exist
        prod_columns = _get_table_columns(cursor, db, schema, "WELL_DAILY_PRODUCTION")
        sensor_columns = _get_table_columns(cursor, db, schema, "WELL_SENSOR_READINGS")

        # Exclude auto/system columns from insert (order matches twin data)
        _prod_col_order = [
            "organization_id",
            "well_id",
            "date",
            "oil_volume",
            "gas_volume",
            "water_volume",
            "boe",
            "lift_type",
            "status",
            "notes",
            "test_oil",
            "test_water",
            "test_gas",
            "facility_id",
        ]
        insert_prod_cols = [c for c in _prod_col_order if c in prod_columns]
        if not insert_prod_cols:
            print(
                "ERROR: well_daily_production has no insertable columns. Run sync_snowflake_schema.py or create table."
            )
            return

        # Sensor: use only columns that exist in the table (order preserved)
        insert_sensor_cols = [
            c for c in sensor_columns if c not in ("id", "created_at")
        ]
        if not insert_sensor_cols:
            print("ERROR: well_sensor_readings has no insertable columns.")
            return

        print(f"Production columns for insert: {insert_prod_cols}")
        print(
            f"Sensor columns for insert: {len(insert_sensor_cols)} (first 10: {insert_sensor_cols[:10]}...)"
        )

        print(
            "Clearing previous data for this organization in well_daily_production and well_sensor_readings..."
        )
        cursor.execute("DELETE FROM well_daily_production WHERE organization_id = %s", (organization_id,))
        cursor.execute("DELETE FROM well_sensor_readings WHERE organization_id = %s", (organization_id,))

        # --- Daily production: only insert into columns that exist ---
        daily_tuples = []
        for _, row in master_df.iterrows():
            oil = _clean_val(row["true_oil"]) or 0
            gas = _clean_val(row["true_gas"]) or 0
            boe = compute_boe(oil, gas)
            # Build tuple in same order as insert_prod_cols
            vals = []
            for col in insert_prod_cols:
                if col == "organization_id":
                    vals.append(organization_id)
                elif col == "well_id":
                    vals.append(row["asset_id"])
                elif col == "date":
                    vals.append(row["date"].strftime("%Y-%m-%d"))
                elif col == "oil_volume":
                    vals.append(oil)
                elif col == "gas_volume":
                    vals.append(gas)
                elif col == "water_volume":
                    vals.append(_clean_val(row["true_water"]) or 0)
                elif col == "boe":
                    vals.append(boe)
                elif col == "lift_type":
                    vals.append(row["lift_type"])
                elif col == "status":
                    vals.append(str(row["status"]) if row.get("status") else None)
                elif col == "notes":
                    vals.append(str(row["notes"])[:500] if row.get("notes") else None)
                elif col == "test_oil":
                    vals.append(_clean_val(row.get("test_oil")))
                elif col == "test_water":
                    vals.append(_clean_val(row.get("test_water")))
                elif col == "test_gas":
                    vals.append(_clean_val(row.get("test_gas")))
                elif col == "facility_id":
                    vals.append(
                        str(row["facility_id"]) if row.get("facility_id") else None
                    )
                else:
                    vals.append(None)
            daily_tuples.append(tuple(vals))

        placeholders_prod = ", ".join(["%s"] * len(insert_prod_cols))
        cursor.executemany(
            f"INSERT INTO well_daily_production ({', '.join(insert_prod_cols)}) VALUES ({placeholders_prod})",
            daily_tuples,
        )
        print(f"  Inserted {len(daily_tuples):,} daily production records.")

        # --- Hourly sensor readings: only columns that exist in DB ---
        # NOTE: We add small per-hour noise so the same value is NOT repeated 24 times.
        # Repeating the same value 24h makes rolling 24h std = 0 and "current vs baseline"
        # ratios ~1.0 on normal days, so physics checkers (ENERGY_DIV, FLUID_POUND, etc.)
        # rarely fire. With hourly noise, variance is non-zero and anomalies stand out.
        HOURLY_NOISE_PCT = 0.01  # ±1% per hour so 24h has small variance
        twin_to_db = {}
        for twin_col, db_col in TWIN_TO_SENSOR_COL.items():
            if db_col in insert_sensor_cols:
                twin_to_db[twin_col] = db_col
            elif twin_col == "freq_hz" and "frequency" in insert_sensor_cols:
                twin_to_db[twin_col] = "frequency"

        sensor_rows = []
        for _, row in master_df.iterrows():
            base_ts = pd.Timestamp(row["date"])
            for hour in range(24):
                ts = (base_ts + pd.Timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")
                rec = {col: None for col in insert_sensor_cols}
                if "organization_id" in insert_sensor_cols:
                    rec["organization_id"] = organization_id
                rec["well_id"] = row["asset_id"]
                rec["timestamp"] = ts
                rec["lift_type"] = row["lift_type"]
                for twin_col, db_col in twin_to_db.items():
                    if twin_col in row.index:
                        val = _clean_val(row[twin_col])
                        if val is not None:
                            # Add small hourly variation so 24h is not flat (helps physics checkers)
                            noise = 1.0 + np.random.uniform(
                                -HOURLY_NOISE_PCT, HOURLY_NOISE_PCT
                            )
                            rec[db_col] = round(val * noise, 2)
                if (
                    row["lift_type"] == "Gas Lift"
                    and "injection_pressure" in insert_sensor_cols
                    and rec.get("injection_pressure") is None
                ):
                    rec["injection_pressure"] = _clean_val(
                        row.get("avg_casing_pressure")
                    )
                if "well_status" in insert_sensor_cols and row.get("status"):
                    rec["well_status"] = str(row["status"])
                sensor_rows.append(rec)

        batch_size = 5000
        placeholders = ", ".join(["%s"] * len(insert_sensor_cols))
        for i in range(0, len(sensor_rows), batch_size):
            batch = sensor_rows[i : i + batch_size]
            data_tuples = [tuple(r.get(c) for c in insert_sensor_cols) for r in batch]
            cursor.executemany(
                f"INSERT INTO well_sensor_readings ({', '.join(insert_sensor_cols)}) VALUES ({placeholders})",
                data_tuples,
            )
            print(
                f"  Sensor batch {i // batch_size + 1}: {min(i + batch_size, len(sensor_rows)):,} / {len(sensor_rows):,}"
            )

        # Upsert WELL_STATUS with static metadata (data_quality_group, facility_id, orientation, spud_date)
        # so we don't overwrite live status fields (STATUS, LAST_READING_TIMESTAMP, etc.) when they exist
        well_status_table = f"{db}.{schema}.WELL_STATUS"
        for w in well_meta:
            try:
                cursor.execute(
                    f"""
                    MERGE INTO {well_status_table} AS target
                    USING (SELECT %s AS WELL_ID, %s AS ORGANIZATION_ID, %s AS LIFT_TYPE, %s AS FACILITY_ID,
                                  %s AS DATA_QUALITY_GROUP, %s AS ORIENTATION, %s AS SPUD_DATE) AS source
                    ON target.WELL_ID = source.WELL_ID
                    WHEN MATCHED THEN UPDATE SET
                        target.ORGANIZATION_ID = source.ORGANIZATION_ID,
                        target.FACILITY_ID = source.FACILITY_ID,
                        target.DATA_QUALITY_GROUP = source.DATA_QUALITY_GROUP,
                        target.ORIENTATION = source.ORIENTATION,
                        target.SPUD_DATE = source.SPUD_DATE
                    WHEN NOT MATCHED THEN INSERT (
                        WELL_ID, ORGANIZATION_ID, LIFT_TYPE, FACILITY_ID, DATA_QUALITY_GROUP,
                        ORIENTATION, SPUD_DATE, STATUS
                    ) VALUES (source.WELL_ID, source.ORGANIZATION_ID, source.LIFT_TYPE, source.FACILITY_ID,
                              source.DATA_QUALITY_GROUP, source.ORIENTATION, source.SPUD_DATE, 'UNKNOWN')
                    """,
                    (
                        w["asset_id"],
                        organization_id,
                        w["lift_type"],
                        w.get("facility_id"),
                        w.get("current_status_goal"),
                        w.get("orientation"),
                        w["spud_date"].strftime("%Y-%m-%d") if w.get("spud_date") else None,
                    ),
                )
            except Exception as e:
                print(f"  Warning: WELL_STATUS upsert for {w['asset_id']}: {e}")
        conn.commit()
        print("SUCCESS: Digital twin data uploaded to Snowflake.")
    except Exception as e:
        conn.rollback()
        print(f"ERROR during Snowflake upload: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


# --- 5. MASTER SIMULATION ---
def run_simulation(upload_snowflake=False, upload_s3=False, write_json=True, organization_id=None):
    print("Generating Engineering-Grade Digital Twin (v3.1)...")
    well_meta, fac_meta = build_hierarchy()

    all_dfs = []
    for w in well_meta:
        df = generate_well_timeline(w)
        if not df.empty:
            all_dfs.append(df)
    master_df = pd.concat(all_dfs, ignore_index=True).sort_values(["date", "facility_id"])

    output = {"company": "Apex Permian", "assets": {}}
    for w in well_meta:
        output["assets"][w["asset_id"]] = {
            "type": "Well",
            "lift": w["lift_type"],
            "history": [],
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

            output["assets"][fid]["history"].append(
                {
                    "date": d_str,
                    "oil_prod": round(total_oil, 2),
                    "status": "ESD" if esd else "OK",
                }
            )

            sum_true = day_data["true_oil"].sum()
            for _, row in day_data.iterrows():
                alloc_oil = 0
                if not esd and sum_true > 0:
                    alloc_oil = total_oil * (row["true_oil"] / sum_true)

                def clean(val):
                    return None if pd.isna(val) else round(val, 2)

                output["assets"][row["asset_id"]]["history"].append(
                    {
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
                        "notes": row["notes"],
                    }
                )

    if write_json:
        with open("apex_permian_v3_1_engineering.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print("Done: apex_permian_v3_1_engineering.json")

    org_id = organization_id or os.getenv("ORGANIZATION_ID")
    if upload_s3:
        upload_to_s3(master_df, well_meta, organization_id=org_id)
    if upload_snowflake:
        upload_to_snowflake(master_df, well_meta, organization_id=org_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate digital twin data (JSON, S3 parquet, and/or Snowflake)."
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Upload parquet to S3 under {ORGANIZATION_ID}/sensor_data/; uses S3_BUCKET and ORGANIZATION_ID from .env",
    )
    parser.add_argument(
        "--snowflake",
        action="store_true",
        help="Upload to Snowflake (same tables as generate_sample_data)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write apex_permian_v3_1_engineering.json to current directory (default: do not write)",
    )
    parser.add_argument(
        "--organization-id",
        default=os.getenv("ORGANIZATION_ID"),
        help="Organization/tenant UUID. Default: ORGANIZATION_ID env.",
    )
    args = parser.parse_args()
    if args.organization_id:
        os.environ["ORGANIZATION_ID"] = args.organization_id
    run_simulation(
        upload_snowflake=args.snowflake,
        upload_s3=args.s3,
        write_json=args.json,
        organization_id=args.organization_id,
    )
