import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────────────
#                        CONFIGURATION
# ───────────────────────────────────────────────────────────────

# How many last years of data you want to generate
GENERATE_LAST_YEARS = 2  # ← Change this value

SAMPLE_MODE = False  # True = very small dataset for testing

if SAMPLE_MODE:
    GENERATE_LAST_YEARS = 0.08  # ~1 month for quick testing

# Snowflake connection settings (loaded from .env)
# Explicitly load .env from the current script directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
}

# Debug: Print loaded (masked) config to verify
print("Debug: Loaded Snowflake Config:")
for k, v in SNOWFLAKE_CONFIG.items():
    if k == "password":
        print(f"  {k}: {'*' * 8 if v else 'None'}")
    else:
        print(f"  {k}: {v}")
print("-" * 30)

# ───────────────────────────────────────────────────────────────
#                       DATE RANGE SETUP
# ───────────────────────────────────────────────────────────────

DAYS_PER_YEAR = 365.25
TOTAL_DAYS = int(GENERATE_LAST_YEARS * DAYS_PER_YEAR)

END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=TOTAL_DAYS)

dates_daily = pd.date_range(start=START_DATE, end=END_DATE, freq="D", inclusive="both")[
    :TOTAL_DAYS
]

TOTAL_HOURS = len(dates_daily) * 24
dates_hf = pd.date_range(start=START_DATE, end=END_DATE, freq="h", inclusive="both")[
    :TOTAL_HOURS
]

print(f"Data period: {dates_daily[0].date()}  →  {dates_daily[-1].date()}")
print(f"Daily records per well:   {len(dates_daily):,d}")
print(f"Hourly records per well:  {len(dates_hf):,d}\n")

# ───────────────────────────────────────────────────────────────
#                  LOAD SENSOR CONFIG FROM DATABASE
# ───────────────────────────────────────────────────────────────


def _get_sensor_ranges_from_db():
    """Load sensor ranges from sensor_definitions (Snowflake)."""
    try:
        from sensor_config import get_sensor_ranges, get_lift_types

        ranges_by_type = {}
        for lt_name in get_lift_types().keys():
            ranges_by_type[lt_name] = get_sensor_ranges(lt_name)
        return ranges_by_type
    except Exception as e:
        print(
            f"WARNING: Could not load from sensor_config: {e}. Using fallback ranges."
        )
        return {
            "Rod Pump": {
                "strokes_per_minute": {"min": 0.5, "max": 12},
                "torque": {"min": 100000, "max": 1824000},
                "polish_rod_load": {"min": 0, "max": 125000},
                "pump_fillage": {"min": 0, "max": 100},
                "tubing_pressure": {"min": 0, "max": 3000},
            },
            "ESP": {
                "motor_temp": {"min": 50, "max": 1500},
                "motor_current": {"min": 0, "max": 200},
                "discharge_pressure": {"min": 0, "max": 5000},
                "pump_intake_pressure": {"min": 0, "max": 5000},
                "motor_voltage": {"min": 0, "max": 5000},
            },
            "Gas Lift": {
                "injection_rate": {"min": 0, "max": 2000},
                "injection_temperature": {"min": 0, "max": 1000},
                "bottomhole_pressure": {"min": 50, "max": 5000},
                "injection_pressure": {"min": 100, "max": 5000},
                "cycle_time": {"min": 5, "max": 180},
            },
        }


def _field_to_pascal(col):
    """Convert field_name (snake_case) to PascalCase for DataFrame column."""
    return "".join(w.capitalize() for w in col.split("_"))


SENSOR_RANGES = _get_sensor_ranges_from_db()
print("Loaded sensor ranges from sensor_definitions (database)")

# ───────────────────────────────────────────────────────────────
#                    DECLINE CURVE FUNCTION
# ───────────────────────────────────────────────────────────────


def generate_decline(days, qi_oil, qi_gas, qi_water):
    Di_annual = 0.35
    b = 0.75
    D_term_annual = 0.06
    t_switch = 365

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


# ───────────────────────────────────────────────────────────────
#                       WELL TYPES (ONLY 3)
# ───────────────────────────────────────────────────────────────

well_types = [
    (15, "Rod Pump", (300, 600, 400)),
    (20, "ESP", (1200, 2000, 1800)),
    (15, "Gas Lift", (800, 1500, 1200)),
]

# ───────────────────────────────────────────────────────────────
#                        DATA GENERATION
# ───────────────────────────────────────────────────────────────

daily_data = []
hf_data = []

count = 1
print("Generating wells...\n")

for count_limit, w_type, initial_rates in well_types:
    wells_to_create = count_limit if not SAMPLE_MODE else min(2, count_limit)

    for _ in range(wells_to_create):
        w_id = f"Well_{count:03d}_{w_type.replace(' ','')}"
        qi_o, qi_g, qi_w = initial_rates

        # Daily production
        qo, qg, qw = generate_decline(len(dates_daily), qi_o, qi_g, qi_w)
        qo *= np.random.normal(1, 0.05, len(qo))
        qg *= np.random.normal(1, 0.05, len(qg))
        qw *= np.random.normal(1, 0.05, len(qw))

        daily_df = pd.DataFrame(
            {
                "WellID": w_id,
                "Date": dates_daily,
                "Oil_Volume": qo,
                "Gas_Volume": qg,
                "Water_Volume": qw,
                "Lift_Type": w_type,
            }
        )
        daily_data.append(daily_df)

        # High frequency sensors - use ranges from sensor_definitions (DB)
        w_hf = pd.DataFrame({"WellID": w_id, "Timestamp": dates_hf})
        ranges = SENSOR_RANGES.get(w_type, {})
        for field_name, rng in ranges.items():
            mn = float(rng.get("min", 0))
            mx = float(rng.get("max", 1))
            if mn >= mx:
                mx = mn + 1
            mean = (mn + mx) / 2
            std = (mx - mn) / 6
            col_pascal = _field_to_pascal(field_name)
            w_hf[col_pascal] = np.clip(
                np.random.normal(mean, max(std, 0.01), len(dates_hf)), mn, mx
            )

        # Add common sensors if not already from ranges
        common = [
            ("surface_pressure", 500, 150, 0, 2000),
            ("casing_pressure", 600, 200, 0, 2500),
            ("wellhead_temp", 40, 10, 20, 80),
            ("tubing_pressure", 800, 100, 0, 3000),
        ]
        for field, mean, std, minv, maxv in common:
            if field not in ranges and _field_to_pascal(field) not in w_hf.columns:
                w_hf[_field_to_pascal(field)] = np.clip(
                    np.random.normal(mean, std, len(dates_hf)), minv, maxv
                )

        hf_data.append(w_hf)

        print(
            f"  Generated {w_id}   ({len(dates_daily):,} daily / {len(dates_hf):,} hourly)"
        )
        count += 1

# ───────────────────────────────────────────────────────────────
#                        SAVE TO SNOWFLAKE
# ───────────────────────────────────────────────────────────────

if not daily_data or not hf_data:
    print("No data was generated. Exiting.")
    exit()

print("\nConnecting to Snowflake...")

try:
    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    print("Clearing previous data...")
    cursor.execute("DELETE FROM well_daily_production")
    cursor.execute("DELETE FROM well_sensor_readings")

    # Daily data insert
    print("Inserting daily production...")
    df_daily = pd.concat(daily_data, ignore_index=True)
    daily_tuples = [
        (
            row["WellID"],
            row["Date"].strftime("%Y-%m-%d"),
            float(row["Oil_Volume"]),
            float(row["Gas_Volume"]),
            float(row["Water_Volume"]),
            row["Lift_Type"],
        )
        for _, row in df_daily.iterrows()
    ]

    cursor.executemany(
        """
        INSERT INTO well_daily_production
        (well_id, date, oil_volume, gas_volume, water_volume, lift_type)
        VALUES (%s, %s, %s, %s, %s, %s)
    """,
        daily_tuples,
    )

    print(f"→ Inserted {len(daily_tuples):,} daily records")

    # High frequency data insert
    print("Inserting high-frequency sensor data...")
    df_hf = pd.concat(hf_data, ignore_index=True)
    df_hf["Timestamp"] = df_hf["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    def _pascal_to_snake(s):
        return (
            "".join("_" + c.lower() if c.isupper() else c for c in s)
            .lstrip("_")
            .replace("__", "_")
        )

    column_mapping = {"WellID": "well_id", "Timestamp": "timestamp"}
    for c in df_hf.columns:
        if c not in column_mapping:
            column_mapping[c] = _pascal_to_snake(c).replace(" ", "_")

    df_hf = df_hf.rename(columns=column_mapping)

    df_hf["lift_type"] = df_hf["well_id"].apply(
        lambda x: (
            "Rod Pump"
            if "RodPump" in x
            else "ESP" if "ESP" in x else "Gas Lift" if "GasLift" in x else "Unknown"
        )
    )

    from sensor_config import get_all_sensor_columns

    try:
        all_columns = get_all_sensor_columns()
    except Exception:
        all_columns = [
            "well_id",
            "timestamp",
            "lift_type",
            "strokes_per_minute",
            "torque",
            "polish_rod_load",
            "pump_fillage",
            "tubing_pressure",
            "motor_temp",
            "motor_current",
            "discharge_pressure",
            "pump_intake_pressure",
            "motor_voltage",
            "injection_rate",
            "injection_temperature",
            "bottomhole_pressure",
            "injection_pressure",
            "cycle_time",
            "surface_pressure",
            "casing_pressure",
            "wellhead_temp",
        ]

    for col in all_columns:
        if col not in df_hf.columns:
            df_hf[col] = None

    df_hf = df_hf[all_columns].replace({np.nan: None, np.inf: None, -np.inf: None})

    batch_size = 5000
    total = len(df_hf)
    placeholders = ", ".join(["%s"] * len(all_columns))

    for i in range(0, total, batch_size):
        batch = df_hf.iloc[i : i + batch_size]
        data_tuples = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        cursor.executemany(
            f"""
            INSERT INTO well_sensor_readings ({', '.join(all_columns)})
            VALUES ({placeholders})
        """,
            data_tuples,
        )
        print(f"  → Batch {i//batch_size + 1}  ({i + len(batch):,} / {total:,})")

    conn.commit()
    print("\nSUCCESS: Data generation and upload completed!")

except Exception as e:
    print(f"\nERROR during Snowflake operation:\n{str(e)}")
    raise

finally:
    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
