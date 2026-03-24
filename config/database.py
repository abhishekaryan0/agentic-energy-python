#!/usr/bin/env python3
"""
Production database setup - creates all tables and seeds config data.
Run once to create database/schema and tables in Snowflake and PostgreSQL.
Uses env: SNOWFLAKE_*, DATABASE_URL
"""

import os
import sys
import logging
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env from project root (same as app and scripts)
_env_dir = Path(__file__).resolve().parent.parent
load_dotenv(_env_dir / ".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "AT_RISK_ASSETS"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "SENSOR_DATA"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
}

DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------------------------
# Snowflake table column definitions (for sync_snowflake_schema + digital twin)
# Each tuple: (column_name_lower, Snowflake type for ALTER ADD COLUMN)
# ---------------------------------------------------------------------------
WELL_DAILY_PRODUCTION_COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("well_id", "VARCHAR(100)"),
    ("date", "DATE"),
    ("oil_volume", "FLOAT"),
    ("gas_volume", "FLOAT"),
    ("water_volume", "FLOAT"),
    ("boe", "FLOAT"),
    ("lift_type", "VARCHAR(50)"),
    ("status", "VARCHAR(50)"),
    ("notes", "VARCHAR(500)"),
    ("test_oil", "FLOAT"),
    ("test_water", "FLOAT"),
    ("test_gas", "FLOAT"),
    ("facility_id", "VARCHAR(100)"),
]

WELL_SENSOR_READINGS_COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("well_id", "VARCHAR(100)"),
    ("timestamp", "TIMESTAMP_NTZ"),
    ("lift_type", "VARCHAR(50)"),
    ("well_status", "VARCHAR(50)"),
    ("strokes_per_minute", "FLOAT"),
    ("torque", "FLOAT"),
    ("polish_rod_load", "FLOAT"),
    ("pump_fillage", "FLOAT"),
    ("surface_stroke_length", "FLOAT"),
    ("downhole_gross_stroke", "FLOAT"),
    ("downhole_net_stroke", "FLOAT"),
    ("runtime", "FLOAT"),
    ("cycles", "FLOAT"),
    ("structural_load", "FLOAT"),
    ("inferred_production", "FLOAT"),
    ("pump_intake_pressure", "FLOAT"),
    ("dry_rod_weight", "FLOAT"),
    ("buoyant_rod_weight", "FLOAT"),
    ("pump_friction", "FLOAT"),
    ("pump_diameter", "FLOAT"),
    ("barrel_length", "FLOAT"),
    ("pump_depth", "FLOAT"),
    ("idle_time_setpoint", "FLOAT"),
    ("motor_temp", "FLOAT"),
    ("motor_current", "FLOAT"),
    ("discharge_pressure", "FLOAT"),
    ("motor_voltage", "FLOAT"),
    ("intake_fluid_temp", "FLOAT"),
    ("vibration_x", "FLOAT"),
    ("vibration_y", "FLOAT"),
    ("discharge_temp", "FLOAT"),
    ("downhole_flow_rate", "FLOAT"),
    ("drive_frequency", "FLOAT"),
    ("tubing_pressure", "FLOAT"),
    ("casing_pressure", "FLOAT"),
    ("drive_input_voltage", "FLOAT"),
    ("output_voltage", "FLOAT"),
    ("input_current", "FLOAT"),
    ("motor_speed", "FLOAT"),
    ("vsd_temp", "FLOAT"),
    ("total_harmonic_distortion", "FLOAT"),
    ("motor_load", "FLOAT"),
    ("set_frequency", "FLOAT"),
    ("acceleration_ramp_time", "FLOAT"),
    ("injection_rate", "FLOAT"),
    ("injection_temperature", "FLOAT"),
    ("bottomhole_pressure", "FLOAT"),
    ("injection_pressure", "FLOAT"),
    ("cycle_time", "FLOAT"),
    ("bottomhole_temp", "FLOAT"),
    ("plunger_arrival_time", "FLOAT"),
    ("plunger_velocity", "FLOAT"),
    ("arrival_count", "FLOAT"),
    ("missed_arrivals", "FLOAT"),
    ("shutin_time", "FLOAT"),
    ("afterflow_time", "FLOAT"),
    ("flow_time", "FLOAT"),
    ("plunger_drop_time", "FLOAT"),
    ("min_shutin_pressure", "FLOAT"),
    ("max_shutin_pressure", "FLOAT"),
    ("open_diff_pressure", "FLOAT"),
    ("well_open", "FLOAT"),
    ("well_close", "FLOAT"),
    ("velocity_limit", "FLOAT"),
    ("wellhead_temp", "FLOAT"),
    ("surface_pressure", "FLOAT"),
]


def setup_snowflake():
    """Create Snowflake database, schema, tables, and seed config data."""
    try:
        from snowflake.connector import connect as snowflake_connect
    except ImportError:
        logger.error(
            "snowflake-connector-python not installed. pip install snowflake-connector-python"
        )
        return False

    required = [
        "user",
        "password",
        "account",
        "warehouse",
        "database",
        "schema",
        "role",
    ]
    missing = [k for k in required if not SNOWFLAKE_CONFIG.get(k)]
    if missing:
        logger.error(f"Missing Snowflake env: {missing}")
        return False

    db = SNOWFLAKE_CONFIG["database"]
    schema = SNOWFLAKE_CONFIG["schema"]
    qual = f"{db}.{schema}"

    try:
        conn = snowflake_connect(
            user=SNOWFLAKE_CONFIG["user"],
            password=SNOWFLAKE_CONFIG["password"],
            account=SNOWFLAKE_CONFIG["account"],
            warehouse=SNOWFLAKE_CONFIG["warehouse"],
            role=SNOWFLAKE_CONFIG["role"],
        )
        cur = conn.cursor()

        cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        logger.info(f"Database {db} created/exists")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {qual}")
        logger.info(f"Schema {qual} created/exists")

        cur.execute(f"USE DATABASE {db}")
        cur.execute(f"USE SCHEMA {schema}")

        tables_sql = [
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.LIFT_TYPES (
                ID INT AUTOINCREMENT PRIMARY KEY,
                LIFT_TYPE_NAME VARCHAR(100) NOT NULL UNIQUE,
                DESCRIPTION VARCHAR(500),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "LIFT_TYPES",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.SENSOR_DEFINITIONS (
                ID INT AUTOINCREMENT PRIMARY KEY,
                LIFT_TYPE_ID INT NOT NULL,
                SENSOR_NAME VARCHAR(200) NOT NULL,
                FIELD_NAME VARCHAR(100) NOT NULL,
                UNIT VARCHAR(50),
                MIN_VALUE FLOAT,
                MAX_VALUE FLOAT,
                IS_FORM_FIELD BOOLEAN DEFAULT FALSE,
                DISPLAY_ORDER INT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "SENSOR_DEFINITIONS",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.ANOMALY_RULES (
                ID INT AUTOINCREMENT PRIMARY KEY,
                LIFT_TYPE_ID INT NOT NULL,
                SENSOR_ID INT NOT NULL,
                RULE_TYPE VARCHAR(50),
                LOWER_BOUND FLOAT,
                UPPER_BOUND FLOAT,
                SEVERITY VARCHAR(50),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "ANOMALY_RULES",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.WELL_DAILY_PRODUCTION (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                DATE DATE NOT NULL,
                OIL_VOLUME FLOAT,
                GAS_VOLUME FLOAT,
                WATER_VOLUME FLOAT,
                BOE FLOAT,
                LIFT_TYPE VARCHAR(50),
                STATUS VARCHAR(50),
                NOTES VARCHAR(500),
                TEST_OIL FLOAT,
                TEST_WATER FLOAT,
                TEST_GAS FLOAT,
                FACILITY_ID VARCHAR(100),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                UNIQUE (WELL_ID, DATE)
            )""",
                "WELL_DAILY_PRODUCTION",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.WELL_SENSOR_READINGS (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                LIFT_TYPE VARCHAR(50),
                WELL_STATUS VARCHAR(50),
                STROKES_PER_MINUTE FLOAT,
                TORQUE FLOAT,
                POLISH_ROD_LOAD FLOAT,
                PUMP_FILLAGE FLOAT,
                SURFACE_STROKE_LENGTH FLOAT,
                DOWNHOLE_GROSS_STROKE FLOAT,
                DOWNHOLE_NET_STROKE FLOAT,
                RUNTIME FLOAT,
                CYCLES FLOAT,
                STRUCTURAL_LOAD FLOAT,
                INFERRED_PRODUCTION FLOAT,
                PUMP_INTAKE_PRESSURE FLOAT,
                DRY_ROD_WEIGHT FLOAT,
                BUOYANT_ROD_WEIGHT FLOAT,
                PUMP_FRICTION FLOAT,
                PUMP_DIAMETER FLOAT,
                BARREL_LENGTH FLOAT,
                PUMP_DEPTH FLOAT,
                IDLE_TIME_SETPOINT FLOAT,
                MOTOR_TEMP FLOAT,
                MOTOR_CURRENT FLOAT,
                DISCHARGE_PRESSURE FLOAT,
                MOTOR_VOLTAGE FLOAT,
                INTAKE_FLUID_TEMP FLOAT,
                VIBRATION_X FLOAT,
                VIBRATION_Y FLOAT,
                DISCHARGE_TEMP FLOAT,
                DOWNHOLE_FLOW_RATE FLOAT,
                DRIVE_FREQUENCY FLOAT,
                TUBING_PRESSURE FLOAT,
                CASING_PRESSURE FLOAT,
                DRIVE_INPUT_VOLTAGE FLOAT,
                OUTPUT_VOLTAGE FLOAT,
                INPUT_CURRENT FLOAT,
                MOTOR_SPEED FLOAT,
                VSD_TEMP FLOAT,
                TOTAL_HARMONIC_DISTORTION FLOAT,
                MOTOR_LOAD FLOAT,
                SET_FREQUENCY FLOAT,
                ACCELERATION_RAMP_TIME FLOAT,
                INJECTION_RATE FLOAT,
                INJECTION_TEMPERATURE FLOAT,
                BOTTOMHOLE_PRESSURE FLOAT,
                INJECTION_PRESSURE FLOAT,
                CYCLE_TIME FLOAT,
                BOTTOMHOLE_TEMP FLOAT,
                PLUNGER_ARRIVAL_TIME FLOAT,
                PLUNGER_VELOCITY FLOAT,
                ARRIVAL_COUNT FLOAT,
                MISSED_ARRIVALS FLOAT,
                SHUTIN_TIME FLOAT,
                AFTERFLOW_TIME FLOAT,
                FLOW_TIME FLOAT,
                PLUNGER_DROP_TIME FLOAT,
                MIN_SHUTIN_PRESSURE FLOAT,
                MAX_SHUTIN_PRESSURE FLOAT,
                OPEN_DIFF_PRESSURE FLOAT,
                WELL_OPEN FLOAT,
                WELL_CLOSE FLOAT,
                VELOCITY_LIMIT FLOAT,
                WELLHEAD_TEMP FLOAT,
                SURFACE_PRESSURE FLOAT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "WELL_SENSOR_READINGS",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.WELL_ANOMALIES (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                ANOMALY_TYPE VARCHAR(500),
                ANOMALY_SCORE FLOAT,
                RAW_VALUES VARIANT,
                MODEL_NAME VARCHAR(100),
                STATUS VARCHAR(50),
                SEVERITY VARCHAR(50),
                CATEGORY VARCHAR(50),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "WELL_ANOMALIES",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.ANOMALY_VIOLATIONS_NEW (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                ANOMALY_ID INT NOT NULL,
                WELL_ID VARCHAR(100) NOT NULL,
                TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                PARAMETER_NAME VARCHAR(100),
                ACTUAL_VALUE FLOAT,
                EXPECTED_MIN FLOAT,
                EXPECTED_MAX FLOAT,
                UNIT VARCHAR(50),
                DEVIATION_PERCENT FLOAT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "ANOMALY_VIOLATIONS_NEW",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.ANOMALY_REVIEW (
                ID VARCHAR(100) PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                EVENT_DATE DATE NOT NULL,
                DETECTED_AT TIMESTAMP_NTZ,
                ANOMALY_CODE VARCHAR(100),
                CATEGORY VARCHAR(100),
                SEVERITY VARCHAR(50),
                TITLE VARCHAR(500),
                UI_TEXT VARIANT,
                IMPACT_VALUE FLOAT,
                IMPACT_METRICS VARIANT,
                CHART_DATA VARIANT,
                STATUS VARCHAR(50),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                UNIQUE (WELL_ID, ANOMALY_CODE, EVENT_DATE)
            )""",
                "ANOMALY_REVIEW",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.OPERATION_RECOMMENDATION (
                ID VARCHAR(100) PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL,
                CATEGORY VARCHAR(100),
                ACTION VARCHAR(500),
                PRIORITY VARCHAR(50),
                STATUS VARCHAR(50),
                REASON VARCHAR(1000),
                EXPECTED_IMPACT VARCHAR(1000),
                CONFIDENCE FLOAT,
                PREDICTION_SOURCE VARCHAR(100),
                PROBABILITY FLOAT,
                OIL_VOLUME_SLOPE FLOAT,
                OIL_VOLUME_ROLL_MEAN FLOAT,
                GAS_VOLUME_SLOPE FLOAT,
                GAS_VOLUME_ROLL_MEAN FLOAT,
                MOTOR_CURRENT_ROLL_MEAN FLOAT,
                MOTOR_TEMP_ROLL_MEAN FLOAT,
                SURFACE_PRESSURE_ROLL_MEAN FLOAT,
                DETAILS VARCHAR(3000),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "OPERATION_RECOMMENDATION",
            ),
            (
                f"""CREATE TABLE IF NOT EXISTS {qual}.WELL_STATUS (
                ID INT AUTOINCREMENT PRIMARY KEY,
                ORGANIZATION_ID VARCHAR(36),
                WELL_ID VARCHAR(100) NOT NULL UNIQUE,
                LIFT_TYPE VARCHAR(50),
                STATUS VARCHAR(50) NOT NULL DEFAULT 'UNKNOWN',
                LAST_READING_TIMESTAMP TIMESTAMP_NTZ,
                LAST_READING_AGE_HOURS FLOAT,
                STROKES_PER_MINUTE FLOAT,
                MOTOR_CURRENT FLOAT,
                INJECTION_RATE FLOAT,
                ANOMALY_COUNT_24H INT DEFAULT 0,
                CRITICAL_ANOMALY_COUNT_24H INT DEFAULT 0,
                STATUS_REASON VARCHAR(500),
                FACILITY_ID VARCHAR(100),
                DATA_QUALITY_GROUP VARCHAR(50),
                ORIENTATION VARCHAR(50),
                SPUD_DATE DATE,
                UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )""",
                "WELL_STATUS",
            ),
        ]

        # Add ORGANIZATION_ID (and WELL_STATUS metadata) to existing tables that were created before these columns existed
        alter_sqls = [
            (
                f"ALTER TABLE {qual}.WELL_DAILY_PRODUCTION ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "WELL_DAILY_PRODUCTION.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.WELL_SENSOR_READINGS ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "WELL_SENSOR_READINGS.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.WELL_ANOMALIES ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "WELL_ANOMALIES.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.ANOMALY_VIOLATIONS_NEW ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "ANOMALY_VIOLATIONS_NEW.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.ANOMALY_REVIEW ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "ANOMALY_REVIEW.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.OPERATION_RECOMMENDATION ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "OPERATION_RECOMMENDATION.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.WELL_STATUS ADD COLUMN ORGANIZATION_ID VARCHAR(36)",
                "WELL_STATUS.ORGANIZATION_ID",
            ),
            (
                f"ALTER TABLE {qual}.WELL_STATUS ADD COLUMN FACILITY_ID VARCHAR(100)",
                "WELL_STATUS.FACILITY_ID",
            ),
            (
                f"ALTER TABLE {qual}.WELL_STATUS ADD COLUMN DATA_QUALITY_GROUP VARCHAR(50)",
                "WELL_STATUS.DATA_QUALITY_GROUP",
            ),
            (
                f"ALTER TABLE {qual}.WELL_STATUS ADD COLUMN ORIENTATION VARCHAR(50)",
                "WELL_STATUS.ORIENTATION",
            ),
            (
                f"ALTER TABLE {qual}.WELL_STATUS ADD COLUMN SPUD_DATE DATE",
                "WELL_STATUS.SPUD_DATE",
            ),
        ]
        for alter_sql, label in alter_sqls:
            try:
                cur.execute(alter_sql)
                logger.info(f"Snowflake: added column {label}")
            except Exception as e:
                if (
                    "already exists" in str(e).lower()
                    or "duplicate column" in str(e).lower()
                ):
                    logger.debug(f"Column {label} already exists, skipping")
                else:
                    logger.warning(f"Snowflake alter {label}: {e}")

        for sql, name in tables_sql:
            try:
                cur.execute(sql)
                logger.info(f"Snowflake: {name} OK")
            except Exception as e:
                logger.warning(f"Snowflake: {name} - {e}")

        conn.commit()

        cur.execute(f"SELECT COUNT(*) FROM {qual}.LIFT_TYPES")
        if cur.fetchone()[0] == 0:
            lift_data = [
                ("Rod Pump", "Sucker rod pumping system"),
                ("ESP", "Electrical Submersible Pump"),
                ("Gas Lift", "Gas lift artificial lift system"),
            ]
            for name, desc in lift_data:
                try:
                    cur.execute(
                        f"INSERT INTO {qual}.LIFT_TYPES (LIFT_TYPE_NAME, DESCRIPTION) VALUES (%s, %s)",
                        (name, desc),
                    )
                except Exception:
                    pass
            logger.info("Seeded LIFT_TYPES")
            conn.commit()

        cur.execute(f"SELECT id, lift_type_name FROM {qual}.LIFT_TYPES")
        lift_map = {row[1]: row[0] for row in cur.fetchall()}
        cur.execute(f"SELECT COUNT(*) FROM {qual}.SENSOR_DEFINITIONS")
        if cur.fetchone()[0] == 0 and lift_map:
            rod = lift_map.get("Rod Pump")
            esp = lift_map.get("ESP")
            gas = lift_map.get("Gas Lift")
            sensors = []
            if rod:
                sensors += [
                    (
                        rod,
                        "Strokes Per Minute",
                        "strokes_per_minute",
                        "SPM",
                        0.5,
                        12.0,
                        True,
                        1,
                    ),
                    (rod, "Torque", "torque", "in-lbs", 100000, 1824000, False, 2),
                    (
                        rod,
                        "Polish Rod Load",
                        "polish_rod_load",
                        "lbs",
                        0,
                        125000,
                        True,
                        3,
                    ),
                    (rod, "Pump Fillage", "pump_fillage", "%", 0, 100, True, 4),
                    (
                        rod,
                        "Surface Stroke Length",
                        "surface_stroke_length",
                        "inches",
                        24,
                        500,
                        False,
                        5,
                    ),
                    (
                        rod,
                        "Downhole Gross Stroke Length",
                        "downhole_gross_stroke",
                        "inches",
                        20,
                        600,
                        False,
                        6,
                    ),
                    (
                        rod,
                        "Downhole Net Stroke Length",
                        "downhole_net_stroke",
                        "inches",
                        20,
                        600,
                        False,
                        7,
                    ),
                    (rod, "Runtime", "runtime", "hours", 0, 24, False, 8),
                    (rod, "Cycles", "cycles", "cycles/day", 0, 100, False, 9),
                    (rod, "Structural Load", "structural_load", "%", 0, 200, False, 10),
                    (
                        rod,
                        "Inferred Production",
                        "inferred_production",
                        "bbls",
                        1,
                        1000,
                        False,
                        11,
                    ),
                    (
                        rod,
                        "Pump Intake Pressure",
                        "pump_intake_pressure",
                        "psi",
                        0,
                        5000,
                        False,
                        12,
                    ),
                    (
                        rod,
                        "Dry Rod Weight",
                        "dry_rod_weight",
                        "lbs",
                        1000,
                        150000,
                        False,
                        13,
                    ),
                    (
                        rod,
                        "Buoyant Rod Weight",
                        "buoyant_rod_weight",
                        "lbs",
                        1000,
                        150000,
                        False,
                        14,
                    ),
                    (rod, "Pump Friction", "pump_friction", "lbs", 0, 1000, False, 15),
                    (
                        rod,
                        "Pump Diameter",
                        "pump_diameter",
                        "inches",
                        1.0,
                        3.25,
                        False,
                        16,
                    ),
                    (rod, "Barrel Length", "barrel_length", "feet", 5, 50, False, 17),
                    (rod, "Pump Depth", "pump_depth", "feet", 500, 12000, False, 18),
                    (
                        rod,
                        "Idle Time Set-point",
                        "idle_time_setpoint",
                        "minutes",
                        0,
                        120,
                        False,
                        19,
                    ),
                ]
            if esp:
                sensors += [
                    (esp, "Motor Temperature", "motor_temp", "°F", 50, 1500, True, 1),
                    (esp, "Motor Current", "motor_current", "A", 0, 200, True, 2),
                    (
                        esp,
                        "Discharge Pressure",
                        "discharge_pressure",
                        "psi",
                        0,
                        5000,
                        True,
                        3,
                    ),
                    (
                        esp,
                        "Pump Intake Pressure",
                        "pump_intake_pressure",
                        "psi",
                        0,
                        5000,
                        False,
                        4,
                    ),
                    (esp, "Motor Voltage", "motor_voltage", "V", 0, 5000, True, 5),
                    (
                        esp,
                        "Intake/Fluid Temperature",
                        "intake_fluid_temp",
                        "°F",
                        50,
                        900,
                        False,
                        6,
                    ),
                    (esp, "Vibration X", "vibration_x", "in/s", 0.0, 1.25, False, 7),
                    (esp, "Vibration Y", "vibration_y", "in/s", 0.0, 1.25, False, 8),
                    (
                        esp,
                        "Discharge Temperature",
                        "discharge_temp",
                        "°F",
                        0,
                        1500,
                        False,
                        9,
                    ),
                    (
                        esp,
                        "Downhole Flow Rate",
                        "downhole_flow_rate",
                        "bpd",
                        0,
                        10000,
                        False,
                        10,
                    ),
                    (
                        esp,
                        "Drive Frequency",
                        "drive_frequency",
                        "Hz",
                        0,
                        100,
                        False,
                        11,
                    ),
                    (
                        esp,
                        "Tubing Pressure",
                        "tubing_pressure",
                        "psi",
                        0,
                        2000,
                        False,
                        12,
                    ),
                    (
                        esp,
                        "Casing Pressure",
                        "casing_pressure",
                        "psi",
                        0,
                        2000,
                        False,
                        13,
                    ),
                    (
                        esp,
                        "Drive Input Voltage",
                        "drive_input_voltage",
                        "V",
                        340,
                        550,
                        False,
                        14,
                    ),
                    (
                        esp,
                        "Output Voltage",
                        "output_voltage",
                        "V",
                        460,
                        4200,
                        False,
                        15,
                    ),
                    (esp, "Input Current", "input_current", "A", 10, 200, False, 16),
                    (esp, "Motor Speed", "motor_speed", "rpm", 0, 7000, False, 17),
                    (esp, "VSD Temperature", "vsd_temp", "°F", 0, 1000, False, 18),
                    (
                        esp,
                        "Total Harmonic Distortion",
                        "total_harmonic_distortion",
                        "%",
                        0,
                        100,
                        False,
                        19,
                    ),
                    (esp, "Motor Load", "motor_load", "%", 0, 150, False, 20),
                    (esp, "Set Frequency", "set_frequency", "Hz", 0, 100, False, 21),
                    (
                        esp,
                        "Acceleration Ramp Time",
                        "acceleration_ramp_time",
                        "seconds",
                        0,
                        60,
                        False,
                        22,
                    ),
                    (esp, "Cycle", "cycle", "cycles", 0, 100, False, 23),
                ]
            if gas:
                sensors += [
                    (gas, "Injection Rate", "injection_rate", "mcf", 0, 2000, True, 1),
                    (
                        gas,
                        "Injection Temperature",
                        "injection_temperature",
                        "°F",
                        0,
                        1000,
                        True,
                        2,
                    ),
                    (
                        gas,
                        "Bottomhole Pressure",
                        "bottomhole_pressure",
                        "psi",
                        50,
                        5000,
                        True,
                        3,
                    ),
                    (
                        gas,
                        "Injection Pressure",
                        "injection_pressure",
                        "psi",
                        100,
                        5000,
                        True,
                        4,
                    ),
                    (gas, "Cycle Time", "cycle_time", "minutes", 5, 180, False, 5),
                    (
                        gas,
                        "Bottomhole Temperature",
                        "bottomhole_temp",
                        "°F",
                        50,
                        500,
                        False,
                        6,
                    ),
                    (
                        gas,
                        "Plunger Arrival Time",
                        "plunger_arrival_time",
                        "minutes",
                        0,
                        45,
                        False,
                        7,
                    ),
                    (
                        gas,
                        "Plunger Velocity",
                        "plunger_velocity",
                        "ft/min",
                        0,
                        2000,
                        False,
                        8,
                    ),
                    (gas, "Arrival Count", "arrival_count", "count", 0, 200, False, 9),
                    (
                        gas,
                        "Missed Arrivals",
                        "missed_arrivals",
                        "count",
                        0,
                        200,
                        False,
                        10,
                    ),
                    (gas, "Shut-in Time", "shutin_time", "minutes", 5, 60, False, 11),
                    (
                        gas,
                        "Afterflow Time",
                        "afterflow_time",
                        "minutes",
                        2,
                        45,
                        False,
                        12,
                    ),
                    (gas, "Flow Time", "flow_time", "minutes", 5, 60, False, 13),
                    (
                        gas,
                        "Plunger Drop Time",
                        "plunger_drop_time",
                        "minutes",
                        1,
                        20,
                        False,
                        14,
                    ),
                    (
                        gas,
                        "Minimum Shut-in Pressure",
                        "min_shutin_pressure",
                        "psi",
                        0,
                        1000,
                        False,
                        15,
                    ),
                    (
                        gas,
                        "Maximum Shut-in Pressure",
                        "max_shutin_pressure",
                        "psi",
                        0,
                        1000,
                        False,
                        16,
                    ),
                    (
                        gas,
                        "Open Differential Pressure",
                        "open_diff_pressure",
                        "psi",
                        50,
                        300,
                        False,
                        17,
                    ),
                    (gas, "Well Open", "well_open", "psi", 100, 500, False, 18),
                    (gas, "Well Close", "well_close", "psi", 20, 100, False, 19),
                    (
                        gas,
                        "Velocity Limit",
                        "velocity_limit",
                        "ft/min",
                        400,
                        2000,
                        False,
                        20,
                    ),
                ]
            for lt_id, sname, fname, unit, mn, mx, is_form, ord_ in sensors:
                try:
                    cur.execute(
                        f"""INSERT INTO {qual}.SENSOR_DEFINITIONS
                        (LIFT_TYPE_ID, SENSOR_NAME, FIELD_NAME, UNIT, MIN_VALUE, MAX_VALUE, IS_FORM_FIELD, DISPLAY_ORDER)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (lt_id, sname, fname, unit, mn, mx, is_form, ord_),
                    )
                except Exception:
                    pass
            conn.commit()
            logger.info("Seeded SENSOR_DEFINITIONS")

        cur.close()
        conn.close()
        logger.info("Snowflake setup complete")
        return True
    except Exception as e:
        logger.error(f"Snowflake setup failed: {e}", exc_info=True)
        return False


def setup_postgres():
    """Create PostgreSQL database if needed, then all tables."""
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set, skipping PostgreSQL")
        return True

    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        logger.error("psycopg2 not installed. pip install psycopg2-binary")
        return False

    parsed = urlparse(DATABASE_URL)
    db_name = parsed.path.lstrip("/").split("?")[0]
    if not db_name:
        logger.warning("Invalid DATABASE_URL, skipping PostgreSQL")
        return True

    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port or 5432,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(
            f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,)
        )
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"PostgreSQL database {db_name} created")
        else:
            logger.info(f"PostgreSQL database {db_name} exists")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"PostgreSQL create database: {e}")

    try:
        from urllib.parse import urlunparse

        clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        conn = psycopg2.connect(clean_url)
        cur = conn.cursor()

        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS operation_suggestion (
                id VARCHAR(50) PRIMARY KEY,
                organization_id UUID,
                well_id VARCHAR(50) NOT NULL,
                category VARCHAR(50) DEFAULT 'Production',
                action VARCHAR(255) NOT NULL,
                status VARCHAR(50) DEFAULT 'New',
                priority VARCHAR(50) DEFAULT 'HIGH',
                confidence_percent FLOAT,
                production_increase_bbl_day FLOAT,
                production_increase_percent FLOAT,
                daily_expense_benefit_usd FLOAT,
                implementation_cost_usd FLOAT,
                net_daily_benefit_usd FLOAT,
                asset_value_increase_usd FLOAT,
                time_reduced_hours FLOAT,
                detailed_analysis TEXT,
                current_performance VARCHAR(255),
                optimal_performance VARCHAR(255),
                conservative_approach TEXT,
                aggressive_optimization TEXT,
                hybrid_model TEXT,
                recommended_model VARCHAR(50),
                citations TEXT,
                referenced_data TEXT,
                supporting_analysis TEXT,
                data_quality VARCHAR(100),
                reason TEXT,
                expected_impact VARCHAR(500),
                metrics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cur.execute(
            "ALTER TABLE operation_suggestion ADD COLUMN IF NOT EXISTS organization_id UUID;"
        )
        logger.info("PostgreSQL: operation_suggestion OK")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS anomaly_suggestions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                organization_id UUID,
                well_id VARCHAR(50) NOT NULL,
                asset_id VARCHAR(50),
                lift_type VARCHAR(50),
                timestamp TIMESTAMP NOT NULL,
                well_status VARCHAR(50),
                alert_title VARCHAR(255),
                severity VARCHAR(50),
                status VARCHAR(50),
                confidence VARCHAR(20),
                description TEXT,
                suggested_actions JSONB,
                explanation TEXT,
                historical_context JSONB,
                risk_analysis JSONB,
                raw_anomaly_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cur.execute(
            "ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS organization_id UUID;"
        )
        cur.execute(
            "ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS well_status VARCHAR(50);"
        )
        cur.execute(
            "ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS asset_id VARCHAR(50);"
        )
        cur.execute(
            "ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS lift_type VARCHAR(50);"
        )
        logger.info("PostgreSQL: anomaly_suggestions OK")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS anomaly_review (
                id VARCHAR(100) PRIMARY KEY,
                organization_id UUID,
                well_id VARCHAR(100) NOT NULL,
                event_date DATE NOT NULL,
                detected_at TIMESTAMP,
                anomaly_code VARCHAR(100),
                category VARCHAR(100),
                severity VARCHAR(50),
                title VARCHAR(500),
                ui_text JSONB,
                impact_value FLOAT,
                impact_metrics JSONB,
                chart_data JSONB,
                mute_till TIMESTAMP,
                open_up BOOLEAN,
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (well_id, anomaly_code, event_date)
            )
        """
        )
        cur.execute(
            "ALTER TABLE anomaly_review ADD COLUMN IF NOT EXISTS organization_id UUID;"
        )
        cur.execute(
            "ALTER TABLE anomaly_review ADD COLUMN IF NOT EXISTS mute_till TIMESTAMP;"
        )
        cur.execute(
            "ALTER TABLE anomaly_review ADD COLUMN IF NOT EXISTS open_up BOOLEAN;"
        )
        logger.info("PostgreSQL: anomaly_review OK")

        conn.commit()
        cur.close()
        conn.close()
        logger.info("PostgreSQL setup complete")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}", exc_info=True)
        return False


def main():
    logger.info("=== Production database setup ===")
    sf_ok = setup_snowflake()
    pg_ok = setup_postgres()
    if sf_ok and pg_ok:
        logger.info("All done.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
