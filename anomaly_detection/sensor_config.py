#!/usr/bin/env python3
"""
Sensor configuration loader - fetches lift types, sensor definitions and rules from Snowflake.
"""

import os
import logging
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Cache for config
_config_cache = {}


def get_db_connection():
    """
    Legacy stub kept for backward compatibility.
    In S3-only mode we no longer connect to Snowflake; callers should avoid using this.
    """
    raise RuntimeError("Snowflake is disabled in this deployment (S3-only mode).")


@lru_cache(maxsize=1)
def _load_sensor_ranges_from_parquet() -> pd.DataFrame:
    """
    Load sensor_ranges from S3 parquet, using either:
    - SENSOR_RANGES_S3_URI env, or
    - s3://S3_BUCKET/ORGANIZATION_ID/sensor_data/sensor_ranges.parquet
    """
    uri = os.getenv("SENSOR_RANGES_S3_URI", "").strip()
    if not uri:
        bucket = os.getenv("S3_BUCKET", "").strip()
        org = os.getenv("ORGANIZATION_ID", "").strip()
        if bucket and org:
            uri = f"s3://{bucket}/{org}/sensor_data/sensor_ranges.parquet"
    if not uri:
        raise RuntimeError("Sensor ranges parquet URI not configured (SENSOR_RANGES_S3_URI or S3_BUCKET/ORGANIZATION_ID).")

    try:
        if uri.startswith("s3://"):
            import boto3
            import io
            from urllib.parse import urlparse

            parsed = urlparse(uri)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            s3 = boto3.client("s3")
            buf = io.BytesIO()
            s3.download_fileobj(bucket, key, buf)
            buf.seek(0)
            df = pd.read_parquet(buf)
        else:
            df = pd.read_parquet(uri)
    except Exception as e:
        from botocore.exceptions import ClientError

        if isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") in (
            "404",
            "NoSuchKey",
        ):
            logger.warning(
                "sensor_ranges parquet not found at %s; rule-based limits will use defaults",
                uri,
            )
            return pd.DataFrame()
        logger.error(f"Failed to load sensor_ranges parquet from {uri}: {e}")
        raise

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    # Backward compat: some parquet use 'lift_type_name' instead of 'lift_type'
    if "lift_type" not in df.columns and "lift_type_name" in df.columns:
        df = df.rename(columns={"lift_type_name": "lift_type"})
    # Expected columns (case-insensitive): lift_type, field_name, min_value, max_value, unit, is_form_field, display_order, sensor_name?
    if "lift_type" not in df.columns or "field_name" not in df.columns:
        raise RuntimeError(
            f"sensor_ranges parquet at {uri} is missing required columns (lift_type, field_name). "
            f"Columns: {list(df.columns)}"
        )

    return df

def get_lift_types():
    """
    Fetch all lift types from sensor_ranges parquet.
    Returns dict {lift_type_name: index_id}.
    """
    if "lift_types" in _config_cache:
        return _config_cache["lift_types"]

    df = _load_sensor_ranges_from_parquet()
    if df.empty:
        lift_types = {"Rod Pump": 1, "ESP": 2, "Gas Lift": 3}
    else:
        lift_types_list = sorted(df["lift_type"].dropna().astype(str).unique().tolist())
        lift_types = {name: i + 1 for i, name in enumerate(lift_types_list)}
    _config_cache["lift_types"] = lift_types
    return lift_types

def get_sensors_for_lift_type(lift_type_name):
    """
    Fetch sensor definitions for a specific lift type.

    Backed by sensor_ranges.parquet in S3. Expects columns:
      - lift_type, field_name, min_value, max_value, unit
      - Optional: sensor_name, is_form_field, display_order
    """
    cache_key = f"sensors_{lift_type_name}"
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    df = _load_sensor_ranges_from_parquet()
    if df.empty:
        sensors = []
        _config_cache[cache_key] = sensors
        return sensors
    sub = df[df["lift_type"].astype(str) == str(lift_type_name)]
    if sub.empty:
        sensors: List[Dict] = []
    else:
        sensors = []
        for idx, row in sub.iterrows():
            sensors.append(
                {
                    "id": idx,
                    "name": row.get("sensor_name") or row.get("field_name"),
                    "field_name": row.get("field_name"),
                    "unit": row.get("unit"),
                    "min": row.get("min_value"),
                    "max": row.get("max_value"),
                    "is_form_field": bool(row.get("is_form_field", True)),
                    "display_order": int(row.get("display_order", idx)),
                }
            )

    _config_cache[cache_key] = sensors
    return sensors

def get_form_sensors(lift_type_name):
    """Get only form fields for a lift type."""
    sensors = get_sensors_for_lift_type(lift_type_name)
    return [s for s in sensors if s['is_form_field']]

def get_sensor_ranges(lift_type_name):
    """Get sensor field names and ranges for a lift type."""
    sensors = get_sensors_for_lift_type(lift_type_name)
    ranges = {}
    for sensor in sensors:
        ranges[sensor['field_name']] = {
            'min': sensor['min'],
            'max': sensor['max'],
            'unit': sensor['unit']
        }
    return ranges

def get_form_fields(lift_type_name):
    """Get form field names for a lift type."""
    sensors = get_form_sensors(lift_type_name)
    return [s['field_name'] for s in sensors]


def get_features_for_lift_type(lift_type_name):
    """Get sensor field names for a lift type (sensors with min/max for training and detection)."""
    sensors = get_sensors_for_lift_type(lift_type_name)
    return [s['field_name'] for s in sensors if s.get('min') is not None and s.get('max') is not None]


def get_all_sensor_columns():
    """Get union of all sensor field names across lift types (for dynamic queries)."""
    seen = set()
    for lt_name in get_lift_types().keys():
        for f in get_features_for_lift_type(lt_name):
            seen.add(f)
    common = ['surface_pressure', 'casing_pressure', 'wellhead_temp', 'tubing_pressure']
    for c in common:
        seen.add(c)
    base = ['well_id', 'timestamp', 'lift_type']
    return base + sorted(seen)


def get_anomaly_rules(lift_type_name):
    """Fetch anomaly rules for a lift type."""
    cache_key = f'rules_{lift_type_name}'
    if cache_key in _config_cache:
        return _config_cache[cache_key]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ar.sensor_id, sd.field_name, ar.lower_bound, ar.upper_bound, ar.severity
        FROM anomaly_rules ar
        JOIN sensor_definitions sd ON ar.sensor_id = sd.id
        JOIN lift_types lt ON ar.lift_type_id = lt.id
        WHERE lt.lift_type_name = %s
    """, (lift_type_name,))
    
    rules = {}
    for row in cursor.fetchall():
        field_name = row[1]
        rules[field_name] = {
            'lower': row[2],
            'upper': row[3],
            'severity': row[4]
        }
    
    conn.close()
    _config_cache[cache_key] = rules
    return rules

def clear_cache():
    """Clear configuration cache."""
    global _config_cache
    _config_cache = {}

def get_all_sensor_info():
    """Get all sensor info for all lift types."""
    lift_types = get_lift_types()
    all_info = {}
    
    for lift_type_name in lift_types.keys():
        all_info[lift_type_name] = {
            'sensors': get_sensors_for_lift_type(lift_type_name),
            'form_fields': get_form_fields(lift_type_name),
            'ranges': get_sensor_ranges(lift_type_name),
            'rules': get_anomaly_rules(lift_type_name)
        }
    
    return all_info

# Test
if __name__ == "__main__":
    print("=== Lift Types ===")
    lift_types = get_lift_types()
    print(lift_types)
    
    print("\n=== Rod Pump Sensors ===")
    sensors = get_sensors_for_lift_type("Rod Pump")
    for s in sensors:
        print(f"  {s['name']}: {s['field_name']} ({s['unit']}) - Form: {s['is_form_field']}")
    
    print("\n=== Rod Pump Form Fields ===")
    form_fields = get_form_fields("Rod Pump")
    print(form_fields)
    
    print("\n=== Rod Pump Ranges ===")
    ranges = get_sensor_ranges("Rod Pump")
    print(ranges)
