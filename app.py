from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional, List
import logging
import os
from uuid import uuid4
import random
import openai
import re

# Custom JSON Encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return super().default(obj)

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

try:
    from snowflake.connector import connect as snowflake_connect
except Exception:
    snowflake_connect = None
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

from operations.suggestions import (
    create_operation_suggestion_table,
    save_operation_suggestion,
    get_operation_suggestions,
    update_operation_suggestion_status,
    get_operation_suggestion_detail,
    get_pg_connection,
    stream_run_operational_pipeline
)

try:
    from operations.recommendations import generate_recommendations_for_well
except ImportError:
    generate_recommendations_for_well = None

try:
    from anomaly_detection.detector import get_detector, check_anomaly as detector_check_anomaly
    ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTOR_AVAILABLE = False
    logging.warning("anomaly_detector module not available")

try:
    from config.sensor_config import get_lift_types, get_form_sensors, get_sensor_ranges, get_form_fields
    DYNAMIC_CONFIG_AVAILABLE = True
except ImportError:
    DYNAMIC_CONFIG_AVAILABLE = False
    logging.warning("sensor_config module not available")

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
APP_URL = os.getenv("APP_URL", "http://localhost:5000")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY is missing in .env")

client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_API_BASE"),
    default_headers={
        "HTTP-Referer": APP_URL,
        "X-Title": "Well Anomaly Detection",
    }
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

REQUIRED_SNOWFLAKE_KEYS = ['user', 'password', 'account', 'warehouse', 'database', 'schema', 'role']
missing_sf = [k for k in REQUIRED_SNOWFLAKE_KEYS if not SNOWFLAKE_CONFIG.get(k)]
if missing_sf:
    logging.warning(f"Missing Snowflake configuration keys: {missing_sf}")

if ANOMALY_DETECTOR_AVAILABLE:
    anomaly_detector = get_detector()
    logging.info("✓ Unified anomaly detector initialized")
else:
    raise RuntimeError("Anomaly detector unavailable")

def get_db_connection():
    try:
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to Snowflake: {e}")
        raise

def get_historical_anomalies(limit=200):
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            well_id, 
            timestamp, 
            category, 
            severity, 
            raw_values, 
            anomaly_score, 
            anomaly_type as violation_summary 
        FROM well_anomalies
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        cursor = conn.cursor()
        try:
            cursor.execute(query, [limit])
            rows = cursor.fetchall()
            cols = [c[0].lower() for c in cursor.description] if cursor.description else []
            df = pd.DataFrame(rows, columns=cols)
        finally:
            cursor.close()
            conn.close()
        return df
    except Exception as e:
        logging.warning(f"Could not fetch historical anomalies: {e}")
        return pd.DataFrame()

def find_similar_anomalies(current_readings, current_violations, historical_df=None, current_well_id=None, lift_type=None, similarity_threshold=None, max_deviation_percent=20.0, exclude_anomaly_id=None):
    """
    Find similar anomalies.
    Restricted to ONLY check the SAME well ID.
    """
    if not current_violations:
        return []

    violation_criteria = []
    current_violation_keys = set()
    
    for v in current_violations:
        field = v.get('field')
        val = v.get('value')
        if field and val is not None:
            current_violation_keys.add(field)
            try:
                val_float = float(val)
                pct = max_deviation_percent / 100.0
                min_bound = val_float * (1.0 - pct)
                max_bound = val_float * (1.0 + pct)
                violation_criteria.append({
                    'field': field,
                    'min': min(min_bound, max_bound),
                    'max': max(min_bound, max_bound)
                })
            except:
                continue

    if not violation_criteria:
        return []

    conn = get_db_connection()

    # --- INTERNAL HELPER ---
    def fetch_candidates(is_same_well):
        cursor = conn.cursor()
        try:
            where_clauses = []
            params = []
            
            for c in violation_criteria:
                where_clauses.append("(v2.parameter_name = %s AND v2.actual_value BETWEEN %s AND %s)")
                params.extend([c['field'], c['min'], c['max']])
            
            where_range_sql = " OR ".join(where_clauses)
            
            # Logic: Strictly enforce same well ID
            well_logic_sql = "a2.well_id = %s"
            params.append(current_well_id)

            if exclude_anomaly_id is not None:
                well_logic_sql += " AND a2.ID != %s"
                params.append(exclude_anomaly_id)

            # Order by Time DESC from DB = Priority #1
            query = f"""
            SELECT
                a.well_id,
                a.timestamp,
                a.anomaly_type,
                a.severity,
                a.raw_values,
                v.parameter_name,
                v.actual_value,
                v.deviation_percent,
                v.unit
            FROM anomaly_violations_new v
            JOIN well_anomalies a ON v.anomaly_id = a.ID
            WHERE a.ID IN (
                SELECT DISTINCT a2.ID
                FROM anomaly_violations_new v2
                JOIN well_anomalies a2 ON v2.anomaly_id = a2.ID
                WHERE ({where_range_sql})
                AND {well_logic_sql}
            )
            ORDER BY a.timestamp DESC
            LIMIT 50
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            grouped = {}
            for r in rows:
                key = f"{r[0]}_{str(r[1])}"
                if key not in grouped:
                    raw_val_str = r[4]
                    try: full_readings = json.loads(raw_val_str) if raw_val_str else {}
                    except: full_readings = {}

                    grouped[key] = {
                        "well_id": r[0],
                        "timestamp": r[1],
                        "alert_title": r[2],
                        "severity": r[3],
                        "raw_anomaly_data": {
                            "full_readings": full_readings,
                            "violations": []
                        }
                    }
                
                unit_str = r[8] if r[8] else ""
                clean_violation_str = f"{r[6]} {unit_str}".strip()

                grouped[key]["raw_anomaly_data"]["violations"].append({
                    "field": r[5],
                    "violation": clean_violation_str,
                    "raw_value": r[6]
                })
            
            valid_records = []
            for record in grouped.values():
                hist_violations = record['raw_anomaly_data']['violations']
                # hist_all_keys = set(hv['field'] for hv in hist_violations)
                
                matched_keys_in_this_record = set()

                for hv in hist_violations:
                    h_field = hv['field']
                    try:
                        h_val = float(hv['raw_value'])
                        for crit in violation_criteria:
                            if crit['field'] == h_field:
                                if crit['min'] <= h_val <= crit['max']:
                                    matched_keys_in_this_record.add(h_field)
                    except:
                        continue

                # Same Well: Relaxed (Keep if ANY matched)
                if len(matched_keys_in_this_record) > 0:
                    valid_records.append(record)
            
            return valid_records

        except Exception as e:
            logging.error(f"Error in fetch_candidates: {e}")
            return []
        finally:
            cursor.close()

    try:
        # ONLY check the same well
        final_records = fetch_candidates(is_same_well=True)
        
        # Cleanup
        for r in final_records:
            for v in r['raw_anomaly_data']['violations']:
                if 'raw_value' in v: del v['raw_value']

        return final_records[:3]

    except Exception as e:
        logging.error(f"Top Level Error in Similarity Search: {e}")
        return []
    finally:
        conn.close()
        
def init_postgres():
    create_operation_suggestion_table()
    conn = get_pg_connection()
    if not conn:
        return
    cur = None
    try:
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'anomaly_suggestions');")
        table_exists = cur.fetchone()[0]

        if not table_exists:
            cur.execute("""
                CREATE TABLE anomaly_suggestions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    well_id VARCHAR(50) NOT NULL,
                    asset_id VARCHAR(50),
                    lift_type VARCHAR(50),
                    timestamp TIMESTAMP NOT NULL,
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
                );
            """)
        else:
            try:
                cur.execute("ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS asset_id VARCHAR(50);")
                cur.execute("ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS lift_type VARCHAR(50);")
            except Exception:
                pass
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

init_postgres()

def _get_sensor_field_list(lift_type):
    """Return list of sensor field names for the given lift type (for building readings dict)."""
    if lift_type == 'Rod Pump':
        return ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure']
    if lift_type == 'ESP':
        return ['motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage']
    if lift_type == 'Gas Lift':
        return ['injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time']
    return ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure']


def _readings_from_row(row, lift_type):
    """Build readings dict from a well_sensor_readings row (dict with lowercase keys)."""
    field_list = _get_sensor_field_list(lift_type)
    readings = {}
    for field in field_list:
        value = row.get(field)
        readings[field] = float(value) if value is not None else None
    return readings


def _record_anomaly_and_suggestion(conn, well_id, timestamp_str, lift_type, readings, anomaly_details, well_status):
    """
    Insert anomaly into well_anomalies + anomaly_violations_new, then generate AI suggestion
    and store in PostgreSQL anomaly_suggestions. Same flow as after inserting a reading.
    Returns suggestion_result dict or None.
    """
    def convert_for_db(val):
        if isinstance(val, (np.integer, int)): return int(val)
        elif isinstance(val, (np.floating, float)): return float(val)
        elif isinstance(val, (np.bool_, bool)): return bool(val)
        elif isinstance(val, np.ndarray): return val.tolist()
        return val

    cursor = conn.cursor()
    try:
        try:
            final_severity = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
        except Exception:
            final_severity = 'MEDIUM'

        violation_summary = "; ".join([v['violation'] for v in anomaly_details.get('violations', [])])
        raw_values = json.dumps(readings, cls=CustomJSONEncoder, default=str)

        anomaly_params = [
            well_id, timestamp_str, violation_summary,
            float(anomaly_details.get('anomaly_score', 0)), raw_values,
            'Rule_Based_Frontend', 'New', final_severity, lift_type
        ]
        cursor.execute(
            """
            INSERT INTO well_anomalies
            (well_id, timestamp, anomaly_type, anomaly_score, raw_values, model_name, status, severity, category)
            SELECT %s, %s, %s, %s, TRY_PARSE_JSON(%s), %s, %s, %s, %s
            """,
            anomaly_params
        )
        cursor.execute(
            "SELECT ID FROM well_anomalies WHERE well_id = %s AND timestamp = %s ORDER BY ID DESC LIMIT 1",
            (well_id, timestamp_str)
        )
        row_id = cursor.fetchone()
        current_anomaly_id = row_id[0] if row_id else None

        if current_anomaly_id and anomaly_details.get('violations'):
            query_vio = """
            INSERT INTO anomaly_violations_new
            (anomaly_id, well_id, timestamp, parameter_name, actual_value, expected_min, expected_max, unit, deviation_percent)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            for v in anomaly_details['violations']:
                dev_pct = v.get('deviation_pct', 0.0)
                e_min = v.get('min') if v.get('min') is not None else 0.0
                e_max = v.get('max') if v.get('max') is not None else 0.0
                params_vio = (
                    current_anomaly_id, well_id, timestamp_str,
                    v.get('field'), v.get('value'), e_min, e_max,
                    v.get('unit', ''), dev_pct
                )
                cursor.execute(query_vio, params_vio)

        conn.commit()

        recent_anomalies_for_prompt = find_similar_anomalies(
            current_readings=readings,
            current_violations=anomaly_details.get('violations', []),
            current_well_id=well_id,
            lift_type=lift_type,
            max_deviation_percent=20.0,
            exclude_anomaly_id=current_anomaly_id
        )

        augmented_anomaly_details = anomaly_details.copy()
        if (not augmented_anomaly_details.get('violations')) and lift_type and DYNAMIC_CONFIG_AVAILABLE:
            try:
                sensor_ranges = get_sensor_ranges(lift_type)
                synth_violations = []
                for field, val in readings.items():
                    if val is None: continue
                    if field in sensor_ranges:
                        sr = sensor_ranges[field]
                        mn, mx = sr.get('min'), sr.get('max')
                        unit = sr.get('unit') or ""
                        if mn is not None and mx is not None and (val < mn or val > mx):
                            rng = mx - mn if (mx - mn) != 0 else 1
                            deviation = ((mn - val) / rng) * 100 if val < mn else ((val - mx) / rng) * 100
                            synth_violations.append({
                                'field': field, 'value': val, 'min': mn, 'max': mx, 'unit': unit,
                                'deviation_pct': round(deviation, 2),
                                'violation': f"Out of range. Expected {mn}-{mx} {unit}, got {val}"
                            })
                if synth_violations:
                    augmented_anomaly_details['violations'] = synth_violations
            except Exception:
                pass

        suggestion_result = generate_suggestion(
            augmented_anomaly_details, well_id, readings, timestamp_str,
            historical_anomalies=recent_anomalies_for_prompt, lift_type=lift_type
        )
        if not suggestion_result:
            suggestion_result = {}
        suggestion_result['asset_id'] = suggestion_result.get('asset_id') or f"ALT-{random.randint(100000, 999999)}"

        try:
            suggestion_result['severity'] = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
            suggestion_result['confidence'] = f"{calculate_data_driven_confidence(anomaly_details)}%"
        except Exception as conf_err:
            logging.warning(f"Error calculating confidence: {conf_err}")

        pg_conn_insert = get_pg_connection()
        if pg_conn_insert:
            try:
                with pg_conn_insert.cursor() as cur:
                    formatted_anomaly_details = format_anomaly_details_for_storage(anomaly_details)
                    insert_query = """
                        INSERT INTO anomaly_suggestions
                        (id, well_id, asset_id, lift_type, timestamp, well_status, alert_title, severity, status, confidence, description, suggested_actions, explanation, historical_context, risk_analysis, raw_anomaly_data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    insert_values = (
                        str(uuid4()), well_id, suggestion_result.get('asset_id'), lift_type, timestamp_str, well_status,
                        suggestion_result.get('alert_title'), suggestion_result.get('severity'),
                        suggestion_result.get('status'), suggestion_result.get('confidence'),
                        suggestion_result.get('description'),
                        json.dumps(suggestion_result.get('suggested_actions'), cls=CustomJSONEncoder, default=str),
                        suggestion_result.get('explanation'),
                        json.dumps(suggestion_result.get('historical_context', {}), cls=CustomJSONEncoder, default=str),
                        json.dumps(suggestion_result.get('risk_analysis', {}), cls=CustomJSONEncoder, default=str),
                        json.dumps(formatted_anomaly_details, cls=CustomJSONEncoder, default=str)
                    )
                    cur.execute(insert_query, insert_values)
                pg_conn_insert.commit()
            except Exception as pg_e:
                logging.error(f"Error storing suggestion: {pg_e}")
                pg_conn_insert.rollback()
            finally:
                pg_conn_insert.close()

        return suggestion_result
    finally:
        cursor.close()


def check_anomaly(readings, well_id=None, lift_type=None):
    return anomaly_detector.check_anomaly(readings, historical_data=None, well_id=well_id, lift_type=lift_type)

def calculate_severity(anomaly_details, anomaly_score, readings):
    violations = anomaly_details.get('violations', []) or []
    num_violations = len(violations)
    deviations = []
    for v in violations:
        try:
            value = v.get('value')
            min_val = v.get('min')
            max_val = v.get('max')
            if value is None or min_val is None or max_val is None:
                continue
            fv = float(value)
            fmin = float(min_val)
            fmax = float(max_val)
            rng = abs(fmax - fmin) if (fmax - fmin) != 0 else 1.0
            if fv < fmin:
                deviation = abs(fmin - fv) / rng * 100.0
            else:
                deviation = abs(fv - fmax) / rng * 100.0
            deviations.append(deviation)
        except Exception:
            continue
    avg_deviation = float(sum(deviations) / len(deviations)) if deviations else 0.0
    a_score = float(anomaly_score or 0.0)

    if (num_violations >= 3 and (a_score >= 0.85 or avg_deviation > 60)) or a_score >= 0.98:
        return "CRITICAL"
    if num_violations == 2:
        if a_score >= 0.60 or avg_deviation > 40: return "HIGH"
        if a_score >= 0.35 or avg_deviation > 20: return "MEDIUM"
        return "LOW"
    if num_violations == 1:
        if a_score >= 0.50 or avg_deviation > 25: return "HIGH"
        if a_score >= 0.35 or avg_deviation > 12: return "MEDIUM"
        return "LOW"
    if a_score >= 0.75: return "HIGH"
    if a_score >= 0.35: return "MEDIUM"
    return "LOW"

# Well status is now managed by well_status_service.py
# Import the status function for real-time reading inserts
try:
    from well_status.service import get_status_for_reading
    WELL_STATUS_SERVICE_AVAILABLE = True
except ImportError:
    WELL_STATUS_SERVICE_AVAILABLE = False
    logging.warning("well_status_service not available, using default status")
    
    def get_status_for_reading(well_id: str, lift_type: str, readings: Dict) -> str:
        """Fallback status function when service is not available."""
        return 'RUNNING'

def format_number_with_commas(num) -> str:
    """Format a number with comma delimiters."""
    if num is None:
        return None
    try:
        num_float = float(num)
        if num_float == int(num_float):
            return f"{int(num_float):,}"
        else:
            return f"{num_float:,.2f}"
    except (ValueError, TypeError):
        return str(num)

def format_violations_for_display(violations: List) -> List:
    """Format violation numbers with comma delimiters for better readability in descriptions."""
    if not violations:
        return violations
    
    formatted = []
    for v in violations:
        formatted_v = v.copy()
        
        # Format numeric values with commas
        for key in ['value', 'min', 'max']:
            if key in formatted_v and formatted_v[key] is not None:
                formatted_v[key] = format_number_with_commas(formatted_v[key])
        
        formatted.append(formatted_v)
    
    return formatted

def format_anomaly_details_for_storage(anomaly_details: Dict) -> Dict:
    """
    Format anomaly_details for storage with comma-delimited numbers in violations.
    This ensures raw_anomaly_data has formatted values for value, min, max.
    """
    if not anomaly_details:
        return anomaly_details
    
    formatted = anomaly_details.copy()
    
    # Format violations
    if 'violations' in formatted and formatted['violations']:
        formatted_violations = []
        for v in formatted['violations']:
            formatted_v = v.copy()
            # Format value, min, max with commas
            for key in ['value', 'min', 'max']:
                if key in formatted_v and formatted_v[key] is not None:
                    formatted_v[key] = format_number_with_commas(formatted_v[key])
            formatted_violations.append(formatted_v)
        formatted['violations'] = formatted_violations
    
    return formatted

def generate_alert_title_from_violations(violations: List) -> str:
    """Generate a proper alert title from the violations list."""
    if not violations or len(violations) == 0:
        return "Sensor Parameter Anomaly"
    
    # Sort by deviation to get the most critical violation
    sorted_violations = sorted(violations, key=lambda x: float(x.get('deviation_pct', 0) or 0), reverse=True)
    top_violation = sorted_violations[0]
    field = top_violation.get('field', 'parameter')
    value = top_violation.get('value')
    min_val = top_violation.get('min')
    max_val = top_violation.get('max')
    
    # Convert field name to readable format
    field_name = format_field_name(field)
    
    # Determine issue type based on value vs range
    issue_type = "Anomaly"
    try:
        if value is not None and min_val is not None and max_val is not None:
            val = float(value)
            mn = float(min_val)
            mx = float(max_val)
            if val > mx:
                issue_type = "High" if val < mx * 1.5 else "Spike"
            elif val < mn:
                issue_type = "Low" if val > mn * 0.5 else "Drop"
    except (ValueError, TypeError):
        pass
    
    return f"{field_name} {issue_type}"

def sanitize_alert_title(title: str, violations: List) -> str:
    """
    Remove any lift type names from alert_title and ensure it's based on violation parameter.
    This is a post-processing safety net in case OpenAI doesn't follow instructions.
    """
    # List of forbidden terms (lift types and generic terms)
    forbidden_terms = ['esp', 'rod pump', 'gas lift', 'rod_pump', 'gas_lift']
    
    if not title:
        return generate_alert_title_from_violations(violations)
    
    title_lower = title.lower()
    
    # Check if ANY forbidden term is in the title
    if any(term in title_lower for term in forbidden_terms):
        # Regenerate completely from violations - don't trust the AI output
        return generate_alert_title_from_violations(violations)
    
    # Also check for generic unhelpful titles
    generic_titles = ['anomaly detected', 'alert detected', 'sensor anomaly detected']
    if title_lower.strip() in generic_titles:
        return generate_alert_title_from_violations(violations)
    
    return title.strip()

def format_field_name(field: str) -> str:
    """Convert field key to human-readable name (e.g., motor_current -> Motor Current)."""
    if not field:
        return "Unknown Parameter"
    return ' '.join(word.capitalize() for word in field.replace('_', ' ').split())

def calculate_data_driven_confidence(anomaly_details: Dict) -> int:
    """
    Calculate confidence as a truly data-driven value based on all available metrics.
    Returns an integer percentage between 1 and 99.
    
    The confidence reflects how certain we are that this is a real anomaly,
    based on multiple factors from the actual data.
    """
    import math
    
    # Get all available scores
    rule_score = float(anomaly_details.get('rule_score', 0.0) or 0.0)
    ml_score = float(anomaly_details.get('ml_score', 0.0) or 0.0)
    lstm_score = float(anomaly_details.get('lstm_score', 0.0) or 0.0)
    combined_score = float(anomaly_details.get('anomaly_score', 0.0) or 0.0)
    
    violations = anomaly_details.get('violations', [])
    
    if not violations:
        # No violations - use combined score directly
        return max(1, min(99, int(round(combined_score * 100))))
    
    # Calculate weighted deviation score from all violations
    total_deviation_weight = 0.0
    deviation_sum = 0.0
    
    for v in violations:
        dev_pct = float(v.get('deviation_pct', 0) or 0)
        # Weight each violation - higher deviations matter more
        weight = 1.0 + (dev_pct / 50.0)  # 50% deviation = 2x weight
        deviation_sum += dev_pct * weight
        total_deviation_weight += weight
    
    # Average weighted deviation
    avg_weighted_deviation = deviation_sum / total_deviation_weight if total_deviation_weight > 0 else 0
    
    # Normalize deviation to 0-1 scale using logarithmic scaling for better distribution
    # This prevents clustering and gives smooth distribution
    deviation_factor = math.log1p(avg_weighted_deviation) / math.log1p(100)  # log(1+x) scaling
    deviation_factor = min(1.0, deviation_factor)
    
    # Calculate components with different weights
    # 1. Deviation component (40% weight) - actual sensor deviation from range
    deviation_component = deviation_factor * 0.40
    
    # 2. Rule score component (25% weight) - rule-based detection strength
    rule_component = rule_score * 0.25
    
    # 3. ML score component (20% weight) - machine learning model confidence
    ml_component = ml_score * 0.20
    
    # 4. Violation count component (15% weight) - number of simultaneous violations
    # Use sqrt for diminishing returns: 1 vio=1, 2 vio=1.41, 3 vio=1.73, 4 vio=2
    violation_count_factor = min(1.0, math.sqrt(len(violations)) / 2.0)
    violation_component = violation_count_factor * 0.15
    
    # Combine all components
    base_confidence = deviation_component + rule_component + ml_component + violation_component
    
    # Add micro-variance based on actual deviation values for uniqueness
    # This ensures even similar cases get slightly different confidence
    variance_seed = sum(float(v.get('deviation_pct', 0) or 0) * (i + 1) for i, v in enumerate(violations))
    micro_variance = (variance_seed % 7) / 100.0  # Adds 0-6% based on actual data
    
    final_confidence = base_confidence + micro_variance
    
    # Convert to percentage and clamp
    conf_percent = int(round(final_confidence * 100))
    conf_percent = max(1, min(99, conf_percent))
    
    return conf_percent

def generate_suggestion(anomaly_details, well_id, readings, timestamp_str, historical_anomalies=None, lift_type=None):
    try:
        # Prepare historical context
        formatted_history = []
        if historical_anomalies:
            for anom in historical_anomalies:
                raw_data = anom.get('raw_anomaly_data')
                if isinstance(raw_data, str):
                    try:
                        raw_data = json.loads(raw_data)
                    except:
                        raw_data = {}
                
                past_violations_list = []
                if raw_data and 'violations' in raw_data and isinstance(raw_data['violations'], list):
                    for v in raw_data['violations']:
                        raw_msg = v.get('violation', '')
                        # Regex cleaner to remove deviation % or "(Historical)" text before sending to AI
                        clean_msg = re.sub(r'\s*\(.*?\)', '', raw_msg).strip() 
                        past_violations_list.append({
                            "field": v.get('field'),
                            "violation": clean_msg 
                        })

                formatted_history.append({
                    "well": anom.get('well_id'),          
                    "date": str(anom.get('timestamp')),   
                    "issue": anom.get('alert_title'),
                    "severity": anom.get('severity'),
                    "violations": past_violations_list,
                    "readings": raw_data.get('full_readings', {}),
                })
        
        
        history_context_str = json.dumps(formatted_history, indent=2, cls=CustomJSONEncoder, default=str)
        
        history_guidance = ""
        if not formatted_history:
            history_guidance = """
NOTE: No similar historical incidents found matching the current anomaly within 20% tolerance.
This could indicate:
1. Historical database has limited data
2. The violation pattern is unique

IMPORTANT: You MUST still provide a 'key_learnings' field stating:
"No similar historical incidents found."
"""
    
        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
 
        sensor_ranges_str = "{}"
        try:
            if lift_type and DYNAMIC_CONFIG_AVAILABLE:
                sensor_ranges = get_sensor_ranges(lift_type)
                sensor_ranges_str = json.dumps(sensor_ranges, indent=2, cls=CustomJSONEncoder, default=str)
        except Exception:
            sensor_ranges_str = "{}"
        
        # Format violations with comma-delimited numbers for display
        formatted_violations = format_violations_for_display(anomaly_details.get('violations', []))
            
        # RESTORED FULL ORIGINAL PROMPT
        prompt = f"""
        You are an expert Production Engineer AI assistant. Analyze the new anomaly.
        
        ---
        New Anomaly Details:
        Well ID: {well_id}
        Timestamp: {timestamp_str}
        Current Sensor Readings:
        {json.dumps(readings, indent=2, cls=CustomJSONEncoder, default=str)}
        Violations Detected:
        {json.dumps(formatted_violations, indent=2, cls=CustomJSONEncoder, default=str)}
        
        Authoritative Sensor Ranges for lift type ({lift_type}):
        {sensor_ranges_str}
        ---
        Real Database History for this well (for context):
        {history_context_str}
        
        {history_guidance}
        ---
 
        IMPORTANT: Use the units provided in each violation entry exactly as given. Do NOT convert units (e.g., do not change °F to °C) or invent ranges. Use the 'unit', 'min', and 'max' fields from the Violations Detected objects when describing values and expected ranges.
        Additionally, do not use raw parameter keys as names every where. Convert them into clear, human-readable labels (for example, rod_pump → Rod Pump).

        Based on the information above, provide a structured JSON response.

        SEVERITY CLASSIFICATION GUIDE (Authoritative - follow code logic):
        - CRITICAL: 3+ violations with strong signal (>=0.85) or avg deviation >60%, or anomaly_score >= 0.98
        - HIGH: 2 violations with moderate signal (>=0.60) or avg deviation >40%, or anomaly_score >= 0.75
        - MEDIUM: 1 violation with modest signal (>=0.35) or avg deviation >12%
        - LOW: otherwise (small deviations or weak ML signal)
        
        CONFIDENCE CALCULATION:
        Calculate confidence as a SINGLE percentage value directly derived from the anomaly score.
        DO NOT use bucketed ranges. Use this exact rule:
        - Compute `confidence_percent = round(anomaly_score * 100)`.
        - Clamp `confidence_percent` into the inclusive range 1 to 99.
        - Return `confidence` as a string with a percent sign, e.g. "78%".
        The returned confidence MUST match this calculation and be a single number string.
        
        PRE-CLASSIFIED SEVERITY: {calculated_severity}
        ANOMALY SCORE: {anomaly_details.get('anomaly_score', 0):.2f}
        Strictly assign severity level that matches or stays close to this pre-classification.

        Important:
        1. If 'Real Database History' above is not an empty list, populate 'similar_incidents' using that data.
        
        Required JSON Structure:
        1. "alert_title": STRICT RULES - Read carefully:
            A) NEVER include "ESP", "Rod Pump", "Gas Lift", or any lift type name in the title.
            B) ONLY use the violated parameter name from 'Violations Detected' section.
            C) Format: "[Parameter Name] [Issue Type]" - nothing else.
            
            CORRECT Examples (use these exact patterns):
            - "Torque Overload" (for torque violation)
            - "Strokes Per Minute Anomaly" (for strokes_per_minute violation)
            - "Polish Rod Load Spike" (for polish_rod_load violation)
            - "Pump Fillage Low" (for pump_fillage violation)
            - "Tubing Pressure High" (for tubing_pressure violation)
            - "Motor Current Spike" (for motor_current violation)
            - "Motor Temperature High" (for motor_temp violation)
            - "Discharge Pressure Drop" (for discharge_pressure violation)
            - "Injection Rate Anomaly" (for injection_rate violation)
            - "Bottomhole Pressure Low" (for bottomhole_pressure violation)
            
            WRONG Examples (NEVER do this):
            - "ESP Motor Current Spike" ← WRONG (contains ESP)
            - "Rod Pump Torque Overload" ← WRONG (contains Rod Pump)
            - "Gas Lift Injection Anomaly" ← WRONG (contains Gas Lift)
            - "ESP Anomaly Detected" ← WRONG (generic and contains ESP)
            
            Convert parameter keys to human-readable names (e.g., motor_current → Motor Current, strokes_per_minute → Strokes Per Minute).
            Use the MOST CRITICAL violation (highest deviation) from 'Violations Detected' to determine the title.
        2. "severity": Assign based on severity matrix above (CRITICAL, HIGH, MEDIUM, or LOW).
        3. "status": "ACTIVE".
        4. "confidence": IMPORTANT - Single Percentage string (e.g., "78%") calculated from anomaly score. MUST be a single number. DO NOT return a range.
        5. "description": Write a concise, technical paragraph. Within this paragraph, you MUST clearly describe each parameter from 'Violations Detected' that is out of range.
            For each parameter, state the nature of the anomaly by including its observed value and the expected range, both with their units.
            Only mention violations detected, not all parameters.
            For example: "The motor temperature reached 265 °F, exceeding the expected operational range of 100–250 °F." Use the 'unit' field from each violation object.
        6. "suggested_actions": A list of three clear, concise and actionable steps.
        7. "explanation": Detailed explanation of the new anomaly.
        
        CRITICAL FORMATTING REQUIREMENT FOR ALL TEXT FIELDS (description, suggested_actions, explanation):
        Use comma-delimited numbers (e.g., "1,234,567.89" NOT "1234567.89"). This applies to all numeric values.
        Use the exact formatted values provided in the 'Violations Detected' section above - they already have proper comma delimiters.
        8. "historical_context": Object containing:
            - "asset_history": Object with keys "commissioned" (date), "operating_hours" (int), "last_inspection" (date). Since this data is not provided, you MUST use the string "Data Not Available" for each value.
            - "similar_incidents": A list of objects based ONLY on the 'Real Database History' provided above. If the history is empty, this MUST be an empty list []. For each incident in the history, create an object with the following keys:
                - "well": Use the "well" value from the history.
                - "date": Use the "date" value from the history, use format "MM/DD/YYYY HH:MM".
                - "issue": Summarize the technical issue briefly (e.g., "Motor Current High" or "Pressure Violation"). Do NOT copy long "Out of range" error strings.
                - "severity": Use the "severity" value from the history.
                - "violations": Use the "violations" array from history (containing "field" and "violation", violation must be in simple value example 300 A). Do NOT repeat the parameter name or add words like "value" or "Historical" in value.
            - "key_learnings": String field (NOT inside similar_incidents). ALWAYS REQUIRED. 
                * If similar_incidents is empty, write: "No similar historical incidents found."
                * If similar_incidents exist, analyze patterns in frequency, severity, or affected parameters (e.g., "Three pressure-related incidents occurred within 24 hours, suggesting potential equipment degradation").
       
        9. "risk_analysis": Object containing:
            - "severity_breakdown": Object with keys "safety_risk", "production_impact", "financial_exposure".
            - "mitigation_strategies": Object with keys "immediate", "short_term", "long_term".
            - "decision_support": A final string recommendation.
            
        Ensure your entire output is a single, valid JSON object.
        """
 
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for oil and gas anomaly detection. Output valid JSON only, matching the required schema exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
 
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        suggestion_data = json.loads(content.strip())
        
        # Sanitize alert_title to remove any lift type names that OpenAI may have included
        if 'alert_title' in suggestion_data:
            suggestion_data['alert_title'] = sanitize_alert_title(
                suggestion_data.get('alert_title', ''),
                anomaly_details.get('violations', [])
            )
        
        if 'historical_context' not in suggestion_data:
            suggestion_data['historical_context'] = {}
        
        hc = suggestion_data['historical_context']
        
        if 'key_learnings' not in hc or not hc['key_learnings']:
            if not hc.get('similar_incidents'):
                hc['key_learnings'] = "No similar historical incidents found within 20% tolerance. This appears to be a novel anomaly pattern for this well, requiring careful investigation."
            else:
                hc['key_learnings'] = "Historical patterns identified - see similar incidents for details."
        
        if 'similar_incidents' not in hc:
            hc['similar_incidents'] = []
        
        if 'asset_history' not in hc:
            hc['asset_history'] = {
                "commissioned": "Data Not Available",
                "operating_hours": "Data Not Available",
                "last_inspection": "Data Not Available"
            }
        
        return suggestion_data
 
    except Exception as e:
        logging.error(f"Error generating suggestion: {e}")
        # Generate alert_title from violations using the dedicated function
        violations = anomaly_details.get('violations', [])
        fallback_title = generate_alert_title_from_violations(violations)
        
        return {
            "alert_title": fallback_title,
            "severity": "UNKNOWN",
            "status": "ACTIVE",
            "confidence": "0%",
            "description": "Could not generate detailed analysis.",
            "suggested_actions": ["Investigate manually"],
            "explanation": f"AI Generation failed: {str(e)}",
            "historical_context": {
                "asset_history": {"commissioned": "N/A", "operating_hours": "N/A", "last_inspection": "N/A"},
                "similar_incidents": [],
                "key_learnings": "AI generation failed."
            },
            "risk_analysis": {}
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/form-config', methods=['GET'])
def get_form_config():
    try:
        if not DYNAMIC_CONFIG_AVAILABLE:
            return jsonify({"error": "Dynamic config not available"}), 503
        
        lift_types = get_lift_types()
        config = {}
        for lift_type_name in lift_types.keys():
            config[lift_type_name] = {
                'sensors': get_form_sensors(lift_type_name),
                'ranges': get_sensor_ranges(lift_type_name)
            }
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/insert-reading', methods=['POST'])
def insert_reading():
    try:
        data = request.get_json()
        
        if 'well_id' not in data or 'timestamp' not in data:
            return jsonify({"error": "Missing required field"}), 400

        well_id = data.get('well_id')
        timestamp_str = data.get('timestamp')
        lift_type = data.get('lift_type')
        
        # Validate lift_type is provided and not empty
        if lift_type is None:
            return jsonify({"error": "lift_type is required - must provide a valid well type (Rod Pump, ESP, or Gas Lift)"}), 400
        
        # Check for string 'None' or 'null'
        if isinstance(lift_type, str) and lift_type.lower() in ['none', 'null', '']:
            return jsonify({"error": "lift_type is required - must provide a valid well type (Rod Pump, ESP, or Gas Lift)"}), 400
        
        # Convert to string and strip whitespace
        lift_type = str(lift_type).strip()
        
        if not lift_type or lift_type == '':
            return jsonify({"error": "lift_type is required - must provide a valid well type (Rod Pump, ESP, or Gas Lift)"}), 400
        
        logging.info(f"DEBUG: ===== VALIDATION PASSED, lift_type={repr(lift_type)} =====")
        
        field_list = _get_sensor_field_list(lift_type)
        readings = {field: (float(data[field]) if data.get(field) is not None else None) for field in field_list}

        anomaly_details = check_anomaly(readings, well_id=well_id, lift_type=lift_type)
        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
        
        try:
            from anomaly_detection.detector import ML_ONLY_ANOMALY_THRESHOLD
            if (not anomaly_details.get('violations')) and float(anomaly_details.get('anomaly_score', 0)) < ML_ONLY_ANOMALY_THRESHOLD:
                anomaly_details['is_anomaly'] = False
        except Exception:
            pass

        # ===== DETERMINE WELL STATUS =====
        # Status is determined by well_status_service based on production metrics
        # SHUTDOWN status is handled separately by the background service (no readings for 24h)
        well_status = get_status_for_reading(well_id, lift_type, readings)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            def convert_for_db(val):
                if isinstance(val, (np.integer, int)): return int(val)
                elif isinstance(val, (np.floating, float)): return float(val)
                elif isinstance(val, (np.bool_, bool)): return bool(val)
                elif isinstance(val, np.ndarray): return val.tolist()
                return val
            
            # --- 1. INSERT RAW READING ---
            field_names = ['well_id', 'timestamp', 'lift_type', 'well_status'] + list(readings.keys())
            field_values = [well_id, timestamp_str, lift_type, well_status] + [convert_for_db(v) for v in readings.values()]
            placeholders = ', '.join(['%s'] * len(field_values))
            fields_str = ', '.join(field_names)
            
            query_readings = f"INSERT INTO well_sensor_readings ({fields_str}) VALUES ({placeholders})"
            
            try:
                cursor.execute(query_readings, field_values)
            except Exception as e:
                # If lift_type or well_status column is missing, retry without them
                if "lift_type" in str(e).lower() or "well_status" in str(e).lower() or "invalid identifier" in str(e).lower():
                    field_names = ['well_id', 'timestamp'] + list(readings.keys())
                    field_values = [well_id, timestamp_str] + [convert_for_db(v) for v in readings.values()]
                    placeholders = ', '.join(['%s'] * len(field_values))
                    fields_str = ', '.join(field_names)
                    query_readings = f"INSERT INTO well_sensor_readings ({fields_str}) VALUES ({placeholders})"
                    cursor.execute(query_readings, field_values)
                else:
                    raise
            
            conn.commit()

            # --- 2. INSERT ANOMALY (If Detected) ---
            suggestion_result = None
            if anomaly_details['is_anomaly']:
                suggestion_result = _record_anomaly_and_suggestion(
                    conn, well_id, timestamp_str, lift_type, readings, anomaly_details, well_status
                )

        finally:
            cursor.close()
            conn.close()
        
        # Format violations with comma-delimited numbers for API response
        formatted_violations = format_violations_for_display(anomaly_details.get('violations', []))
        
        return jsonify({
            "success": True,
            "message": f"Reading inserted for {well_id}",
            "anomaly_detected": anomaly_details['is_anomaly'],
            "anomaly_score": anomaly_details['anomaly_score'],
            "violations": formatted_violations,
            "suggestion": suggestion_result
        }), 200

    except Exception as e:
        logging.error(f"Error inserting reading: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route('/api/scan-recent-readings', methods=['GET', 'POST'])
def scan_recent_readings():
    """
    Fetch the last N (default 200) rows from well_sensor_readings, run anomaly detection
    on each, and for any detected anomaly record it and generate suggestion (same flow as
    after inserting a reading). Skips readings that already have an anomaly record.
    """
    limit = request.args.get('limit', 200)
    try:
        limit = int(limit)
        limit = max(1, min(limit, 10000))
    except (TypeError, ValueError):
        limit = 200

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM well_sensor_readings ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            rows = cursor.fetchall()
            cols = [c[0].lower() for c in cursor.description] if cursor.description else []
        finally:
            cursor.close()

        if not rows:
            conn.close()
            return jsonify({
                "success": True,
                "scanned": 0,
                "anomalies_detected": 0,
                "anomalies_recorded": 0,
                "message": "No readings found",
                "details": []
            }), 200

        # Build list of dicts for each row
        readings_list = [dict(zip(cols, r)) for r in rows]
        conn.close()
    except Exception as e:
        logging.error(f"Error fetching recent readings: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    anomalies_detected = 0
    anomalies_recorded = 0
    details = []

    for row in readings_list:
        well_id = row.get('well_id')
        ts = row.get('timestamp')
        lift_type = row.get('lift_type') or (str(row.get('lift_type', '')).strip())
        if not lift_type or lift_type.lower() in ('none', 'null', ''):
            continue
        timestamp_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        readings = _readings_from_row(row, lift_type)

        # Skip if this reading already has an anomaly record
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM well_anomalies WHERE well_id = %s AND timestamp = %s LIMIT 1",
                (well_id, timestamp_str)
            )
            already = cur.fetchone() is not None
            cur.close()
            conn.close()
            if already:
                continue
        except Exception as e:
            logging.warning(f"Check existing anomaly failed for {well_id} {timestamp_str}: {e}")
            continue

        anomaly_details = check_anomaly(readings, well_id=well_id, lift_type=lift_type)
        try:
            from anomaly_detection.detector import ML_ONLY_ANOMALY_THRESHOLD
            if (not anomaly_details.get('violations')) and float(anomaly_details.get('anomaly_score', 0)) < ML_ONLY_ANOMALY_THRESHOLD:
                anomaly_details = {**anomaly_details, 'is_anomaly': False}
        except Exception:
            pass

        if not anomaly_details.get('is_anomaly'):
            continue

        anomalies_detected += 1
        well_status = row.get('well_status') or get_status_for_reading(well_id, lift_type, readings)

        try:
            conn = get_db_connection()
            suggestion_result = _record_anomaly_and_suggestion(
                conn, well_id, timestamp_str, lift_type, readings, anomaly_details, well_status
            )
            conn.close()
            anomalies_recorded += 1
            details.append({
                "well_id": well_id,
                "timestamp": timestamp_str,
                "lift_type": lift_type,
                "anomaly_score": anomaly_details.get('anomaly_score'),
                "alert_title": suggestion_result.get('alert_title') if suggestion_result else None,
            })
        except Exception as e:
            logging.error(f"Error recording anomaly for {well_id} {timestamp_str}: {e}")
            details.append({
                "well_id": well_id,
                "timestamp": timestamp_str,
                "lift_type": lift_type,
                "error": str(e),
            })

    return jsonify({
        "success": True,
        "scanned": len(readings_list),
        "anomalies_detected": anomalies_detected,
        "anomalies_recorded": anomalies_recorded,
        "message": f"Scanned {len(readings_list)} readings; detected {anomalies_detected} anomalies, recorded {anomalies_recorded}.",
        "details": details,
    }), 200


@app.route('/api/well-history/<well_id>', methods=['GET'])
def get_well_history(well_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 20", (well_id,))
            rows = cursor.fetchall()
            cols = [c[0].lower() for c in cursor.description] if cursor.description else []
            readings_df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
        finally:
            cursor.close()

        suggestions_list = []
        pg_conn = get_pg_connection()
        if pg_conn:
            try:
                with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM anomaly_suggestions WHERE well_id = %s ORDER BY timestamp DESC LIMIT 20", (well_id,))
                    suggestions_list = cur.fetchall()
                    for item in suggestions_list:
                        item['timestamp'] = str(item.get('timestamp'))
                        item['created_at'] = str(item.get('created_at'))
            except Exception:
                pass
            finally:
                pg_conn.close()

        if not suggestions_list:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM well_anomalies WHERE well_id = %s ORDER BY timestamp DESC LIMIT 20", (well_id,))
                rows = cursor.fetchall()
                cols = [c[0].lower() for c in cursor.description] if cursor.description else []
                anomalies_df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
            finally:
                cursor.close()

            for _, row in anomalies_df.iterrows():
                record = row.to_dict()
                record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                if 'timestamp' in record: record['timestamp'] = str(record['timestamp'])
                suggestions_list.append({
                    "alert_title": "Anomaly Detected (Legacy)",
                    "severity": "UNKNOWN",
                    "status": record.get("status", "New"),
                    "confidence": "N/A",
                    "description": record.get("anomaly_type", "Unknown"),
                    "timestamp": record.get("timestamp"),
                    "raw_anomaly_data": record
                })

        conn.close()
        
        readings_list = []
        for _, row in readings_df.iterrows():
            record = row.to_dict()
            record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            if 'timestamp' in record: record['timestamp'] = str(record['timestamp'])
            readings_list.append(record)
        
        return jsonify({
            "well_id": well_id,
            "readings": readings_list,
            "anomalies": suggestions_list 
        }), 200
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/wells', methods=['GET'])
def get_wells():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT well_id FROM well_sensor_readings ORDER BY well_id")
            rows = cursor.fetchall()
            wells = [r[0] for r in rows] if rows else []
        finally:
            cursor.close()
            conn.close()
        return jsonify({"wells": wells}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/api/operation-recommendations', methods=['GET'])
def get_operation_recommendations():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT ID, WELL_ID, ACTION, PRIORITY, STATUS, REASON, EXPECTED_IMPACT, CONFIDENCE, PREDICTION_SOURCE, PROBABILITY, DETAILS, CREATED_AT
        FROM OPERATION_RECOMMENDATION ORDER BY CREATED_AT DESC, WELL_ID
        """)
        results = cursor.fetchall()
        columns = [desc[0].lower() for desc in cursor.description] if cursor.description else []
        cursor.close()
        conn.close()
        recommendations = []
        if results:
            for row in results:
                rec = {}
                for i, col in enumerate(columns):
                    rec[col] = row[i] if i < len(row) else None
                recommendations.append(rec)
        return jsonify({"status": "success", "count": len(recommendations), "recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-pipeline', methods=['GET'])
def run_pipeline_api():
    """Run the operational pipeline with optional `confidence` query param and stream logs via SSE."""
    confidence = request.args.get('confidence', '0.2')
    try:
        # validate numeric
        _ = float(confidence)
    except Exception:
        confidence = '0.2'

    def generator():
        for chunk in stream_run_operational_pipeline(confidence):
            yield chunk

    return Response(stream_with_context(generator()), mimetype='text/event-stream')

@app.route('/api/operation-suggestions', methods=['GET'])
def get_operation_suggestions_api():
    try:
        well_id = request.args.get('well_id')
        status = request.args.get('status')
        priority = request.args.get('priority')
        limit = int(request.args.get('limit', 100))
        suggestions = get_operation_suggestions(well_id=well_id, status=status, priority=priority, limit=limit)
        for sugg in suggestions:
            for key, value in sugg.items():
                if hasattr(value, 'isoformat'): sugg[key] = value.isoformat()
        return jsonify({"status": "success", "count": len(suggestions), "suggestions": suggestions}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/operation-suggestions/<suggestion_id>', methods=['GET'])
def get_operation_suggestion_api(suggestion_id):
    try:
        suggestion = get_operation_suggestion_detail(suggestion_id)
        if not suggestion: return jsonify({"status": "error", "message": "Suggestion not found"}), 404
        for key, value in suggestion.items():
            if hasattr(value, 'isoformat'): suggestion[key] = value.isoformat()
        return jsonify({"status": "success", "suggestion": suggestion}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/operation-suggestions', methods=['POST'])
def create_operation_suggestion_api():
    try:
        data = request.get_json()
        well_id = data.get('well_id')
        action = data.get('action')
        if not well_id or not action: return jsonify({"status": "error", "message": "Required fields missing"}), 400
        
        success = save_operation_suggestion(
            well_id=well_id,
            action=action,
            status=data.get('status', 'New'),
            priority=data.get('priority', 'HIGH'),
            confidence=float(data.get('confidence', 0.0)),
            production_data=data.get('production_data'),
            sensor_metrics=data.get('sensor_metrics'),
            reason=data.get('reason', ''),
            expected_impact=data.get('expected_impact', '')
        )
        if success: return jsonify({"status": "success", "message": "Created"}), 201
        return jsonify({"status": "error", "message": "Failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/operation-suggestions/<suggestion_id>/status', methods=['PUT'])
def update_operation_suggestion_status_api(suggestion_id):
    try:
        data = request.get_json()
        new_status = data.get('status')
        if not new_status: return jsonify({"status": "error", "message": "status required"}), 400
        success = update_operation_suggestion_status(suggestion_id, new_status)
        if success: return jsonify({"status": "success"}), 200
        return jsonify({"status": "error", "message": "Failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== WELL STATUS ENDPOINTS =====

@app.route('/api/well-statuses', methods=['GET'])
def get_all_well_statuses_api():
    """Get all well statuses from WELL_STATUS table."""
    try:
        from well_status.service import get_all_well_statuses
        statuses = get_all_well_statuses()
        
        # Convert datetime objects to strings
        for status in statuses:
            for key, value in status.items():
                if hasattr(value, 'isoformat'):
                    status[key] = value.isoformat()
        
        # Count by status
        summary = {'RUNNING': 0, 'IDLE': 0, 'SHUTDOWN': 0, 'UNKNOWN': 0}
        for s in statuses:
            status_val = s.get('status', 'UNKNOWN')
            summary[status_val] = summary.get(status_val, 0) + 1
        
        return jsonify({
            "status": "success",
            "count": len(statuses),
            "summary": summary,
            "well_statuses": statuses
        }), 200
    except ImportError:
        return jsonify({"status": "error", "message": "well_status_service not available"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/well-statuses/<well_id>', methods=['GET'])
def get_well_status_api(well_id):
    """Get status for a specific well."""
    try:
        from well_status.service import get_well_status
        status = get_well_status(well_id)
        
        if not status:
            return jsonify({"status": "error", "message": f"Well {well_id} not found in status table"}), 404
        
        # Convert datetime objects to strings
        for key, value in status.items():
            if hasattr(value, 'isoformat'):
                status[key] = value.isoformat()
        
        return jsonify({
            "status": "success",
            "well_status": status
        }), 200
    except ImportError:
        return jsonify({"status": "error", "message": "well_status_service not available"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/well-statuses/refresh', methods=['POST'])
def refresh_well_statuses_api():
    """Manually trigger a refresh of all well statuses."""
    try:
        from well_status.service import update_all_well_statuses
        result = update_all_well_statuses()
        
        return jsonify({
            "status": "success",
            "message": "Well statuses refreshed",
            "summary": result
        }), 200
    except ImportError:
        return jsonify({"status": "error", "message": "well_status_service not available"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.getenv("APP_DEBUG", "False").lower() == "true"
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "5000"))
    app.run(debug=debug_mode, host=host, port=port)