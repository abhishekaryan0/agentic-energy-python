#!/usr/bin/env python3
"""
Well Status Service
Manages well operational status (RUNNING, IDLE, SHUTDOWN) based on sensor readings activity.

Status Logic:
- SHUTDOWN: No sensor readings for last 24 hours (1 day)
- IDLE: Recent readings but low/zero production metrics based on lift type
- RUNNING: Normal operation with recent readings

Run this service periodically (e.g., every hour via cron) to update well statuses.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Logging disabled - set to WARNING to suppress INFO logs
# Change to logging.INFO if you need to see detailed logs
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SHUTDOWN_THRESHOLD_HOURS = 24  # No readings for 24 hours = SHUTDOWN

# IDLE detection thresholds - multiple metrics per lift type
# Each metric has: field_name, threshold, operator ('lt' = less than, 'gt' = greater than), weight
IDLE_METRICS = {
    'Rod Pump': [
        # Primary indicators
        {'field': 'strokes_per_minute', 'threshold': 2.0, 'operator': 'lt', 'weight': 3, 'description': 'SPM very low'},
        {'field': 'pump_fillage', 'threshold': 10.0, 'operator': 'lt', 'weight': 2, 'description': 'Pump fillage very low'},
        {'field': 'torque', 'threshold': 50000, 'operator': 'lt', 'weight': 2, 'description': 'Torque minimal'},
        # Secondary indicators
        {'field': 'polish_rod_load', 'threshold': 5000, 'operator': 'lt', 'weight': 1, 'description': 'Rod load minimal'},
        {'field': 'tubing_pressure', 'threshold': 50, 'operator': 'lt', 'weight': 1, 'description': 'Tubing pressure very low'},
    ],
    'ESP': [
        # Primary indicators
        {'field': 'motor_current', 'threshold': 5.0, 'operator': 'lt', 'weight': 3, 'description': 'Motor current very low'},
        {'field': 'motor_temp', 'threshold': 80.0, 'operator': 'lt', 'weight': 2, 'description': 'Motor temp at ambient'},
        {'field': 'discharge_pressure', 'threshold': 100, 'operator': 'lt', 'weight': 2, 'description': 'Discharge pressure minimal'},
        # Secondary indicators
        {'field': 'motor_voltage', 'threshold': 100, 'operator': 'lt', 'weight': 1, 'description': 'Motor voltage very low'},
        {'field': 'pump_intake_pressure', 'threshold': 50, 'operator': 'lt', 'weight': 1, 'description': 'Intake pressure very low'},
    ],
    'Gas Lift': [
        # Primary indicators
        {'field': 'injection_rate', 'threshold': 5.0, 'operator': 'lt', 'weight': 3, 'description': 'Injection rate very low'},
        {'field': 'injection_pressure', 'threshold': 200, 'operator': 'lt', 'weight': 2, 'description': 'Injection pressure low'},
        {'field': 'bottomhole_pressure', 'threshold': 100, 'operator': 'lt', 'weight': 2, 'description': 'BHP very low'},
        # Secondary indicators
        {'field': 'injection_temperature', 'threshold': 50, 'operator': 'lt', 'weight': 1, 'description': 'Injection temp at ambient'},
        {'field': 'cycle_time', 'threshold': 5, 'operator': 'lt', 'weight': 1, 'description': 'Cycle time very short'},
    ]
}

# Minimum weighted score to consider well as IDLE
IDLE_SCORE_THRESHOLD = 4  # Need at least 4 weighted points to be IDLE

SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}


def get_snowflake_connection():
    """Get Snowflake database connection."""
    try:
        from snowflake.connector import connect as snowflake_connect
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise


def get_all_wells() -> List[Dict]:
    """
    Get all distinct well_ids from well_sensor_readings table
    along with their latest reading info including all relevant sensor values.
    """
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        query = """
        WITH latest_readings AS (
            SELECT 
                WELL_ID,
                MAX(TIMESTAMP) AS LAST_TIMESTAMP
            FROM WELL_SENSOR_READINGS
            GROUP BY WELL_ID
        )
        SELECT 
            r.WELL_ID,
            r.TIMESTAMP AS LAST_READING_TIMESTAMP,
            r.LIFT_TYPE,
            -- Rod Pump fields
            r.STROKES_PER_MINUTE,
            r.TORQUE,
            r.POLISH_ROD_LOAD,
            r.PUMP_FILLAGE,
            r.TUBING_PRESSURE,
            -- ESP fields
            r.MOTOR_TEMP,
            r.MOTOR_CURRENT,
            r.DISCHARGE_PRESSURE,
            r.PUMP_INTAKE_PRESSURE,
            r.MOTOR_VOLTAGE,
            -- Gas Lift fields
            r.INJECTION_RATE,
            r.INJECTION_TEMPERATURE,
            r.BOTTOMHOLE_PRESSURE,
            r.INJECTION_PRESSURE,
            r.CYCLE_TIME
        FROM WELL_SENSOR_READINGS r
        INNER JOIN latest_readings lr 
            ON r.WELL_ID = lr.WELL_ID AND r.TIMESTAMP = lr.LAST_TIMESTAMP
        ORDER BY r.WELL_ID
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [c[0].lower() for c in cursor.description] if cursor.description else []
        
        wells = []
        for row in rows:
            well = dict(zip(cols, row))
            wells.append(well)
        
        logger.info(f"Found {len(wells)} wells in database")
        return wells
        
    except Exception as e:
        logger.error(f"Error fetching wells: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_anomaly_counts(well_id: str, hours: int = 24) -> Tuple[int, int]:
    """
    Get anomaly counts for a well in the last N hours.
    Returns (total_count, critical_count)
    """
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT 
            COUNT(*) AS total_count,
            SUM(CASE WHEN SEVERITY = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_count
        FROM WELL_ANOMALIES
        WHERE WELL_ID = %s
        AND TIMESTAMP >= DATEADD(hour, -%s, CURRENT_TIMESTAMP())
        """
        
        cursor.execute(query, (well_id, hours))
        row = cursor.fetchone()
        
        if row:
            return (int(row[0] or 0), int(row[1] or 0))
        return (0, 0)
        
    except Exception as e:
        logger.error(f"Error fetching anomaly counts for {well_id}: {e}")
        return (0, 0)
    finally:
        cursor.close()
        conn.close()


def calculate_idle_score(well: Dict, lift_type: str) -> Tuple[int, List[str]]:
    """
    Calculate IDLE score based on multiple metrics.
    Returns: (total_weighted_score, list_of_triggered_reasons)
    """
    if not lift_type or lift_type not in IDLE_METRICS:
        return (0, [])
    
    metrics = IDLE_METRICS[lift_type]
    total_score = 0
    triggered_reasons = []
    
    for metric in metrics:
        field = metric['field']
        threshold = metric['threshold']
        operator = metric['operator']
        weight = metric['weight']
        description = metric['description']
        
        actual_value = well.get(field)
        
        # Skip if no value available
        if actual_value is None:
            continue
        
        try:
            actual_value = float(actual_value)
            
            # Check if metric triggers IDLE condition
            triggered = False
            if operator == 'lt' and actual_value < threshold:
                triggered = True
            elif operator == 'gt' and actual_value > threshold:
                triggered = True
            elif operator == 'eq' and actual_value == threshold:
                triggered = True
            
            if triggered:
                total_score += weight
                triggered_reasons.append(f"{description} ({field}={actual_value:.2f}, threshold={threshold})")
                
        except (ValueError, TypeError):
            continue
    
    return (total_score, triggered_reasons)


def determine_well_status(well: Dict) -> Tuple[str, str]:
    """
    Determine well status based on:
    1. Time since last reading (SHUTDOWN if > 24 hours)
    2. Multi-metric IDLE detection (IDLE if weighted score >= threshold)
    3. Default to RUNNING
    
    Returns: (status, reason)
    """
    last_timestamp = well.get('last_reading_timestamp')
    lift_type = well.get('lift_type')
    
    # Check if we have timestamp
    if not last_timestamp:
        return ('SHUTDOWN', 'No sensor readings found')
    
    # Calculate hours since last reading
    now = datetime.utcnow()
    if isinstance(last_timestamp, str):
        try:
            last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        except:
            last_timestamp = datetime.strptime(last_timestamp[:19], '%Y-%m-%d %H:%M:%S')
    
    # Handle timezone-naive comparison
    if hasattr(last_timestamp, 'tzinfo') and last_timestamp.tzinfo is not None:
        last_timestamp = last_timestamp.replace(tzinfo=None)
    
    hours_since_reading = (now - last_timestamp).total_seconds() / 3600
    
    # CONDITION 1: SHUTDOWN if no readings for > 24 hours
    if hours_since_reading > SHUTDOWN_THRESHOLD_HOURS:
        return ('SHUTDOWN', f'No sensor readings for {hours_since_reading:.1f} hours (threshold: {SHUTDOWN_THRESHOLD_HOURS}h)')
    
    # CONDITION 2: Multi-metric IDLE detection
    if lift_type and lift_type in IDLE_METRICS:
        idle_score, idle_reasons = calculate_idle_score(well, lift_type)
        
        if idle_score >= IDLE_SCORE_THRESHOLD:
            # Summarize top reasons (max 3)
            top_reasons = idle_reasons[:3]
            reason_summary = "; ".join(top_reasons)
            return ('IDLE', f'{lift_type} shows low activity (score={idle_score}/{IDLE_SCORE_THRESHOLD}): {reason_summary}')
    
    # DEFAULT: RUNNING
    return ('RUNNING', f'Normal operation, last reading {hours_since_reading:.1f} hours ago')


def update_well_status(well_id: str, status: str, reason: str, well_data: Dict):
    """
    Update or insert well status in WELL_STATUS table.
    """
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Calculate reading age in hours
        last_timestamp = well_data.get('last_reading_timestamp')
        reading_age_hours = None
        if last_timestamp:
            now = datetime.utcnow()
            if isinstance(last_timestamp, str):
                try:
                    last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                except:
                    last_timestamp = datetime.strptime(last_timestamp[:19], '%Y-%m-%d %H:%M:%S')
            if hasattr(last_timestamp, 'tzinfo') and last_timestamp.tzinfo is not None:
                last_timestamp = last_timestamp.replace(tzinfo=None)
            reading_age_hours = (now - last_timestamp).total_seconds() / 3600
        
        # Get anomaly counts
        anomaly_count, critical_count = get_anomaly_counts(well_id, 24)
        
        # Use MERGE to upsert
        merge_query = """
        MERGE INTO WELL_STATUS AS target
        USING (SELECT %s AS WELL_ID) AS source
        ON target.WELL_ID = source.WELL_ID
        WHEN MATCHED THEN UPDATE SET
            LIFT_TYPE = %s,
            STATUS = %s,
            LAST_READING_TIMESTAMP = %s,
            LAST_READING_AGE_HOURS = %s,
            STROKES_PER_MINUTE = %s,
            MOTOR_CURRENT = %s,
            INJECTION_RATE = %s,
            ANOMALY_COUNT_24H = %s,
            CRITICAL_ANOMALY_COUNT_24H = %s,
            STATUS_REASON = %s,
            UPDATED_AT = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN INSERT (
            WELL_ID, LIFT_TYPE, STATUS, LAST_READING_TIMESTAMP, LAST_READING_AGE_HOURS,
            STROKES_PER_MINUTE, MOTOR_CURRENT, INJECTION_RATE,
            ANOMALY_COUNT_24H, CRITICAL_ANOMALY_COUNT_24H, STATUS_REASON
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            # For USING clause
            well_id,
            # For UPDATE
            well_data.get('lift_type'),
            status,
            well_data.get('last_reading_timestamp'),
            reading_age_hours,
            well_data.get('strokes_per_minute'),
            well_data.get('motor_current'),
            well_data.get('injection_rate'),
            anomaly_count,
            critical_count,
            reason,
            # For INSERT
            well_id,
            well_data.get('lift_type'),
            status,
            well_data.get('last_reading_timestamp'),
            reading_age_hours,
            well_data.get('strokes_per_minute'),
            well_data.get('motor_current'),
            well_data.get('injection_rate'),
            anomaly_count,
            critical_count,
            reason
        )
        
        cursor.execute(merge_query, params)
        conn.commit()
        
        logger.info(f"[{well_id}] Status updated: {status} - {reason}")
        
    except Exception as e:
        logger.error(f"Error updating status for {well_id}: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def get_well_status(well_id: str) -> Optional[Dict]:
    """
    Get current status for a specific well from WELL_STATUS table.
    """
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT 
            WELL_ID, LIFT_TYPE, STATUS, LAST_READING_TIMESTAMP,
            LAST_READING_AGE_HOURS, STROKES_PER_MINUTE, MOTOR_CURRENT,
            INJECTION_RATE, ANOMALY_COUNT_24H, CRITICAL_ANOMALY_COUNT_24H,
            STATUS_REASON, UPDATED_AT
        FROM WELL_STATUS
        WHERE WELL_ID = %s
        """
        
        cursor.execute(query, (well_id,))
        row = cursor.fetchone()
        
        if row:
            cols = [c[0].lower() for c in cursor.description]
            return dict(zip(cols, row))
        return None
        
    except Exception as e:
        logger.error(f"Error fetching status for {well_id}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def get_all_well_statuses() -> List[Dict]:
    """
    Get all well statuses from WELL_STATUS table.
    """
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT 
            WELL_ID, LIFT_TYPE, STATUS, LAST_READING_TIMESTAMP,
            LAST_READING_AGE_HOURS, STROKES_PER_MINUTE, MOTOR_CURRENT,
            INJECTION_RATE, ANOMALY_COUNT_24H, CRITICAL_ANOMALY_COUNT_24H,
            STATUS_REASON, UPDATED_AT
        FROM WELL_STATUS
        ORDER BY 
            CASE STATUS 
                WHEN 'SHUTDOWN' THEN 1 
                WHEN 'IDLE' THEN 2 
                ELSE 3 
            END,
            WELL_ID
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [c[0].lower() for c in cursor.description] if cursor.description else []
        
        return [dict(zip(cols, row)) for row in rows]
        
    except Exception as e:
        logger.error(f"Error fetching all statuses: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def update_all_well_statuses():
    """
    Main function to update status for all wells.
    Should be run periodically (e.g., hourly via cron).
    """
    logger.info("=" * 50)
    logger.info("Starting well status update job")
    logger.info("=" * 50)
    
    wells = get_all_wells()
    
    if not wells:
        logger.warning("No wells found in database")
        return
    
    status_counts = {'RUNNING': 0, 'IDLE': 0, 'SHUTDOWN': 0}
    
    for well in wells:
        well_id = well.get('well_id')
        status, reason = determine_well_status(well)
        update_well_status(well_id, status, reason, well)
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.info("=" * 50)
    logger.info("Well status update complete")
    logger.info(f"Summary: RUNNING={status_counts['RUNNING']}, IDLE={status_counts['IDLE']}, SHUTDOWN={status_counts['SHUTDOWN']}")
    logger.info("=" * 50)
    
    return status_counts


def get_status_for_reading(well_id: str, lift_type: str, readings: Dict) -> str:
    """
    Get status for a new reading being inserted.
    This is called from app.py when inserting a reading.
    
    Since we're inserting a reading, well is active (not SHUTDOWN).
    Determine RUNNING vs IDLE based on multiple production metrics.
    
    Uses the same multi-metric scoring as the background service.
    """
    if not lift_type or lift_type not in IDLE_METRICS:
        return 'RUNNING'
    
    # Calculate IDLE score using the same logic as background service
    idle_score, idle_reasons = calculate_idle_score(readings, lift_type)
    
    if idle_score >= IDLE_SCORE_THRESHOLD:
        logger.info(f"[{well_id}] Real-time status: IDLE (score={idle_score}, reasons={idle_reasons[:2]})")
        return 'IDLE'
    
    return 'RUNNING'


# =============================================================================
# SCHEDULER CONFIGURATION NOTE:
# Currently set to 24-hour interval for batch processing.
# When real-time data pipeline is available, change to 10-minute schedule:
#   run_scheduler(interval_hours=0.167)  # 10 minutes = 0.167 hours
# Or modify the function to accept minutes directly.
# =============================================================================

def run_scheduler(interval_hours: int = 24):
    """
    Run the well status update automatically every N hours.
    Default: every 24 hours.
    
    Args:
        interval_hours: Hours between each run (default 24)
    """
    import time
    
    interval_seconds = interval_hours * 3600
    
    logger.info(f"Starting automatic well status scheduler (interval: {interval_hours} hours)")
    
    while True:
        try:
            # Run the update
            update_all_well_statuses()
            
            # Log next run time
            next_run = datetime.utcnow() + timedelta(hours=interval_hours)
            logger.info(f"Next automatic run scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Sleep until next run
            time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            # Wait 5 minutes before retrying on error
            time.sleep(300)


if __name__ == "__main__":
    """
    Run this script directly to update all well statuses.
    
    Usage:
        python well_status_service.py          # Run once
        python well_status_service.py --auto   # Run automatically every 24 hours
        python well_status_service.py --auto 12  # Run automatically every 12 hours
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Automatic mode - run every N hours (default 24)
        interval = 24
        if len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                interval = 24
        run_scheduler(interval_hours=interval)
    else:
        # Single run mode
        update_all_well_statuses()
