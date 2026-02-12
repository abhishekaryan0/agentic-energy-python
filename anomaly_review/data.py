"""
Data fetching and processing functions for anomaly review.

Provides functions for:
- Database connections (Snowflake, PostgreSQL)
- Well data fetching
- Rolling baseline calculations
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger('AnomalyReviewData')

# =============================================================================
# DATABASE CONNECTIONS
# =============================================================================

def get_snowflake_conn():
    """Create and return a Snowflake connection."""
    from snowflake.connector import connect as snowflake_connect
    
    return snowflake_connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )


def get_db_url() -> Optional[str]:
    """Get PostgreSQL database URL from environment."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    # Clean up schema parameter if present
    if "schema=" in url:
        return url.replace("?schema=public", "").replace("&schema=public", "")
    return url


def get_openai_client():
    """Create and return an OpenAI client."""
    import openai
    
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_well_data(well_id: str, days: int = None) -> pd.DataFrame:
    """
    Fetch well sensor and production data from Snowflake.
    
    Uses 'SELECT s.*' to automatically grab any new sensors added to the DB.
    
    Args:
        well_id: Well identifier
        days: Number of days of history to fetch (None = all)
        
    Returns:
        DataFrame with sensor and production data
    """
    params = [well_id]
    where_clause = ""
    
    if days:
        where_clause = "AND s.timestamp >= %s"
        cutoff_date = datetime.now() - timedelta(days=days)
        params.append(cutoff_date)
    
    query = f"""
    SELECT 
        s.*,
        p.oil_volume, p.water_volume, p.gas_volume
    FROM well_sensor_readings s
    LEFT JOIN well_daily_production p 
        ON s.well_id = p.well_id AND DATE(s.timestamp) = p.date
    WHERE s.well_id = %s
      {where_clause}
    ORDER BY s.timestamp ASC
    """
    
    try:
        with get_snowflake_conn() as conn:
            df = pd.read_sql(query, conn, params=params)
        
        # Normalize columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Handle timestamp variants
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        elif 'event_ts' in df.columns:
            df.rename(columns={'event_ts': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Safe forward fill for standard operational sensors
        operational_cols = [
            'motor_current', 'tubing_pressure', 'casing_pressure', 
            'strokes_per_minute', 'discharge_pressure', 'pump_intake_pressure'
        ]
        for c in operational_cols:
            if c in df.columns:
                df[c] = df[c].ffill()
        
        # Production data defaults to 0 if missing
        for prod_col in ['oil_volume', 'water_volume', 'gas_volume']:
            if prod_col in df.columns:
                df[prod_col] = df[prod_col].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Fetch failed for {well_id}: {e}")
        return pd.DataFrame()


def fetch_all_active_wells() -> List[str]:
    """Fetch list of all distinct well_ids from daily production."""
    query = "SELECT DISTINCT well_id FROM well_daily_production"
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(query).fetchall()
            return [r[0] for r in rows if r[0]]
    except Exception as e:
        logger.error(f"Failed to fetch all wells: {e}")
        return []


def get_recent_anomaly_history(
    well_id: str,
    anomaly_code: str,
    exclude_date: str = None,
) -> str:
    """
    Fetch recent anomaly history for a well (same code) for historical_context.
    Tries Postgres first, then Snowflake, so history is found regardless of where reviews are saved.
    """
    # 1) Try Postgres (same table we write to when DATABASE_URL is set)
    url = get_db_url()
    if url:
        try:
            import psycopg2
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT event_date, severity, title
                           FROM anomaly_review
                           WHERE well_id = %s AND anomaly_code = %s
                           ORDER BY event_date DESC
                           LIMIT 10""",
                        (well_id, anomaly_code),
                    )
                    rows = cur.fetchall()
            if rows:
                history_list = _format_history_rows(rows, exclude_date)
                if history_list:
                    return "\n".join(history_list)
        except Exception as e:
            logger.warning(f"get_recent_anomaly_history (Postgres) failed: {e}")

    # 2) Try Snowflake with fully qualified table (same as save path)
    try:
        from anomaly_review.config import SNOWFLAKE_ANOMALY_REVIEW_TABLE
        table = SNOWFLAKE_ANOMALY_REVIEW_TABLE
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""SELECT event_date, severity, title
                    FROM {table}
                    WHERE well_id = %s AND anomaly_code = %s
                    ORDER BY event_date DESC
                    LIMIT 10""",
                (well_id, anomaly_code),
            )
            rows = cur.fetchall()
        if rows:
            history_list = _format_history_rows(rows, exclude_date)
            if history_list:
                return "\n".join(history_list)
    except Exception as e:
        logger.warning(f"get_recent_anomaly_history (Snowflake) failed: {e}")

    return "No prior history found."


def _format_history_rows(rows, exclude_date: str = None):
    """Build list of history lines from (event_date, severity, title) rows."""
    out = []
    for r in rows:
        r_date = str(r[0]) if r[0] is not None else ""
        if exclude_date and r_date == str(exclude_date):
            continue
        sev = r[1] if r[1] is not None else "N/A"
        title = (r[2] or "Anomaly") if len(r) > 2 else "Anomaly"
        out.append(f"- Date: {r_date}, Severity: {sev}, Title: {title}")
    return out


def has_recent_anomaly(well_id: str, anomaly_code: str, within_days: int) -> bool:
    """
    Return True if this well already has an anomaly_review row for the given code
    with event_date within the last `within_days`. Used to avoid duplicate daily
    alerts (e.g. PATTERN_SHIFT only once per episode per well).
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=within_days)).isoformat()
    url = get_db_url()
    if url:
        try:
            import psycopg2
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM anomaly_review WHERE well_id = %s AND anomaly_code = %s AND event_date >= %s LIMIT 1",
                        (well_id, anomaly_code, cutoff),
                    )
                    if cur.fetchone():
                        return True
        except Exception as e:
            logger.warning(f"has_recent_anomaly (Postgres) failed: {e}")
    try:
        from anomaly_review.config import SNOWFLAKE_ANOMALY_REVIEW_TABLE
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT 1 FROM {SNOWFLAKE_ANOMALY_REVIEW_TABLE} WHERE well_id = %s AND anomaly_code = %s AND event_date >= %s LIMIT 1",
                (well_id, anomaly_code, cutoff),
            )
            if cur.fetchone():
                return True
    except Exception as e:
        logger.warning(f"has_recent_anomaly (Snowflake) failed: {e}")
    return False


def get_existing_anomaly_event_date(well_id: str, anomaly_code: str, within_days: int):
    """
    Return the event_date (YYYY-MM-DD string) of the most recent anomaly_review row
    for this well+code within the last `within_days`. Used to update that row with
    fresh content when the same anomaly is still severe (so it stays visible).
    Returns None if no row found.
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=within_days)).isoformat()
    url = get_db_url()
    if url:
        try:
            import psycopg2
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT event_date FROM anomaly_review
                           WHERE well_id = %s AND anomaly_code = %s AND event_date >= %s
                           ORDER BY event_date DESC LIMIT 1""",
                        (well_id, anomaly_code, cutoff),
                    )
                    row = cur.fetchone()
                    if row:
                        return str(row[0]) if row[0] else None
        except Exception as e:
            logger.warning(f"get_existing_anomaly_event_date (Postgres) failed: {e}")
    try:
        from anomaly_review.config import SNOWFLAKE_ANOMALY_REVIEW_TABLE
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""SELECT event_date FROM {SNOWFLAKE_ANOMALY_REVIEW_TABLE}
                    WHERE well_id = %s AND anomaly_code = %s AND event_date >= %s
                    ORDER BY event_date DESC LIMIT 1""",
                (well_id, anomaly_code, cutoff),
            )
            row = cur.fetchone()
            if row:
                return str(row[0]) if row[0] else None
    except Exception as e:
        logger.warning(f"get_existing_anomaly_event_date (Snowflake) failed: {e}")
    return None


# Global cache for sensor metadata
_SENSOR_METADATA_CACHE = None

def fetch_sensor_metadata() -> pd.DataFrame:
    """
    Fetch sensor definitions (Min/Max limits) from SENSOR_DEFINITIONS table.
    Caches result to avoid repeated DB hits.
    
    Returns:
        DataFrame with columns [FIELD_NAME, MIN_VALUE, MAX_VALUE]
    """
    global _SENSOR_METADATA_CACHE
    
    if _SENSOR_METADATA_CACHE is not None:
        return _SENSOR_METADATA_CACHE

    query = """
    SELECT FIELD_NAME, MIN_VALUE, MAX_VALUE 
    FROM SENSOR_DEFINITIONS 
    WHERE MIN_VALUE IS NOT NULL AND MAX_VALUE IS NOT NULL
    """
    
    try:
        with get_snowflake_conn() as conn:
            df = pd.read_sql(query, conn)
            
        # Normalize to lowercase to match well data columns
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            # field_name also needs to be lowercased for matching
            df['field_name'] = df['field_name'].str.lower()
            
        _SENSOR_METADATA_CACHE = df
        return df
    except Exception as e:
        logger.error(f"Failed to fetch sensor metadata: {e}")
        # Return empty DF so flow doesn't break, just won't filter
        return pd.DataFrame(columns=['field_name', 'min_value', 'max_value'])


# =============================================================================
# DATA PROCESSING
# =============================================================================

def calculate_rolling_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling baselines for various metrics.
    
    Args:
        df: DataFrame with sensor data
        
    Returns:
        DataFrame with added baseline columns
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp').sort_index()
    
    MIN_PERIODS = 7
    
    # Production baselines
    if 'oil_volume' in df.columns:
        df['oil_baseline_90d'] = df['oil_volume'].rolling('90D', min_periods=MIN_PERIODS).mean()
        df['oil_avg_7d'] = df['oil_volume'].rolling('7D', min_periods=3).mean()
        df['oil_avg_30d'] = df['oil_volume'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['oil_std_90d'] = df['oil_volume'].rolling('90D', min_periods=MIN_PERIODS).std()
    
    if 'gas_volume' in df.columns:
        df['gas_baseline_90d'] = df['gas_volume'].rolling('90D', min_periods=MIN_PERIODS).mean()
        df['gas_avg_7d'] = df['gas_volume'].rolling('7D', min_periods=3).mean()
        df['gas_avg_30d'] = df['gas_volume'].rolling('30D', min_periods=MIN_PERIODS).mean()

    if 'water_volume' in df.columns:
        df['water_avg_30d'] = df['water_volume'].rolling('30D', min_periods=MIN_PERIODS).mean()

    # Physics baselines
    if 'pump_fillage' in df.columns:
        df['fillage_avg_24h'] = df['pump_fillage'].rolling('24h', min_periods=1).mean()
        df['fillage_avg_3d'] = df['pump_fillage'].rolling('3D', min_periods=3).mean()
    
    if 'injection_rate' in df.columns:
        df['injection_avg_7d'] = df['injection_rate'].rolling('7D', min_periods=MIN_PERIODS).mean()
    
    if 'tubing_pressure' in df.columns:
        df['tp_avg'] = df['tubing_pressure'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['tp_std'] = df['tubing_pressure'].rolling('3D', min_periods=3).std()

    if 'discharge_pressure' in df.columns:
        df['dp_avg'] = df['discharge_pressure'].rolling('30D', min_periods=MIN_PERIODS).mean()

    if 'motor_current' in df.columns:
        df['amps_avg'] = df['motor_current'].rolling('30D', min_periods=MIN_PERIODS).mean()

    # Lagged baselines for ENERGY_DIV: at time t use baseline as of t-1 day so the
    # current (possibly anomalous) day does not dilute the baseline.
    if 'oil_volume' in df.columns:
        roll_oil = df['oil_volume'].rolling('90D', min_periods=MIN_PERIODS).mean()
        df['oil_baseline_90d_1d_ago'] = roll_oil.reindex(roll_oil.index - pd.Timedelta(days=1)).set_axis(roll_oil.index)
    if 'motor_current' in df.columns:
        roll_amps = df['motor_current'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['amps_avg_1d_ago'] = roll_amps.reindex(roll_amps.index - pd.Timedelta(days=1)).set_axis(roll_amps.index)
    
    if 'strokes_per_minute' in df.columns:
        df['spm_avg'] = df['strokes_per_minute'].rolling('30D', min_periods=MIN_PERIODS).mean()

    df = df.reset_index()
    return df
