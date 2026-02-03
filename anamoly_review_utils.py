import os
import json
import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from snowflake.connector import connect as snowflake_connect
import psycopg2
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnomalyReviewUtils')

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
OIL_PRICE = 75.00
WATER_DISPOSAL_COST = 1.50
GAS_PRICE = 2.50

DETECTION_WINDOW_DAYS = 5
TRAINING_CUTOFF_DAYS = 7
ROLLING_BASELINE_DAYS = 90

# SENSOR METADATA (Future-Proofing for UI/LLM)
SENSOR_DISPLAY_NAMES = {
    'oil_volume': 'Oil Volume',
    'water_volume': 'Water Volume',
    'gas_volume': 'Gas Volume',
    'strokes_per_minute': 'Strokes Per Minute (SPM)',
    'torque': 'Torque',
    'pump_fillage': 'Pump Fillage',
    'tubing_pressure': 'Tubing Pressure',
    'pump_intake_pressure': 'Pump Intake Pressure',
    'casing_pressure': 'Casing Pressure',
    'wellhead_temp': 'Wellhead Temperature',
    'motor_current': 'Motor Current',
    'discharge_pressure': 'Discharge Pressure',
    'intake_fluid_temp': 'Intake Fluid Temperature',
    'injection_rate': 'Injection Rate',
    'injection_pressure': 'Injection Pressure',
    'surface_pressure': 'Surface Pressure',
    'vibration_x': 'Vibration X',
    'vibration_y': 'Vibration Y',
    'downhole_net_stroke': 'Downhole Net Stroke'
}

SEVERITY_THRESHOLDS = {
    # Physics Thresholds (Score based, not $)
    "FLUID_POUND": {"critical": 90, "high": 80, "moderate": 50}, 
    "PUMP_WEAR": {"critical": 30, "high": 20, "moderate": 10},   
    "GAS_HOG": {"critical": 50, "high": 30, "moderate": 15},     
    # Statistical Thresholds
    "EFFICIENCY_DEGRADATION": {"critical": 50, "high": 30, "moderate": 15},
    "NEW_PRODUCTION_BASELINE": {"critical": 60, "high": 40, "moderate": 25},
    "MONTHLY_DECLINE": {"critical": 50, "high": 30, "moderate": 20},
    "PRESSURE_INSTABILITY": {"critical": 60, "high": 40, "moderate": 22},
    "PATTERN_SHIFT": {"critical": 500, "high": 200, "moderate": 50},
    "GHOST_PROD": {"critical": 100, "high": 95, "moderate": 90},
}

def get_snowflake_conn():
    return snowflake_connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )

def get_db_url():
    url = os.getenv("DATABASE_URL")
    if not url: return None
    if "schema=" in url:
        return url.replace("?schema=public", "").replace("&schema=public", "")
    return url

def get_openai_client():
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

def format_number(value, metric_name=None):
    if pd.isna(value): return "N/A"
    try:
        val_float = float(value)
    except:
        return str(value)

    # RULE 1: Exception for values < 1 (Use 1 decimal place)
    # Example: 0.5 bbl
    if abs(val_float) < 1 and val_float != 0:
        return f"{val_float:,.1f}"
        
    # RULE 2: Exception for Strokes Per Minute (Use 1 decimal place)
    # Example: 4.5 SPM
    if metric_name and ('stroke' in str(metric_name).lower() or 'spm' in str(metric_name).lower()):
        return f"{val_float:,.1f}"
        
    # RULE 3: Universal Standard (Comma delimited, No decimals)
    # Example: 1,000 or 300,000,000 or 46
    return f"{val_float:,.0f}"

# =============================================================================
# DATA FETCHING & PROCESSING
# =============================================================================

def fetch_well_data(well_id: str, days: int = None) -> pd.DataFrame:
    """
    FUTURE-PROOF FETCH:
    Uses 'SELECT s.*' to automatically grab ANY new sensors added to the DB.
    """
    # Fix: Use parameterized queries to prevent SQL Injection
    params = [well_id]
    where_clause = ""
    
    if days:
        where_clause = "AND s.timestamp >= %s"
        # Calculate date in Python to avoid raw string injection
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
            
        # Normalize columns to lowercase immediately to handle Snowflake UPPERCASE
        df.columns = [c.lower() for c in df.columns]
        
        # Handle timestamp variants
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        elif 'event_ts' in df.columns:
             df.rename(columns={'event_ts': 'timestamp'}, inplace=True)
             df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Safe forward fill for standard operational sensors only.
        operational_cols = ['motor_current', 'tubing_pressure', 'casing_pressure', 'strokes_per_minute', 'discharge_pressure', 'pump_intake_pressure']
        for c in operational_cols:
            if c in df.columns:
                df[c] = df[c].ffill().fillna(0)
        
        # Production data defaults to 0 if missing
        for prod_col in ['oil_volume', 'water_volume', 'gas_volume']:
            if prod_col in df.columns:
                df[prod_col] = df[prod_col].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Fetch failed for {well_id}: {e}")
        return pd.DataFrame()

def get_recent_anomaly_history(well_id: str, anomaly_code: str, exclude_date: str = None) -> str:
    query = """
    SELECT event_date, severity, title 
    FROM anomaly_review 
    WHERE well_id = %s 
    AND anomaly_code = %s 
    ORDER BY event_date DESC
    LIMIT 5
    """
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(query, (well_id, anomaly_code)).fetchall()
            
        if not rows:
            return "No prior history found."
            
        history_list = []
        for r in rows:
            # Snowflake dates return as objects, cast safely
            r_date = str(r[0])
            if exclude_date and r_date == str(exclude_date):
                continue
            history_list.append(f"- Date: {r_date}, Severity: {r[1]}, Title: {r[2]}")
        
        if not history_list:
            return "No prior history found."
            
        return "\n".join(history_list)
    except Exception as e:
        logger.warning(f"History fetch failed: {e}")
        return "History check unavailable."

def calculate_rolling_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp').sort_index()
    
    # Fix: Use 7 days min_periods to prevent noisy data on new wells (was 1)
    MIN_PERIODS = 7

    # 1. Standard Production Baselines (If columns exist)
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

    # 2. Physics Baselines
    if 'pump_fillage' in df.columns:
        df['fillage_avg_24h'] = df['pump_fillage'].rolling('24h', min_periods=1).mean()
    
    if 'injection_rate' in df.columns:
        df['injection_avg_7d'] = df['injection_rate'].rolling('7D', min_periods=MIN_PERIODS).mean()
        
    if 'tubing_pressure' in df.columns:
        df['tp_avg'] = df['tubing_pressure'].rolling('30D', min_periods=MIN_PERIODS).mean()
        df['tp_std'] = df['tubing_pressure'].rolling('3D', min_periods=3).std()

    if 'motor_current' in df.columns:
        df['amps_avg'] = df['motor_current'].rolling('30D', min_periods=MIN_PERIODS).mean()
        
    if 'strokes_per_minute' in df.columns:
        df['spm_avg'] = df['strokes_per_minute'].rolling('30D', min_periods=MIN_PERIODS).mean()

    df = df.reset_index()
    return df

def get_or_train_model(well_id: str, df: pd.DataFrame) -> Dict:
    # Use generic features that are commonly available for efficiency calculation
    features = ['tubing_pressure', 'motor_current', 'strokes_per_minute', 'casing_pressure', 'discharge_pressure']
    # Fix: Ensure at least 2 features exist and are not constant
    available = [f for f in features if f in df.columns and df[f].std() > 0]
    
    if len(df) < 30 or len(available) < 2: return {}
    if 'oil_volume' not in df.columns: return {}
    
    # Fix: Only train on days with actual production to avoid learning shutdown patterns as "efficient"
    # --- FIX: TRAINING CUTOFF ---
    # Do NOT train on the last 7 days. This prevents the model from "learning" 
    # the crash as normal behavior. It ensures predicted_oil remains high (300) 
    # while actual oil drops (15), creating the efficiency gap we need.
    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=7)
    train_df = df[(df['oil_volume'] > 0.1) & (df['timestamp'] < cutoff_date)]
    
    # Fallback for new wells with very little history
    if len(train_df) < 30:
        train_df = df[df['oil_volume'] > 0.1]
    if len(train_df) < 30: return {}

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    try:
        X = scaler.fit_transform(imputer.fit_transform(train_df[available]))
        model.fit(X, train_df['oil_volume'])
        return {'model': model, 'scaler': scaler, 'imputer': imputer, 'features': available}
    except Exception as e:
        logger.error(f"Model training failed during get_or_train_model: {e}")
        return {}

# =============================================================================
# PHYSICS ENGINE & DIAGNOSIS
# =============================================================================

def diagnose_mechanics(ctx: Dict) -> Dict:
    """
    Fallback Diagnosis for Standard Checks.
    Guesses the root cause based on correlations if a specific Physics Check didn't catch it.
    """
    amps = ctx.get('motor_current', 0)
    amps_avg = ctx.get('amps_avg', 0)
    pressure = ctx.get('tubing_pressure', 0)
    pressure_avg = ctx.get('tp_avg', 0)
    # Fix: Use correct column name 'oil_volume'
    oil = ctx.get('oil_volume', 0)
    pred_oil = ctx.get('predicted_oil', 1)
    
    # Fix: Avoid division by zero
    efficiency = (oil / pred_oil) * 100 if pred_oil > 0.1 else 100
    amps_ratio = amps / amps_avg if amps_avg > 0.1 else 1.0
    pressure_ratio = pressure / pressure_avg if pressure_avg > 0.1 else 1.0
    
    if amps_ratio < 0.6: 
        return {
            'type': 'Parted Rods or Belt Break', 
            'confidence': 'High', 
            'indicators': f'Motor Current dropped significantly ({amps_ratio:.2f}x)'
        }

    # 2. Seized Pump / Locked Rotor (High Confidence) - NEW
    # High Amps + Low Efficiency (Pump is fighting but moving nothing)
    elif amps_ratio > 1.5 and efficiency < 20:
        return {
            'type': 'Seized Pump / Locked Rotor',
            'confidence': 'High',
            'indicators': f'High Amps ({amps_ratio:.1f}x) with near-zero flow'
        }

    # 3. Flowline Blockage (High Confidence)
    elif pressure_ratio > 1.3:
        return {
            'type': 'Flowline Blockage / High Line Pressure',
            'confidence': 'High',
            'indicators': f'Tubing Pressure elevated ({pressure_ratio:.1f}x)'
        }

    # 4. Tubing Leak (Moderate Confidence)
    # Normal Amps + Low Efficiency
    elif (0.9 < amps_ratio < 1.1) and efficiency < 50:
        return {
            'type': 'Tubing Leak',
            'confidence': 'Moderate',
            'indicators': f'Normal Amps but low efficiency ({efficiency:.0f}%)'
        }
        
    # 5. Pump Wear (Moderate Confidence)
    # Slightly Low Amps + Low Efficiency
    elif (0.7 <= amps_ratio <= 0.9) and efficiency < 60:
        return {
            'type': 'Pump Wear / Clearance Issues',
            'confidence': 'Moderate',
            'indicators': f'Lower load and low efficiency'
        }

    else:
        return {
            'type': 'Operational Volatility / Data Variance',
            'confidence': 'Low',
            'indicators': f'Inconsistent readings: Eff {efficiency:.0f}%, Amps {amps_ratio:.2f}x'
        }

def determine_severity(anomaly_code: str, metrics: Dict) -> str:
    """
    Severity is now Score-Based (How weird is it?), not Money-Based.
    """
    thresholds = SEVERITY_THRESHOLDS.get(anomaly_code, {"high": 999, "moderate": 999})
    
    # 1. Physics Score Priority
    if 'severity_score' in metrics:
        score = metrics['severity_score']
        # Fix: Use .get to safely access thresholds
        if score >= thresholds.get('critical', 999): return "Critical"
        if score >= thresholds.get('high', 999): return "High"
        if score >= thresholds.get('moderate', 999): return "Moderate"
        return "Low"
    
    # 2. Percentage Drop Priority
    for k in ['efficiency_drop', 'decline_pct', 'volatility_pct', 'frequency_increase_pct']:
        if k in metrics:
            val = abs(metrics[k])
            if val >= thresholds.get('critical', 999): return "Critical"
            if val >= thresholds.get('high', 999): return "High"
            if val >= thresholds.get('moderate', 999): return "Moderate"
            return "Low"

    return "Moderate" # Default

# =============================================================================
# STRICT PHYSICS CHECKERS (Super Important - Abort if Missing)
# =============================================================================

def check_fluid_pound(ctx: Dict) -> Optional[Dict]:
    """Rod Pump: Detects Fluid Pound (Shock Loading)."""
    # Fix: Ensure check only runs on Rod Pumps
    lift_type = str(ctx.get('lift_type', '')).lower().replace(' ', '')
    if 'rod' not in lift_type: return None

    if pd.isna(ctx.get('pump_fillage')) or pd.isna(ctx.get('strokes_per_minute')): return None
    
    fillage = ctx['pump_fillage']
    spm = ctx['strokes_per_minute']
    
    if spm > 4.0 and fillage < 80.0:
        severity_score = 100 - fillage
        revenue_loss = ctx.get('oil_avg_7d', 0) * ((100 - fillage) / 100) * OIL_PRICE
        
        # BOE Logic
        loss_pct = (100 - fillage) / 100
        oil_comp = ctx.get('oil_avg_7d', 0) * loss_pct
        gas_comp = ctx.get('gas_avg_7d', 0) * loss_pct
        boe_loss = oil_comp + (gas_comp / 6)
        
        data_summary = f"""
        DETECTION TYPE: Rod Pump Fluid Pound
        SENSORS: Pump Fillage ({format_number(fillage)}%), SPM ({format_number(spm, "spm")})
        ROOT CAUSE: The pump is not filling completely on the downstroke, causing shock load.
        """
        
        return {
            "code": "FLUID_POUND",
            "category": "OPERATIONAL",
            "raw_data": data_summary,
            "severity_metrics": {'severity_score': severity_score, 'impact_value': revenue_loss},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
            },
            "chart_metric": "efficiency_trend",
            "diagnosis": {
                "type": "Fluid Pound (Gas Interference/Starvation)",
                "confidence": "High",
                "indicators": f"Pump Fillage {format_number(fillage)}% vs SPM {format_number(spm, 'spm')}"
            }

        }
    return None

def check_pump_wear(ctx: Dict) -> Optional[Dict]:
    """ESP: Detects Pump Wear (Head Degradation)."""
    # Fix: Ensure check only runs on ESPs
    lift_type = str(ctx.get('lift_type', '')).lower().replace(' ', '')
    if 'esp' not in lift_type: return None

    req = ['discharge_pressure', 'pump_intake_pressure', 'motor_current']
    if any(pd.isna(ctx.get(k)) for k in req): return None
    
    head = ctx['discharge_pressure'] - ctx['pump_intake_pressure']
    amps = ctx['motor_current']
    
    if head < 800 and amps > 30:
        revenue_loss = 2000.0
        # BOE Logic
        boe_loss = revenue_loss / OIL_PRICE
        
        data_summary = f"""
        DETECTION TYPE: ESP Head Degradation
        SENSORS: TDH ({format_number(head)} psi), Amps ({format_number(amps)})
        ROOT CAUSE: Pump is running but generating insufficient lift.
        """
        return {
            "code": "PUMP_WEAR",
            "category": "OPERATIONAL",
            "raw_data": data_summary,
            "severity_metrics": {'severity_score': 80, 'impact_value': revenue_loss},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
            },
            "chart_metric": "pressure_trend",
            "diagnosis": {
                "type": "ESP Pump Wear",
                "confidence": "Moderate",
                "indicators": f"Low Head ({format_number(head)} psi) with Active Motor"
            }
        }
    return None

def check_gas_injection_inefficiency(ctx: Dict) -> Optional[Dict]:
    """Gas Lift: Detects Gas Hog."""
    # Fix: Ensure check only runs on Gas Lift
    lift_type = str(ctx.get('lift_type', '')).lower().replace(' ', '')
    if 'gas' not in lift_type: return None

    req = ['injection_rate', 'casing_pressure', 'tubing_pressure']
    if any(pd.isna(ctx.get(k)) for k in req): return None
    
    rate = ctx['injection_rate']
    cp = ctx['casing_pressure']
    tp = ctx['tubing_pressure']
    
    if rate > 100 and (cp - tp) < 50:
        gas_cost = rate * GAS_PRICE
        data_summary = f"""
        DETECTION TYPE: Gas Lift Inefficiency
        SENSORS: Rate ({format_number(rate)}), Diff ({format_number(cp-tp)})
        ROOT CAUSE: Gas is cycling without lifting fluid efficiently.
        """
        return {
            "code": "GAS_HOG",
            "category": "PROCESS",
            "raw_data": data_summary,
            "severity_metrics": {'severity_score': min(rate/10, 100), 'impact_value': gas_cost},
            "impact_value": gas_cost,
            "impact_payload": {
                "financial": {"value": gas_cost, "label": f"${format_number(gas_cost)}/day cost"},
                "production": {"value": 0, "label": "0 BOE/day"}
            },
            "chart_metric": "pressure_trend",
            "diagnosis": {
                "type": "Inefficient Gas Injection",
                "confidence": "High",
                "indicators": f"High Rate ({format_number(rate)}) with minimal casing differential"
            }
        }
    return None

# =============================================================================
# STANDARD CHECKERS
# =============================================================================

def check_mechanical_failure(ctx: Dict) -> Optional[Dict]:
    """
    Detects catastrophic failure: High Energy Input (Amps) + Zero/Low Output.
    """
    # 1. Get Baselines (Use 90d baseline as it's more stable than predicted_oil)
    oil_current = ctx.get('oil_volume', 0)
    oil_baseline = ctx.get('oil_baseline_90d', 1) # Prevent div/0
    
    amps_current = ctx.get('motor_current', 0)
    amps_avg = ctx.get('amps_avg', 1)

    # 2. Logic: Amps Exploded (>150% baseline) AND Production Crashed (<20% baseline)
    # This is the signature of a stuck pump, seized rotor, or parted rods (if amps low)
    
    # CASE A: SEIZED / STUCK (High Amps, No Oil)
    if amps_current > (1.5 * amps_avg) and oil_current < (0.2 * oil_baseline):
        oil_loss = oil_baseline
        loss = oil_baseline * OIL_PRICE
        revenue_loss = loss # Maintain variable name consistency if needed, but explicit formula is key
        
        # BOE Logic
        gas_loss = ctx.get('gas_baseline_90d', 0)
        boe_loss = oil_loss + (gas_loss / 6)
        
        return {
            "code": "MECHANICAL_FAILURE",
            "category": "OPERATIONAL",
            "raw_data": f"CRITICAL FAILURE: Amps {format_number(amps_current)} (High) vs Oil {format_number(oil_current)} bbl (Low)",
            "severity_metrics": {'severity_score': 100, 'impact_value': revenue_loss},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
            },
            "chart_metric": "mechanical_failure_view",
            "diagnosis": {
                "type": "Seized Pump / Downhole Failure",
                "confidence": "High",
                "indicators": f"Motor load {format_number(amps_current/amps_avg, 'ratio')}x normal with {format_number(oil_current/oil_baseline*100)}% production capture."
            }
        }
    return None

def check_sustained_production_change(ctx: Dict) -> Optional[Dict]:
    base = ctx.get('oil_baseline_90d', 0)
    curr = ctx.get('oil_avg_7d', 0)
    
    if base > 5:
        change_pct = ((base - curr) / base) * 100
        if change_pct > 25:
            revenue_loss = (base - curr) * OIL_PRICE
            
            # BOE Logic
            oil_comp = (base - curr)
            gas_base = ctx.get('gas_baseline_90d', 0)
            gas_curr = ctx.get('gas_avg_7d', 0)
            gas_comp = max(0, gas_base - gas_curr)
            boe_loss = oil_comp + (gas_comp / 6)

            return {
                "code": "NEW_PRODUCTION_BASELINE",
                "category": "PRODUCTION",
                "raw_data": f"Prod dropped {format_number(change_pct)}% from 90d baseline",
                "severity_metrics": {'decline_pct': change_pct, 'impact_value': revenue_loss},
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                    "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
                },
                "chart_metric": "baseline_shift",
                "diagnosis": diagnose_mechanics(ctx)
            }
    return None

def check_efficiency_degradation(ctx: Dict) -> Optional[Dict]:
    pred = ctx.get('predicted_oil', 0) # Fix: use 'predicted_oil' from service
    actual = ctx.get('oil_volume', 0) # Fix: use correct column name
    
    if pred > 5 and actual >= 0:
        eff = (actual / pred) * 100
        avg_eff = ctx.get('efficiency_avg_30d', 100)
        drop = avg_eff - eff
        
        if drop > 15:
            revenue_loss = (pred - actual) * OIL_PRICE
            
            # BOE Logic
            boe_loss = (pred - actual)

            return {
                "code": "EFFICIENCY_DEGRADATION",
                "category": "OPERATIONAL",
                "raw_data": f"Efficiency dropped {format_number(drop)}% (Current: {format_number(eff)}%)",
                "severity_metrics": {'efficiency_drop': drop, 'impact_value': revenue_loss},
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                    "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
                },
                "chart_metric": "efficiency_trend",
                "diagnosis": diagnose_mechanics(ctx)
            }
    return None

def check_pressure_instability(ctx: Dict) -> Optional[Dict]:
    """
    Detects Volatility/Slugging, but IGNORES massive step-changes (Crashes).
    """
    tp_avg = ctx.get('tp_avg', 0)
    tp_std = ctx.get('tp_std', 0)
    tp_current = ctx.get('tubing_pressure', 0)
    
    if tp_avg > 50:
        volatility_pct = (tp_std / tp_avg) * 100
        
        # --- FIX: IGNORE STEP CHANGES ---
        # If the current pressure is wildly different from the average (>50% shift),
        # it's a "Crash" or "New Baseline", NOT "Instability".
        # Instability oscillates AROUND the average.
        deviation_from_mean = abs(tp_current - tp_avg) / tp_avg
        if deviation_from_mean > 0.5: 
            return None # Let other checkers (Mechanical Failure) handle this.

        # Standard Logic
        if volatility_pct > 22 and tp_std > 40:
            revenue_loss = ctx.get('oil_volume', 0) * 0.10 * OIL_PRICE

            # BOE Logic
            oil_comp = ctx.get('oil_volume', 0) * 0.10
            gas_comp = ctx.get('gas_volume', 0) * 0.10
            boe_loss = oil_comp + (gas_comp / 6)

            return {
                "code": "PRESSURE_INSTABILITY",
                "category": "OPERATIONAL",
                "raw_data": f"Pressure Volatility {format_number(volatility_pct)}%",
                "severity_metrics": {'volatility_pct': volatility_pct, 'impact_value': revenue_loss},
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                    "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
                },
                "chart_metric": "pressure_trend",
                "diagnosis": {"type": "Slug Flow / Instability", "confidence": "Moderate", "indicators": f"High Volatility {format_number(volatility_pct)}%"}
            }
    return None

def check_ghost_production(ctx: Dict) -> Optional[Dict]:
    is_active = (ctx.get('strokes_per_minute',0) > 0.1) or (ctx.get('motor_current',0) > 5)
    # Fix: Use correct column names
    total_vol = ctx.get('oil_volume',0) + ctx.get('water_volume',0)
    
    # Fix: Prevent false positives on stripper wells (check baseline)
    baseline_vol = ctx.get('oil_baseline_90d', 1)
    
    if is_active and total_vol < 0.05 and baseline_vol > 0.5:
        revenue_loss = ctx.get('oil_avg_30d', 0) * OIL_PRICE
        
        # BOE Logic
        oil_comp = ctx.get('oil_avg_30d', 0)
        gas_comp = ctx.get('gas_avg_30d', 0)
        boe_loss = oil_comp + (gas_comp / 6)

        return {
            "code": "GHOST_PROD",
            "category": "OPERATIONAL",
            "raw_data": "Pump Active, Zero Volume",
            "severity_metrics": {'impact_value': revenue_loss, 'severity_score': 100},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
            },
            "chart_metric": "production_bar",
            "diagnosis": {"type": "Meter Failure or Parted Rods", "confidence": "High", "indicators": "Active Pump, 0 Vol"}
        }
    return None

def check_operational_pattern_shift(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    if 'timestamp' not in ctx: return None
    current_timestamp = ctx['timestamp']
    max_timestamp = df['timestamp'].max()
    if current_timestamp != max_timestamp: return None
    
    violation_rules = {
        'tubing_pressure': {'min': 50, 'max': 2000, 'label': 'Tubing Pressure Spikes'},
        'motor_current': {'min': 5, 'max': 200, 'label': 'Motor Current Anomalies'}
    }
    
    for sensor, rules in violation_rules.items():
        if sensor not in df.columns: continue
        
        sensor_data = df[[sensor, 'timestamp']].copy()
        sensor_data = sensor_data[sensor_data[sensor] > 0]
        if len(sensor_data) < 60: continue
        
        sensor_data['violation'] = ((sensor_data[sensor] < rules['min']) | (sensor_data[sensor] > rules['max'])).astype(int)
        
        historical_cutoff = df['timestamp'].max() - pd.Timedelta(days=90)
        recent_cutoff = df['timestamp'].max() - pd.Timedelta(days=30)
        
        historical_rate = sensor_data[sensor_data['timestamp'] < historical_cutoff]['violation'].mean() * 100
        recent_rate = sensor_data[sensor_data['timestamp'] >= recent_cutoff]['violation'].mean() * 100
        
        if recent_rate > 10 and recent_rate > (historical_rate * 2.5):
            frequency_increase_pct = ((recent_rate - historical_rate) / max(historical_rate, 1)) * 100
            
            # Fix: Use correct column name 'oil_volume'
            revenue_loss = ctx.get('oil_volume', 0) * 0.15 * OIL_PRICE
            
            # BOE Logic (Derived)
            oil_component = ctx.get('oil_volume', 0) * 0.15
            gas_component = ctx.get('gas_volume', 0) * 0.15
            boe_loss = oil_component + (gas_component / 6)

            data_summary = f"Operational Pattern Shift - {rules['label']}\nFrequency Increase: {format_number(frequency_increase_pct)}%"
            
            return {
                "code": "PATTERN_SHIFT",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity_metrics": {'frequency_increase_pct': frequency_increase_pct, 'impact_value': revenue_loss},
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                    "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
                },
                "chart_metric": "sensor_violations",
                "diagnosis": {"type": "Increased Operational Violations", "confidence": "Moderate", "indicators": f"Frequency of {rules['label']} increased by {format_number(frequency_increase_pct)}%"}
            }
    return None

def check_monthly_production_decline(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    if 'timestamp' not in df.columns or df['timestamp'].empty: return None
    recent_df = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=120))].copy()
    recent_df['month'] = recent_df['timestamp'].dt.to_period('M')
    
    # Fix: Use correct column name 'oil_volume'
    monthly = recent_df.groupby('month').agg({'oil_volume': ['sum', 'count']}).reset_index()
    monthly.columns = ['month', 'oil_total', 'oil_days']
    monthly['oil_rate'] = monthly['oil_total'] / monthly['oil_days']
    
    if len(monthly) >= 2:
        latest = monthly.iloc[-1]
        previous = monthly.iloc[-2]
        
        if latest['oil_days'] < 3: return None

        if previous['oil_rate'] > 5:
            decline_pct = ((previous['oil_rate'] - latest['oil_rate']) / previous['oil_rate']) * 100
            
            if decline_pct > 15:
                revenue_loss = max(0, previous['oil_rate'] - latest['oil_rate']) * OIL_PRICE
                
                # BOE Logic
                boe_loss = max(0, previous['oil_rate'] - latest['oil_rate'])
                
                data_summary = f"Monthly Decline: {format_number(decline_pct)}%"
                
                return {
                    "code": "MONTHLY_DECLINE",
                    "category": "PRODUCTION",
                    "raw_data": data_summary,
                    "severity_metrics": {'decline_pct': decline_pct, 'impact_value': revenue_loss},
                    "impact_value": revenue_loss,
                    "impact_payload": {
                        "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                        "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
                    },
                    "chart_metric": "monthly_production",
                    "diagnosis": {"type": "Accelerated Monthly Decline", "confidence": "Low", "indicators": "Comparison of monthly production rates."}
                }
    return None

# =============================================================================
# LLM GENERATION
# =============================================================================

def generate_llm_content(anomaly_data: str, lift_type: str, chart_metric: str, historical_context: str, detected_at_str: str, diagnosis: Dict = None, impact_metrics: Dict = None) -> Dict:
    # Fix: Clean NaN strings to prevent LLM hallucinations
    anomaly_data = str(anomaly_data).replace('nan', 'Unknown')
    if diagnosis:
        diagnosis = {k: str(v).replace('nan', 'Unknown') for k, v in diagnosis.items()}

    try:
        client = get_openai_client()
        if not client:
            raise Exception("No OpenAI Client")
        
        try:
            if 'T' in str(detected_at_str):
                dt_obj = datetime.fromisoformat(str(detected_at_str).replace('Z', '+00:00'))
            else:
                dt_obj = datetime.strptime(str(detected_at_str), '%Y-%m-%d %H:%M:%S.%f')
            formatted_date = dt_obj.strftime('%m/%d/%Y %H:%M:%S')
        except:
            formatted_date = str(detected_at_str)

        # Handling "Low Confidence" diagnoses in the prompt
        confidence_instruction = ""
        diagnosis_context = ""
        
        if diagnosis:
            if diagnosis.get('confidence') == 'Low':
                diagnosis_context = f"DIAGNOSIS (LOW CONFIDENCE): {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                NOTE: The root cause confidence is LOW. 
                - In 'suspected_root_cause', state "Possible causes include [Diagnosis] depending on validation."
                - DO NOT use the word "Unconfirmed" in the title. Treat the anomaly as valid but needing review.
                - Emphasize the need for further field validation in the description.
                """
            else:
                diagnosis_context = f"CONFIRMED DIAGNOSIS: {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                - Use ONLY declarative statements ("The root cause is...").
                - NO hedging ("might", "could").
                - Be confident and direct in all fields.
                """
        
        # Prepare financial context string
        impact_context = ""
        if impact_metrics:
            fin = impact_metrics.get('financial', {})
            prod = impact_metrics.get('production', {})
            impact_context = f"""
            CALCULATED IMPACT METRICS (Use EXACLTY these values, DO NOT re-calculate or guess):
            - Financial Loss: {fin.get('label', 'N/A')} (Value: {fin.get('value', 0)})
            - Production Loss: {prod.get('label', 'N/A')} (Value: {prod.get('value', 0)})
            """

        prompt = f"""
ROLE: Senior Production Engineer with 20+ years in oil & gas operations, specializing in rod pump systems and anomaly detection.
TASK: Generate a detailed, professional report on a detected well anomaly. The report must be factual, concise, and actionable for field operators and engineers. Use technical terminology appropriately but explain implications clearly.

{confidence_instruction}

ANOMALY DATA (raw metrics and indicators - use these exactly in your report):
{anomaly_data}

{impact_context}

DIAGNOSIS (root cause assessment - integrate this into the report):
{diagnosis_context}

HISTORICAL CONTEXT (past similar events - reference in historical_context field):
{historical_context}

LIFT TYPE: {lift_type} (tailor explanations to this, e.g., rod pump specifics like SPM, amps, tubing pressure correlations)

CHART METRIC: {chart_metric} (mention relevant visualization in risk_analysis if applicable)

DETECTED AT: {formatted_date}

NUMERICAL CONFORMITY STANDARDS (STRICTLY FOLLOW):
1. COMMA DELIMITED: Always use commas for thousands (e.g., 1,500 or 300,000).
2. NO DECIMALS: Round all metrics and currency to the nearest whole number (e.g., 46 Amps, $3,500).
3. EXCEPTIONS (Use exactly 1 decimal place):
   - Strokes Per Minute (e.g., 4.2 SPM)
   - Values less than 1 (e.g., 0.5 bbl)

IMPORTANT GUIDELINES FOR ALL FIELDS:
- ALWAYS explicitly name the primary sensor(s) involved (e.g., "Tubing Pressure", "Motor Current", "Strokes Per Minute") in the description, why_is_this_an_anomaly, and suspected_root_cause.
- SENSOR NAMING: Never use snake_case (e.g. oil_volume) in the output text. Always convert to Title Case (e.g. Oil Volume).
- If volatility, deviation, drop, or other percentages are in ANOMALY DATA, PROMINENTLY INCLUDE THE EXACT NUMBERS.
- Prioritize key metrics: volatility % over small mean deviations for instability; efficiency drop % for degradation; decline % for production changes.
- Use industry-specific language: reference rod pump dynamics, gas interference, pump off, parted rods, etc., where relevant.
- Ensure the report is actionable.
- Financial impacts: You MUST use the 'CALCULATED_REVENUE_LOSS' provided in the data. Do NOT calculate it yourself.

Generate a JSON response with these EXACT keys and STRICT formats:

1. "title": 
   - If High Confidence Diagnosis: Use diagnosis type (e.g., "Tubing Leak Detected").
   - If Low Confidence: Use the observed symptom (e.g., "Efficiency Drop Detected").
   - Max 6 words. No $ symbols. Make it attention-grabbing and specific.

2. "description": 
   - Write EXACTLY 3-4 sentences.
   - FIRST SENTENCE: State current value vs baseline with specific numbers and sensor name.
     Example: "Tubing pressure is currently at 1166.0 psi, a 2.9% deviation below the 30-day average of 1200.1 psi, with volatility at 24.5%."
   - SECOND SENTENCE: Explain the physical/mechanical implication, naming sensors.
     Example: "This high tubing pressure volatility indicates gas interference disrupting pump fillage and efficiency."
   - THIRD SENTENCE: State the financial impact.
     * CRITICAL: Use the 'CALCULATED_REVENUE_LOSS' value from ANOMALY DATA.
     Example: "This instability is costing an estimated $[VALUE] per day in lost oil production."
   - FOURTH SENTENCE (optional): Add operational context or recommendation.
     Example: "Immediate review of downhole conditions recommended to prevent escalation."

3. "why_is_this_an_anomaly": 
   - ONE clear sentence stating the deviation math, including sensor and key percentages.
   - Use format: "The [sensor/metric] shows [X value] with [Y% volatility/deviation/drop], compared to the historical average of [Z]."
   - Example: "The tubing pressure volatility of 24.5% is a 150% increase over normal (<10%), with current value 2.9% below the historical average of 1200.1 psi."

4. "suspected_root_cause": 
   - Return a SINGLE, DETAILED sentence (30-60 words).
   - If ROOT CAUSE DIAGNOSIS is provided, use this format EXACTLY:
     "The root cause is {{diagnosis.get('type') if diagnosis else 'mechanical failure'}}, confirmed by {{diagnosis.get('indicators') if diagnosis else 'operational data'}}, likely involving [expand with sensor correlations from ANOMALY DATA]."
   - Otherwise, infer most probable physical cause based on sensor correlations and lift type.
   - Include sensor names and numbers.
   - DO NOT provide a list.
   - Example: "The root cause is gas interference in the rod pump system, evidenced by tubing pressure volatility at 24.5% and elevated gas-to-oil ratio exceeding 10:1, leading to incomplete pump fillage and reduced stroke efficiency."

5. "economic_impact": 
   - ONE sentence in this format: "This [problem type, e.g., pressure instability] is resulting in an estimated daily revenue loss of $[X], based on current rates."
   - CRITICAL: You MUST extract $[X] from 'CALCULATED_REVENUE_LOSS' in ANOMALY DATA. Do not recalculate.
   - Example: "This pressure instability is resulting in an estimated daily revenue loss of $1,200, based on production impact at current rates."

6. "risk_analysis": 
   - You MUST follow this EXACT two-paragraph structure, no objects just string:
   
   PARAGRAPH 1 (Immediate Risk - Detection Statement):
   "The production anomaly [Insert specific anomaly type from title] was detected through continuous monitoring of [list key sensors, e.g., tubing pressure and motor current]. [Insert ONE sentence with the specific percentage drop or deviation, including numbers]. [Insert ONE sentence about immediate operational stability or safety concern - focus on equipment damage risk (e.g., pump seizure) or safety hazard (e.g., pressure buildup) if applicable, tailored to lift type]."
   
   PARAGRAPH 2 (Long-term Consequence):
   "Detected at: {formatted_date}
   Root cause: [Insert the EXACT root cause from field 4 above]
   
   [Insert 2-3 sentences detailing potential long-term consequences if left unaddressed. MUST MENTION the 'CALCULATED_BOE_LOSS' volume as production at risk. Focus on: permanent reservoir damage, equipment failure cascade (e.g., rod failure leading to well shutdown), compounded financial loss, regulatory non-compliance (e.g., emissions from gas venting), or safety escalation (e.g., blowout risk). Be specific about timelines and dollar amounts where possible, using provided impact_value]. This analysis is based on historical equipment performance data and industry failure progression patterns."

7. "historical_context": 
   - Return a JSON object with:
     {{
       "recurrence_status": "[First occurrence|Recurring issue - X previous events|Chronic problem - Y events in Z days]" (analyze from HISTORICAL CONTEXT),
       "history_timeline": [
         {{"date": "MM/DD/YYYY", "severity": "High|Moderate|Low", "title": "Previous event title"}},
         ...
       ] (limit to 5, format dates properly)
     }}

Return ONLY valid JSON. Ensure all fields are populated based on data provided.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM content generation failed: {e}")
        return {
            "title": "Anomaly Detected",
            "description": "An operational deviation was detected. Review sensor data manually. Impact unknown.",
            "why_is_this_an_anomaly": "Metrics deviate from baseline.",
            "suspected_root_cause": "Unknown - data analysis required.",
            "economic_impact": "Impact calculation unavailable.",
            "risk_analysis": "Review required for immediate and long-term risks.",
            "historical_context": {"recurrence_status": "Unknown", "history_timeline": []}
        }

# =============================================================================
# CHART GENERATION SYSTEM
# =============================================================================

def generate_chart_config(anomaly_data: str, chart_metric: str, df: pd.DataFrame, anomaly_code: str = None, severity_metrics: Dict = None) -> Dict:
    """
    Enhanced chart generation with anomaly-specific visualizations.
    """
    if df.empty:
        return create_fallback_chart(df, "No Data")

    try:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date_only'] = df['timestamp'].dt.date
        
        detection_date = extract_detection_date(anomaly_data)
        
        # Route to anomaly-specific chart generators
        if anomaly_code == "MECHANICAL_FAILURE":
            return generate_mechanical_failure_chart(df, anomaly_data, detection_date, severity_metrics)

        if chart_metric == "monthly_production":
            return generate_monthly_decline_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "baseline_shift":
            return generate_baseline_shift_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "efficiency_trend":
            return generate_efficiency_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "tubing_pressure" or chart_metric == "pressure_trend":
            return generate_pressure_instability_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "sensor_violations":
            return generate_pattern_shift_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "production_bar":
            return generate_ghost_production_chart(df, anomaly_data, detection_date, severity_metrics)
        else:
            return generate_default_enhanced_chart(df, anomaly_data, detection_date)
            
    except Exception as e:
        logger.error(f"Enhanced chart generation failed: {e}")
        return create_fallback_chart(df, str(e))


def extract_detection_date(anomaly_data: str) -> Optional[str]:
    match = re.search(r'(\d{4}-\d{2}-\d{2})', anomaly_data)
    return match.group(1) if match else None


def prepare_daily_aggregation(df: pd.DataFrame, days: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    start_date = df['timestamp'].max() - pd.Timedelta(days=days)
    recent_df = df[df['timestamp'] >= start_date].copy()
    
    # Dynamically Aggregating Available Columns
    agg_dict = {
        'oil_volume': 'mean',
        'water_volume': 'mean',
        'gas_volume': 'mean'
    }
    
    for col in ['tubing_pressure', 'motor_current', 'strokes_per_minute', 'pump_fillage', 'injection_rate', 'casing_pressure', 'predicted_oil']:
        if col in df.columns:
            agg_dict[col] = 'mean'
            
    daily_df = recent_df.groupby('date_only').agg(agg_dict).reset_index()
    
    daily_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in daily_df.columns]
    
    # Renaming back to simple names for chart ease
    for col in daily_df.columns:
        if col.endswith('_mean'):
            daily_df.rename(columns={col: col.replace('_mean', '')}, inplace=True)

    labels = [d.strftime('%Y-%m-%d') for d in daily_df['date_only']]
    
    return daily_df, labels


def create_anomaly_zone_annotation(detection_date: str, labels: List[str]) -> Dict:
    if not detection_date or detection_date not in labels:
        return {}
    
    detection_index = labels.index(detection_date)
    
    return {
        'type': 'line',
        'xMin': detection_index,
        'xMax': detection_index,
        'borderColor': '#ef4444',
        'borderWidth': 2,
        'borderDash': [5, 5],
        'label': {
            'display': True,
            'content': 'Anomaly Detected',
            'position': 'start',
            'backgroundColor': '#ef4444',
            'color': '#ffffff',
            'font': {'size': 11, 'weight': 'bold'}
        }
    }


def create_threshold_band(labels: List[str], threshold_value: float, label: str, color: str = '#10b981') -> Dict:
    return {
        'type': 'line',
        'yMin': threshold_value,
        'yMax': threshold_value,
        'borderColor': color,
        'borderWidth': 2,
        'borderDash': [8, 4],
        'label': {
            'display': True,
            'content': label,
            'position': 'end',
            'backgroundColor': color,
            'color': '#ffffff',
            'font': {'size': 10}
        }
    }

def generate_mechanical_failure_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    """
    Shows the divergence between Energy Input (Amps) and Production Output (Oil).
    Visually proves 'Seized Pump' (High Amps, Low Oil).
    """
    daily_df, labels = prepare_daily_aggregation(df, days=30)
    
    oil_data = daily_df['oil_volume'].fillna(0).round(1).tolist()
    
    # Check if we have Amps, otherwise fallback
    if 'motor_current' in daily_df.columns:
        sensor_data = daily_df['motor_current'].fillna(0).round(1).tolist()
        sensor_label = "Motor Current (Amps)"
        sensor_color = "#ef4444" # Red for danger
    else:
        sensor_data = daily_df.get('tubing_pressure', [0]*len(labels)).tolist()
        sensor_label = "Tubing Pressure (PSI)"
        sensor_color = "#f59e0b"

    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Critical Failure Analysis: Input vs Output', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': 'High energy input with collapsed production indicates mechanical seizure.', 'font': {'size': 12}, 'color': '#ef4444'}
    
    # Left Axis: Oil
    options['scales']['y'] = {
        'type': 'linear', 'position': 'left', 
        'title': {'display': True, 'text': 'Oil Production (bbl/day)'},
        'grid': {'color': 'rgba(200, 200, 200, 0.1)'}
    }
    # Right Axis: Amps
    options['scales']['y1'] = {
        'type': 'linear', 'position': 'right', 
        'title': {'display': True, 'text': sensor_label},
        'grid': {'drawOnChartArea': False}
    }

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Oil Production", 
                    "data": oil_data, 
                    "borderColor": "#3b82f6", 
                    "backgroundColor": "rgba(59, 130, 246, 0.2)", 
                    "borderWidth": 2, 
                    "fill": True, 
                    "yAxisID": "y"
                },
                {
                    "label": sensor_label, 
                    "data": sensor_data, 
                    "borderColor": sensor_color, 
                    "borderWidth": 3, 
                    "yAxisID": "y1",
                    "tension": 0.1 # Sharp lines for sudden spikes
                }
            ]
        },
        "options": options
    }

# =============================================================================
# ANOMALY-SPECIFIC CHART GENERATORS
# =============================================================================

def generate_efficiency_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    
    if 'efficiency' not in daily_df.columns and 'predicted_oil' in daily_df.columns and 'oil_volume' in daily_df.columns:
        daily_df['efficiency'] = (daily_df['oil_volume'] / daily_df['predicted_oil'].replace(0, 1)) * 100
        daily_df['efficiency'] = daily_df['efficiency'].clip(0, 120)
    
    actual_oil = daily_df['oil_volume'].fillna(0).round(1).tolist()
    predicted_oil = daily_df.get('predicted_oil', daily_df['oil_volume']).fillna(0).round(1).tolist()
    efficiency = daily_df.get('efficiency', [100] * len(labels)).fillna(100).round(1).tolist()
    
    # If Fluid Pound, we prefer Fillage over Efficiency if available
    if 'FLUID POUND' in anomaly_data.upper() and 'pump_fillage' in daily_df.columns:
        efficiency = daily_df['pump_fillage'].fillna(0).round(1).tolist()
        title_text = 'Pump Fillage & Production Analysis'
        y_axis_text = 'Pump Fillage (%)'
    else:
        title_text = 'Pump Efficiency Degradation Analysis'
        y_axis_text = 'Efficiency (%)'

    baseline_efficiency = 100
    match = re.search(r"HISTORICAL EFFICIENCY.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try: baseline_efficiency = float(match.group(1))
        except: pass
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    if baseline_efficiency > 0:
        annotations.append(create_threshold_band(labels, baseline_efficiency, f'Normal ({baseline_efficiency:.0f}%)', '#10b981'))
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': title_text, 'font': {'size': 16, 'weight': 'bold'}}
    
    options['scales']['y'] = {'type': 'linear', 'position': 'left', 'title': {'display': True, 'text': y_axis_text}, 'min': 0, 'max': 120, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    options['scales']['y1'] = {'type': 'linear', 'position': 'right', 'title': {'display': True, 'text': 'Production (bbl/day)'}, 'grid': {'drawOnChartArea': False}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Actual Production", "data": actual_oil, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y1"},
                {"label": "Predicted Production", "data": predicted_oil, "borderColor": "#9ca3af", "borderDash": [5, 5], "yAxisID": "y1"},
                {"label": y_axis_text, "data": efficiency, "borderColor": "#8b5cf6", "borderWidth": 3, "yAxisID": "y"}
            ]
        },
        "options": options
    }


def generate_baseline_shift_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=90)
    actual_data = daily_df['oil_volume'].round(1).fillna(0).tolist()
    
    baseline_val = None
    match = re.search(r"90-DAY HISTORICAL AVG.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try: baseline_val = float(match.group(1))
        except: pass
    if baseline_val is None or baseline_val == 0:
        baseline_val = daily_df['oil_volume'].quantile(0.75)
    
    baseline_line = [baseline_val] * len(labels)
    if len(daily_df) >= 7:
        rolling_avg = daily_df['oil_volume'].rolling(7, min_periods=3).mean().fillna(method='bfill').round(1).tolist()
    else:
        rolling_avg = actual_data
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    annotations.append(create_threshold_band(labels, baseline_val, f'90-Day Baseline ({baseline_val:.0f} bbl)', '#10b981'))
    
    if severity_metrics and severity_metrics.get('decline_pct', 0) > 25:
        current_level = baseline_val * (1 - (severity_metrics.get('decline_pct')/100))
        annotations.append(create_threshold_band(labels, current_level, 'New Baseline', '#ef4444'))
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Production Baseline Shift Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Decline: {severity_metrics.get('decline_pct', 0):.1f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Daily Production", "data": actual_data, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True},
                {"label": "7-Day Trend", "data": rolling_avg, "borderColor": "#0ea5e9", "borderWidth": 3, "pointRadius": 0},
                {"label": "Historical Baseline", "data": baseline_line, "borderColor": "#10b981", "borderWidth": 2, "borderDash": [8, 4], "pointRadius": 0}
            ]
        },
        "options": options
    }


def generate_pressure_instability_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    
    # Determine which pressure to show (Discharge for ESP, Tubing for others)
    if 'discharge_pressure' in daily_df.columns:
        pressure_avg = daily_df['discharge_pressure'].fillna(0).round(1).tolist()
        label_text = "Discharge Pressure"
    else:
        pressure_avg = daily_df.get('tubing_pressure', [0] * len(labels)).fillna(0).round(1).tolist()
        label_text = "Tubing Pressure"

    oil_production = daily_df['oil_volume'].fillna(0).round(1).tolist()
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': f'{label_text} Instability & Production Correlation', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Volatility: {severity_metrics.get('volatility_pct', 0):.1f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {'type': 'linear', 'position': 'left', 'title': {'display': True, 'text': 'Pressure (PSI)'}, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    options['scales']['y1'] = {'type': 'linear', 'position': 'right', 'title': {'display': True, 'text': 'Oil Production (bbl/day)'}, 'grid': {'drawOnChartArea': False}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": f"Avg {label_text}", "data": pressure_avg, "borderColor": "#ef4444", "borderWidth": 3, "yAxisID": "y"},
                {"label": "Oil Production", "data": oil_production, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y1"}
            ]
        },
        "options": options
    }


def generate_monthly_decline_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    start_date = df['timestamp'].max() - pd.Timedelta(days=180)
    monthly_raw = df[df['timestamp'] >= start_date].copy()
    monthly_raw['month_str'] = monthly_raw['timestamp'].dt.to_period('M').astype(str)
    
    monthly_df = monthly_raw.groupby('month_str').agg({
        'oil_volume': 'sum',
        'timestamp': 'count'
    }).reset_index()
    
    monthly_df['days_count'] = (monthly_df['timestamp'] / 24).round(0).clip(lower=1)
    if monthly_df['days_count'].mean() < 32: monthly_df['days_count'] = monthly_df['timestamp'] # Fallback
    
    monthly_df['rate'] = (monthly_df['oil_volume'] / monthly_df['days_count']).round(1)
    monthly_df['change_pct'] = monthly_df['rate'].pct_change() * 100
    
    labels = monthly_df['month_str'].tolist()
    rates = monthly_df['rate'].tolist()
    changes = monthly_df['change_pct'].fillna(0).round(1).tolist()
    
    bar_colors = ['#ef4444' if c < -15 else '#f59e0b' if c < 0 else '#10b981' for c in changes]
    
    options = get_common_options()
    options['plugins']['title'] = {'display': True, 'text': 'Monthly Production Decline Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Overall Decline: {severity_metrics.get('decline_pct', 0):.1f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    options['plugins']['datalabels'] = {'display': True, 'anchor': 'end', 'align': 'top', 'font': {'size': 10, 'weight': 'bold'}}
    
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Avg Daily Rate (bbl/d)",
                    "data": rates,
                    "backgroundColor": bar_colors,
                    "borderRadius": 6,
                    "borderWidth": 2,
                    "borderColor": "#ffffff"
                }
            ]
        },
        "options": options
    }


def generate_pattern_shift_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=90)
    
    sensor_col = 'motor_current'
    sensor_label = 'Motor Current (Amps)'
    violation_label = 'Current Anomalies'
    
    if 'tubing' in anomaly_data.lower() or 'pressure' in anomaly_data.lower():
        sensor_col = 'tubing_pressure'
        sensor_label = 'Tubing Pressure (PSI)'
        violation_label = 'Pressure Spikes'
    
    sensor_data = daily_df.get(sensor_col, [0] * len(labels)).fillna(0).round(1).tolist()
    
    # Simulated violation rate for visualization since daily aggregation hides high-freq spikes
    # In real implementation, this would aggregate boolean flags from high-freq data
    violation_rate = [0] * len(labels) 
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': f'Operational Pattern Shift - {violation_label}', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Violation Frequency Increase: {severity_metrics.get('frequency_increase_pct', 0):.0f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {'type': 'linear', 'position': 'left', 'title': {'display': True, 'text': sensor_label}, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    options['scales']['y1'] = {'type': 'linear', 'position': 'right', 'title': {'display': True, 'text': 'Violation Rate (%)'}, 'min': 0, 'max': 100, 'grid': {'drawOnChartArea': False}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": sensor_label, "data": sensor_data, "borderColor": "#64748b", "backgroundColor": "rgba(100, 116, 139, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y"},
                {"label": "Violation Rate (7-day %)", "data": violation_rate, "borderColor": "#ef4444", "backgroundColor": "rgba(239, 68, 68, 0.2)", "borderWidth": 3, "fill": True, "yAxisID": "y1"}
            ]
        },
        "options": options
    }


def generate_ghost_production_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=30)
    
    oil_data = daily_df['oil_volume'].fillna(0).round(1).tolist()
    water_data = daily_df.get('water_volume', [0] * len(labels)).fillna(0).round(1).tolist()
    gas_data = daily_df.get('gas_volume', [0] * len(labels)).fillna(0).round(1).tolist()
    
    motor_current = daily_df.get('motor_current', [0] * len(labels)).fillna(0).round(1).tolist()
    spm = daily_df.get('strokes_per_minute', [0] * len(labels)).fillna(0).round(1).tolist()
    
    pump_running = [1 if (mc > 5 or s > 0.1) else 0 for mc, s in zip(motor_current, spm)]
    pump_running_scaled = [pr * max(oil_data + water_data + gas_data + [10]) * 0.1 for pr in pump_running]
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Ghost Production Analysis - Equipment vs Output', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': 'Pump running but zero production - potential SCADA/meter failure', 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {'stacked': True, 'title': {'display': True, 'text': 'Production Volume'}, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Oil", "data": oil_data, "backgroundColor": "#10b981", "borderRadius": 4, "stack": "production"},
                {"label": "Water", "data": water_data, "backgroundColor": "#3b82f6", "borderRadius": 4, "stack": "production"},
                {"label": "Gas", "data": gas_data, "backgroundColor": "#f59e0b", "borderRadius": 4, "stack": "production"},
                {"label": "Pump Active (Indicator)", "data": pump_running_scaled, "type": "line", "borderColor": "#ef4444", "borderWidth": 3, "borderDash": [5, 5], "fill": False, "pointRadius": 0, "yAxisID": "y"}
            ]
        },
        "options": options
    }


def generate_default_enhanced_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    oil_data = daily_df['oil_volume'].fillna(0).round(1).tolist()
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Production Trend Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Oil Production", "data": oil_data, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True, "tension": 0.3}]
        },
        "options": options
    }


def get_common_options():
    return {
        "responsive": True,
        "maintainAspectRatio": False,
        "interaction": {
            "mode": "index",
            "intersect": False
        },
        "plugins": {
            "legend": {
                "position": "top",
                "labels": {
                    "usePointStyle": True,
                    "padding": 15,
                    "font": {"size": 11}
                }
            },
            "tooltip": {
                "usePointStyle": True,
                "backgroundColor": "rgba(15, 23, 42, 0.95)",
                "titleColor": "#f8fafc",
                "bodyColor": "#f1f5f9",
                "borderColor": "#334155",
                "borderWidth": 1,
                "padding": 12,
                "titleFont": {"size": 13, "weight": "bold"},
                "bodyFont": {"size": 12},
                "displayColors": True
            }
        },
        "scales": {
            "x": {
                "grid": {"display": False, "drawBorder": True},
                "ticks": {
                    "maxTicksLimit": 10,
                    "maxRotation": 45,
                    "minRotation": 0,
                    "font": {"size": 10}
                }
            },
            "y": {
                "grid": {"color": "rgba(200, 200, 200, 0.15)"},
                "beginAtZero": False,
                "ticks": {"font": {"size": 10}}
            }
        },
        "elements": {
            "point": {
                "radius": 0,
                "hitRadius": 20,
                "hoverRadius": 6,
                "hoverBorderWidth": 2
            },
            "line": {
                "borderJoinStyle": "round"
            }
        }
    }


def create_fallback_chart(df: pd.DataFrame, error_msg: str) -> Dict:
    if df.empty:
        return {
            "type": "line",
            "data": {"labels": [], "datasets": []},
            "options": {
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "No Data Available",
                        "font": {"size": 14, "weight": "bold"},
                        "color": "#ef4444"
                    },
                    "subtitle": {
                        "display": True,
                        "text": "Insufficient data to generate chart",
                        "font": {"size": 11},
                        "color": "#64748b"
                    }
                }
            }
        }
    
    daily_df, labels = prepare_daily_aggregation(df, days=30)
    data = daily_df['oil_volume'].fillna(0).round(1).tolist()
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Oil Production (Fallback View)",
                "data": data,
                "borderColor": "#64748b",
                "backgroundColor": "rgba(100, 116, 139, 0.1)",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.3
            }]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Production Data (Limited View)",
                    "font": {"size": 14}
                },
                "subtitle": {
                    "display": True,
                    "text": f"Chart generation issue: {error_msg[:50]}",
                    "font": {"size": 10},
                    "color": "#f59e0b"
                }
            }
        }
    }