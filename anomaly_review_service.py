import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import shared utilities and checkers
from anomaly_review_utils import (
    fetch_well_data, 
    calculate_rolling_baseline, 
    get_or_train_model, 
    get_recent_anomaly_history, 
    generate_llm_content, 
    generate_chart_config, 
    determine_severity, 
    get_snowflake_conn, 
    get_db_url,
    # Specific Checkers
    check_ghost_production, 
    check_mechanical_failure, 
    check_fluid_pound, 
    check_gas_injection_inefficiency,
    check_pump_wear, 
    check_efficiency_degradation, 
    check_sustained_production_change,
    check_pressure_instability, 
    check_operational_pattern_shift, 
    check_monthly_production_decline,
    DETECTION_WINDOW_DAYS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnomalyReviewService')

# =============================================================================
# HIERARCHY OF TRUTH (Anomaly Suppression Logic)
# =============================================================================
# This system prevents alert fatigue by suppressing "Symptom" anomalies 
# when a specific "Root Cause" anomaly is detected on the same day.

SUPPRESSION_RULES = {
    # --- NEW TOP-LEVEL RULE ---
    'MECHANICAL_FAILURE': ['FLUID_POUND', 'PUMP_WEAR', 'EFFICIENCY_DEGRADATION', 'NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE', 'PRESSURE_INSTABILITY', 'PATTERN_SHIFT'],
    
    # --- EXISTING RULES ---
    'FLUID_POUND': ['EFFICIENCY_DEGRADATION', 'NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE', 'PRESSURE_INSTABILITY', 'PATTERN_SHIFT'],
    'PUMP_WEAR': ['EFFICIENCY_DEGRADATION', 'NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE', 'PATTERN_SHIFT'],
    'GAS_HOG': ['EFFICIENCY_DEGRADATION', 'NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE', 'PRESSURE_INSTABILITY'],
    
    # --- UPDATED RULES ---
    'PRESSURE_INSTABILITY': ['NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE', 'EFFICIENCY_DEGRADATION'],
    'PATTERN_SHIFT': ['NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE', 'EFFICIENCY_DEGRADATION'],
    'EFFICIENCY_DEGRADATION': ['NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE'],
    'NEW_PRODUCTION_BASELINE': ['MONTHLY_DECLINE']
}

# Priority Ranking (1 = Highest Priority / Most Specific)
ANOMALY_PRIORITY = {
    'GHOST_PROD': 1,
    'MECHANICAL_FAILURE': 1,     # <--- TOP PRIORITY
    'FLUID_POUND': 2,
    'PUMP_WEAR': 2,
    'GAS_HOG': 2, 
    'EFFICIENCY_DEGRADATION': 3,
    'PRESSURE_INSTABILITY': 3,
    'PATTERN_SHIFT': 3,
    'NEW_PRODUCTION_BASELINE': 4,
    'MONTHLY_DECLINE': 5
}

def select_top_anomalies_per_well(anomalies: list[dict]) -> list[dict]:
    """
    Filters anomalies based on hierarchy to prevent redundant alerts.
    Example: If 'Fluid Pound' and 'Efficiency Drop' are detected, only return 'Fluid Pound'.
    """
    if not anomalies: return []
    
    # Group anomalies by well_id
    grouped = defaultdict(list)
    for a in anomalies:
        grouped[a['well_id']].append(a)
        
    final_list = []
    
    for well_id, well_anomalies in grouped.items():
        # 1. FATAL OVERRIDE: Ghost Production usually implies sensor failure or critical break.
        # If this exists, it invalidates physics checks like Fluid Pound.
        ghost = next((a for a in well_anomalies if a['anomaly_code'] == 'GHOST_PROD'), None)
        if ghost:
            final_list.append(ghost)
            continue

        # 2. Sort: Most recent date first, then by Priority (lower number is better)
        well_anomalies.sort(key=lambda x: (x['event_date'], -ANOMALY_PRIORITY.get(x['anomaly_code'], 99)), reverse=True)

        # 3. Apply Suppression Rules
        present_codes = {a['anomaly_code'] for a in well_anomalies}
        suppressed_codes = set()
        
        for code in present_codes:
            if code in SUPPRESSION_RULES:
                suppressed_codes.update(SUPPRESSION_RULES[code])
        
        # 4. Filter and Deduplicate
        kept_anomalies = {}
        for a in well_anomalies:
            code = a['anomaly_code']
            # specific logic: if we have multiple days of the same anomaly, keep only the most recent
            if code not in suppressed_codes:
                if code not in kept_anomalies:
                    kept_anomalies[code] = a
        
        final_list.extend(kept_anomalies.values())
                
    return final_list

def detect_anomalies(well_id: str, df: pd.DataFrame = None, lookback_days: int = DETECTION_WINDOW_DAYS) -> list[dict]:
    logger.info(f"Starting anomaly detection for {well_id}...")
    
    # 1. Fetch Data (Fetch ~6 months to ensure valid baselines)
    full_df = fetch_well_data(well_id, days=180)
    
    if full_df.empty or len(full_df) < 30:
        logger.warning(f"Insufficient data for {well_id} (Rows: {len(full_df)})")
        return []

    # Ensure lift_type exists (Forward fill to handle intermittent metadata)
    if 'lift_type' not in full_df.columns:
        full_df['lift_type'] = 'Unknown'
    full_df['lift_type'] = full_df['lift_type'].ffill().bfill().fillna('Unknown')
    current_lift_type = full_df['lift_type'].iloc[-1]

    # 2. Pre-Processing & Rolling Baselines
    full_df = calculate_rolling_baseline(full_df)
    
    # 3. Model Training (Predictive Efficiency)
    full_df['predicted_oil'] = np.nan
    full_df['efficiency'] = 100.0
    
    model_art = get_or_train_model(well_id, full_df)
    
    if model_art:
        try:
            # Prepare features for prediction
            features = model_art['features']
            predict_df = full_df[features].fillna(method='ffill').fillna(0)
            
            # Predict
            X = model_art['scaler'].transform(model_art['imputer'].transform(predict_df))
            full_df['predicted_oil'] = model_art['model'].predict(X)
            
            # Calculate Efficiency (Safe Division)
            safe_pred = full_df['predicted_oil'].replace(0, 0.001)
            full_df['efficiency'] = (full_df['oil_volume'] / safe_pred) * 100

            # Clip outlier values (e.g., 5000% efficiency)
            full_df['efficiency'] = full_df['efficiency'].clip(0, 120)

            # Calculate rolling average (requires DatetimeIndex or on='timestamp')
            # Since calculate_rolling_baseline resets index, we must handle this.
            if 'timestamp' in full_df.columns:
                # rolling() on a column with 'on' param is only supported in newer pandas, 
                # safer to set index temporarily.
                temp_df = full_df.set_index('timestamp').sort_index()
                full_df['efficiency_avg_30d'] = temp_df['efficiency'].rolling('30D', min_periods=7).mean().values
            else:
                # Fallback if no timestamp
                full_df['efficiency_avg_30d'] = full_df['efficiency'].rolling(30, min_periods=7).mean()
        except Exception as e:
            logger.error(f"ML Prediction failed for {well_id}: {e}")

    # 4. Run Checkers on the Detection Window
    # We fetch 180 days for context, but only scan the last `lookback_days` for new alerts.
    check_window_start = full_df['timestamp'].max() - pd.Timedelta(days=lookback_days)
    window_df = full_df[full_df['timestamp'] >= check_window_start].copy()
    
    all_checkers = [
        check_ghost_production,
        check_mechanical_failure, 
        check_fluid_pound,
        check_gas_injection_inefficiency,
        check_pump_wear,
        check_efficiency_degradation,
        check_sustained_production_change,
        check_pressure_instability,
        check_operational_pattern_shift, # Runs on full DF inside
        check_monthly_production_decline # Runs on full DF inside
    ]
    
    raw_findings = []
    
    for idx, row in window_df.iterrows():
        # Build context object for standard checkers
        ctx = row.to_dict()
        ctx['actual_oil'] = ctx.get('oil_volume', 0)
        ctx['actual_water'] = ctx.get('water_volume', 0)
        ctx['pred_oil'] = ctx.get('predicted_oil', ctx.get('actual_oil', 0))
        ctx['lift_type'] = current_lift_type 
        
        for check in all_checkers:
            res = None
            try:
                # Optimized execution: Heavy trend checks only run on the most recent data point
                if check.__name__ in ['check_operational_pattern_shift', 'check_monthly_production_decline']:
                    if row['timestamp'] == window_df['timestamp'].max():
                        res = check(ctx, full_df)
                else:
                    res = check(ctx)
            except Exception as e:
                logger.error(f"Checker '{check.__name__}' failed: {e}")

            if res:
                # Calculate severity based on specific metric thresholds
                severity = determine_severity(res['code'], res.get('severity_metrics', {}))
                
                raw_findings.append({
                    "well_id": well_id,
                    "event_date": str(row['timestamp'].date()),
                    "timestamp": row['timestamp'],
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "anomaly_code": res['code'],
                    "category": res['category'],
                    "severity": severity,
                    "raw_data": res['raw_data'],
                    "chart_metric": res['chart_metric'],
                    "impact_value": res['impact_value'],
                    "impact_payload": res['impact_payload'],
                    "lift_type": str(current_lift_type),
                    "diagnosis": res.get('diagnosis'),
                    "severity_metrics": res.get('severity_metrics', {})
                })

    # 5. Filter & Deduplicate (The Truth Hierarchy)
    final_anomalies = select_top_anomalies_per_well(raw_findings)
    
    # 6. Generate Content & Save
    enriched_results = []
    for finding in final_anomalies:
        # Fetch History
        history_str = get_recent_anomaly_history(well_id, finding['anomaly_code'], exclude_date=finding['event_date'])
        
        # Generate LLM Text
        llm_content = generate_llm_content(
            finding['raw_data'], 
            finding['lift_type'], 
            finding['chart_metric'], 
            history_str, 
            finding['detected_at'],
            diagnosis=finding.get('diagnosis'),
            impact_metrics=finding['impact_payload']
        )
        
        # Generate Chart Configuration
        chart_config = generate_chart_config(
            finding['raw_data'], 
            finding['chart_metric'], 
            full_df,
            anomaly_code=finding['anomaly_code'],
            severity_metrics=finding.get('severity_metrics', {})
        )
        
        enriched_results.append({
            "well_id": well_id,
            "event_date": finding['event_date'],
            "detected_at": finding['detected_at'],
            "anomaly_code": finding['anomaly_code'],
            "category": finding['category'],
            "severity": finding['severity'],
            "title": llm_content.get('title', 'Anomaly Detected'),
            "status": "ACTIVE",
            "ui_text": llm_content,
            "impact_metrics": finding['impact_payload'],
            "impact_value": float(finding['impact_value']), # Ensure native float for DB
            "chart_data": chart_config
        })

    if enriched_results:
        logger.info(f"Found {len(enriched_results)} anomalies for {well_id} after suppression.")
        save_reviews(enriched_results)
    else:
        logger.info(f"No anomalies found for {well_id}.")

    return enriched_results

def save_reviews(reviews):
    if not reviews: 
        return

    # Helper to clean numpy types for JSON serialization
    def clean_json(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [clean_json(v) for v in obj]
        elif pd.isna(obj): return None
        return obj

    cleaned_reviews = [clean_json(r) for r in reviews]

    # ------------------------------------------------------------------
    # --- POSTGRES SAVE (ACTIVE) ---
    # ------------------------------------------------------------------
    url = get_db_url()
    if url:
        try:
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    q = '''
                        INSERT INTO anomaly_review
                        (well_id, event_date, detected_at, anomaly_code, category, severity, 
                         title, ui_text, impact_value, impact_metrics, chart_data, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (well_id, anomaly_code, event_date) DO UPDATE 
                        SET ui_text = EXCLUDED.ui_text, chart_data = EXCLUDED.chart_data
                    '''
                    for r in cleaned_reviews:
                        cur.execute(q, (
                            r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                            r['category'], r['severity'], r['title'],
                            json.dumps(r['ui_text']),
                            r['impact_value'],
                            json.dumps(r['impact_metrics']),
                            json.dumps(r['chart_data']),
                            r['status']
                        ))
                conn.commit()
        except Exception as e:
            logger.error(f"PostgreSQL Save failed: {e}")

    # ------------------------------------------------------------------
    # --- SNOWFLAKE SAVE (ACTIVE) ---
    # ------------------------------------------------------------------
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            insert_q = """
                INSERT INTO AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW
                (id, well_id, event_date, detected_at, anomaly_code, category, severity, 
                 title, ui_text, impact_value, impact_metrics, chart_data, status)
                SELECT UUID_STRING(), %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, 
                       PARSE_JSON(%s), PARSE_JSON(%s), %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW 
                    WHERE well_id = %s AND anomaly_code = %s AND event_date = %s
                )
            """
            for r in cleaned_reviews:
                cur.execute(insert_q, (
                    r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                    r['category'], r['severity'], r['title'],
                    json.dumps(r['ui_text']),
                    r['impact_value'],
                    json.dumps(r['impact_metrics']),
                    json.dumps(r['chart_data']),
                    r['status'],
                    # WHERE params
                    r['well_id'], r['anomaly_code'], r['event_date']
                ))
            conn.commit()
    except Exception as e:
        logger.error(f"Snowflake Save failed: {e}")

if __name__ == "__main__":
    wid = sys.argv[1] if len(sys.argv) > 1 else "Well_001_RodPump"
    # Execute detection and print JSON result to stdout
    results = detect_anomalies(wid)
    print(json.dumps(results, indent=2, default=str))