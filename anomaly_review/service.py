"""
Anomaly Review Service
======================

Core service for detecting and processing anomalies in well data.

Anomaly Review is the middle layer between normal operations and At-Risk events:
something is not normal but not necessarily bad/broken—requires human review,
not automatic alarm. Purpose: "This is unusual / we didn't expect this" →
prompt investigation. See ANOMALY_REVIEW_DEFINITION.md for full definition.

Orchestrates:
- Data fetching and preprocessing
- ML model training and prediction
- Checker execution (pattern change, production inconsistency, unexpected events)
- Anomaly filtering and suppression
- Content generation and persistence
"""

import sys
import json
import logging
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Import from anomaly_review submodules
from anomaly_review.config import (
    DETECTION_WINDOW_DAYS,
    SUPPRESSION_RULES,
    ANOMALY_PRIORITY,
)
from anomaly_review.utils import safe_divide

from anomaly_review.data import (
    fetch_well_data,
    fetch_all_active_wells,
    calculate_rolling_baseline,
    get_recent_anomaly_history,
    fetch_sensor_metadata,
    get_snowflake_conn,
    get_db_url,
)

from anomaly_review.checkers import (
    check_model_divergence,
    check_energy_efficiency_drift,
    check_fluid_pound,
    check_gas_injection_inefficiency,
    check_pump_wear,
    check_efficiency_degradation,
    check_pressure_instability,
    check_monthly_production_decline,
    check_event_frequency,
    check_ml_anomaly,
    determine_severity,
    check_energy_production_divergence,
    check_sensor_state_mismatch,
    diagnose_mechanics,
)

from anomaly_review.models import (
    get_or_train_model,
    predict_with_lstm,
    detect_autoencoder_anomalies,
)

from anomaly_review.charts import generate_chart_config
from anomaly_review.llm import generate_llm_content

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnomalyReviewService')


def select_top_anomalies_per_well(anomalies: List[Dict]) -> List[Dict]:
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
        # 1. Sort: Most recent date first, then by Priority (lower number is better)
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

def detect_anomalies(well_id: str, df: pd.DataFrame = None, lookback_days: int = DETECTION_WINDOW_DAYS, save_to_db: bool = True) -> List[Dict]:
    logger.info(f"Starting anomaly detection for {well_id}...")
    
    # 1. Fetch Data (Fetch ~6 months to ensure valid baselines)
    # 1. Fetch Data (Fetch ~6 months to ensure valid baselines)
    if df is not None:
        full_df = df
    else:
        full_df = fetch_well_data(well_id, days=180)
    
    if full_df.empty or len(full_df) < 5:
        logger.warning(f"Insufficient data for {well_id} (Rows: {len(full_df)})")
        return []

    # --- ALARM GATEKEEPER ---
    # Filter out-of-range values using Sensor Metadata
    try:
        meta_df = fetch_sensor_metadata()
        if not meta_df.empty:
            # Iterate through defined sensors and apply limits
            # meta_df columns: field_name, min_value, max_value
            for _, row_meta in meta_df.iterrows():
                col = row_meta['field_name']
                min_val = row_meta['min_value']
                max_val = row_meta['max_value']
                
                if col in full_df.columns:
                    # Identify out-of-range indices
                    # We use mask to replace values where condition is True
                    mask = (full_df[col] < min_val) | (full_df[col] > max_val)
                    if mask.any():
                        # Ensure column is float before assigning NaN
                        if full_df[col].dtype.kind in 'iu':
                             full_df[col] = full_df[col].astype(float)
                        # Replace with NaN (effectively deleting the specific data point)
                        full_df.loc[mask, col] = np.nan
    except Exception as e:
        logger.error(f"Alarm Gatekeeper failed for {well_id}: {e}")

    # --- FLATLINE DETECTION ---
    # Detect sensors stuck at a single value (Variance = 0) over last 24 hours
    try:
        # Define sensors to check
        flatline_sensors = ['tubing_pressure', 'casing_pressure', 'motor_current', 'pump_fillage', 'strokes_per_minute']
        
        # Get data for last 24 hours
        if not full_df.empty and 'timestamp' in full_df.columns:
            last_ts = full_df['timestamp'].max()
            start_ts = last_ts - pd.Timedelta(hours=24)
            
            # Mask for recent data
            recent_mask = full_df['timestamp'] >= start_ts
            recent_data = full_df.loc[recent_mask]
            
            if len(recent_data) > 2: # Need at least a few points to compute std
                for sensor in flatline_sensors:
                    if sensor in recent_data.columns:
                        std_dev = recent_data[sensor].std()
                        
                        # Check strictly for 0.0 (or extremely close)
                        if pd.notna(std_dev) and std_dev == 0.0:
                            logger.warning(f"Flatline detected for {sensor} on {well_id} (StdDev=0). Masking recent values.")
                            # Mask values in the main dataframe for the recent period
                            full_df.loc[recent_mask, sensor] = np.nan
    except Exception as e:
        logger.error(f"Flatline detection failed for {well_id}: {e}")
    # --------------------------

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
            # 3a. Random Forest Baseline (Always run if available)
            if 'rf_model' in model_art.get('models', {}):
                 features = model_art['features']
                 from anomaly_review.models import safe_prepare_data
                 # Only ffill; leave NaNs for the Imputer to handle (Mean Imputation)
                 predict_df = safe_prepare_data(full_df, features).ffill()
                 X = model_art['scaler'].transform(model_art['imputer'].transform(predict_df))
                 full_df['predicted_oil'] = model_art['models']['rf_model'].predict(X)
            elif 'model' in model_art: # Legacy/Fallback support
                 features = model_art['features']
                 from anomaly_review.models import safe_prepare_data
                 predict_df = safe_prepare_data(full_df, features).ffill()
                 X = model_art['scaler'].transform(model_art['imputer'].transform(predict_df))
                 full_df['predicted_oil'] = model_art['model'].predict(X)

            # 3b. LSTM Forecasting (Override if available)
            if 'lstm_regressor' in model_art.get('models', {}):
                try:
                    logger.info("Using LSTM Regressor for forecasting...")
                    lstm_preds = predict_with_lstm(
                        model_art['models']['lstm_regressor'],
                        full_df,
                        model_art['features'],
                        model_art['scaler'],
                        model_art['imputer']
                    )
                    full_df['predicted_oil'] = lstm_preds.combine_first(full_df['predicted_oil'])
                    if 'oil_volume' in full_df.columns:
                        full_df['predicted_oil'] = full_df['predicted_oil'].fillna(full_df['oil_volume'])
                except Exception as e:
                    logger.error(f"LSTM Forecasting failed: {e}")

            # 3c. Autoencoder Anomaly Detection
            full_df['reconstruction_error'] = 0.0
            autoencoder_threshold = 999.0
            
            if 'autoencoder' in model_art.get('models', {}):
                try:
                    ae_model = model_art['models']['autoencoder']
                    ae_results = detect_autoencoder_anomalies(
                        ae_model, 
                        full_df, 
                        model_art['features'], 
                        model_art['scaler'], 
                        model_art['imputer'],
                        model_art.get('threshold', 0.5)
                    )
                    if not ae_results.empty:
                         full_df['reconstruction_error'] = ae_results['reconstruction_error']
                         autoencoder_threshold = model_art.get('threshold', 0.5)
                except Exception as e:
                    logger.error(f"Autoencoder detection failed: {e}")

            # Recalculate Efficiency with BEST prediction
            safe_pred = full_df['predicted_oil'].replace(0, 0.001)
            full_df['efficiency'] = safe_divide(full_df['oil_volume'], safe_pred) * 100
            full_df['efficiency'] = full_df['efficiency'].clip(0, 120)

            # Rolling avg recalculation
            if 'timestamp' in full_df.columns:
                temp_df = full_df.set_index('timestamp').sort_index()
                full_df['efficiency_avg_30d'] = temp_df['efficiency'].rolling('30D', min_periods=7).mean().values
            else:
                full_df['efficiency_avg_30d'] = full_df['efficiency'].rolling(30, min_periods=7).mean()
                
        except Exception as e:
            logger.error(f"ML Prediction failed for {well_id}: {e}")

    # 4. Run Checkers on the Detection Window
    # We fetch 180 days for context, but only scan the last `lookback_days` for new alerts.
    check_window_start = full_df['timestamp'].max() - pd.Timedelta(days=lookback_days)
    window_df = full_df[full_df['timestamp'] >= check_window_start].copy()
    
    # Checkers aligned with Anomaly Review: pattern change, production inconsistency, unexpected events
    all_checkers = [
        check_model_divergence,           # Production inconsistency: sensors normal, production drops vs expected
        # check_energy_efficiency_drift,  # Optional: subtle efficiency drift
        check_energy_production_divergence,
        check_sensor_state_mismatch,
        check_fluid_pound,
        check_gas_injection_inefficiency,
        check_pump_wear,
        check_efficiency_degradation,
        check_pressure_instability,
        check_monthly_production_decline,  # Runs on full DF
        check_event_frequency,             # Pattern change: e.g. activity 2x/year → monthly (Runs on full DF)
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
                # Checkers that require full dataframe context
                if check.__name__ in ['check_monthly_production_decline', 'check_event_frequency']:
                    # Only run these once per window (e.g. on the last day) to avoid duplicates
                    if row['timestamp'] == window_df['timestamp'].max():
                        res = check(ctx, full_df)
                else:
                    res = check(ctx)
            except Exception as e:
                logger.error(f"Checker '{check.__name__}' failed: {e}")
            
            # ML Anomaly Check (Special case - requires reconstruction error)
            if res is None and 'reconstruction_error' in ctx:
                try:
                    from anomaly_review.checkers import check_ml_anomaly
                    rec_error = ctx.get('reconstruction_error', 0.0)
                    res = check_ml_anomaly(
                        ctx,
                        rec_error,
                        autoencoder_threshold
                    )
                    if res:
                        logger.info(f"[ML_CHECK] Row {idx}: ANOMALY DETECTED recon_error={rec_error:.4f}, threshold={autoencoder_threshold:.4f}")
                except Exception as e:
                    logger.error(f"ML Anomaly checker failed: {e}")

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
                    "severity_metrics": res.get('severity_metrics', {}),
                    "chart_payload": res.get('chart_payload', {})
                })

    # 1. FATAL OVERRIDE: Data Mismatch (prev Ghost Production) implies sensor failure or critical break.
    # If this exists, it invalidates physics checks like Fluid Pound.
    ghost = next((a for a in raw_findings if a['anomaly_code'] == 'DATA_MISMATCH'), None)
    if ghost:
        # If ghost exists, return ONLY ghost (and maybe other criticals), filter out operational noise
        raw_findings = [a for a in raw_findings if a['anomaly_code'] in ['DATA_MISMATCH' , 'ENERGY_DIV' , 'PATTERN_SHIFT']]

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
            impact_metrics=finding['impact_payload'],
            anomaly_code=finding['anomaly_code']
        )
        
        # Check validation result (New Feature)
        if not llm_content.get('is_valid_anomaly', True):
             logger.info(f"LLM Validation Rejected: {finding['anomaly_code']} on {finding['event_date']} for {well_id}")
             continue
        
        # User-friendly title: avoid showing jargon like "Model Divergence" to end users
        title = llm_content.get('title', 'Anomaly Detected')
        if finding['anomaly_code'] == 'MODEL_DIVERGENCE' and title and ('divergence' in title.lower() or 'model performance' in title.lower()):
            title = "Production Below Expected"
            llm_content = {**llm_content, 'title': title}
        
        # Generate Chart Configuration
        chart_config = generate_chart_config(
            finding['raw_data'], 
            finding['chart_metric'], 
            full_df,
            anomaly_code=finding['anomaly_code'],
            severity_metrics=finding.get('severity_metrics', {}),
            chart_payload=finding.get('chart_payload', {})
        )
        
        enriched_results.append({
            "well_id": well_id,
            "event_date": finding['event_date'],
            "detected_at": finding['detected_at'],
            "anomaly_code": finding['anomaly_code'],
            "category": finding['category'],
            "severity": finding['severity'],
            "title": title,
            "status": "ACTIVE",
            "ui_text": llm_content,
            "impact_metrics": finding['impact_payload'],
            "impact_value": float(finding['impact_value']), # Ensure native float for DB
            "chart_data": chart_config
        })

    if enriched_results:
        logger.info(f"Found {len(enriched_results)} anomalies for {well_id} after suppression.")
        if save_to_db:
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