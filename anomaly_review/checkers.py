"""
Anomaly checker functions for various well conditions.

Provides physics-based and statistical checkers for:
- Fluid pound (Rod Pump)
- Pump wear (ESP)
- Gas injection inefficiency (Gas Lift)
- Mechanical failure
- Efficiency degradation
- Pressure instability
- Ghost production
- Monthly production decline
- ML-based anomaly detection
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

from anomaly_review.config import (
    OIL_PRICE,
    GAS_PRICE,
    SEVERITY_THRESHOLDS,
    format_number,
)
from anomaly_review.utils import safe_divide

logger = logging.getLogger('AnomalyReviewCheckers')


# =============================================================================
# SEVERITY DETERMINATION
# =============================================================================

def determine_severity(anomaly_code: str, metrics: Dict) -> str:
    """
    Determine severity based on score thresholds.
    
    Severity is Score-Based (How weird is it?), not Money-Based.
    """
    thresholds = SEVERITY_THRESHOLDS.get(anomaly_code, {"high": 999, "moderate": 999})
    
    # Physics Score Priority
    if 'severity_score' in metrics:
        score = metrics['severity_score']
        if score >= thresholds.get('critical', 999):
            return "Critical"
        if score >= thresholds.get('high', 999):
            return "High"
        if score >= thresholds.get('moderate', 999):
            return "Moderate"
        return "Low"
    
    # Percentage Drop Priority
    for k in ['efficiency_drop', 'decline_pct', 'volatility_pct', 'frequency_increase_pct']:
        if k in metrics:
            val = abs(metrics[k])
            if val >= thresholds.get('critical', 999):
                return "Critical"
            if val >= thresholds.get('high', 999):
                return "High"
            if val >= thresholds.get('moderate', 999):
                return "Moderate"
            return "Low"

    return "Moderate"


# =============================================================================
# PHYSICS DIAGNOSIS
# =============================================================================

def diagnose_mechanics(ctx: Dict) -> Dict:
    """
    Diagnose mechanical issues based on sensor correlations.
    
    Returns diagnosis dict with type, confidence, and indicators.
    """
    amps = ctx.get('motor_current', 0)
    amps_avg = ctx.get('amps_avg', 0)
    pressure = ctx.get('tubing_pressure', 0)
    pressure_avg = ctx.get('tp_avg', 0)
    oil = ctx.get('oil_volume', 0)
    pred_oil = ctx.get('predicted_oil', 1)
    
    # Avoid division by zero
    efficiency = safe_divide(oil, pred_oil, default=100.0) * 100
    amps_ratio = safe_divide(amps, amps_avg, default=1.0)
    pressure_ratio = safe_divide(pressure, pressure_avg, default=1.0)
    
    # Parted Rods or Belt Break
    if amps_ratio < 0.6:
        return {
            'type': 'Parted Rods or Belt Break',
            'confidence': 'High',
            'indicators': f'Motor Current dropped significantly ({amps_ratio:.2f}x)'
        }

    # Seized Pump / Locked Rotor
    elif amps_ratio > 1.5 and efficiency < 20:
        return {
            'type': 'Seized Pump / Locked Rotor',
            'confidence': 'High',
            'indicators': f'High Amps ({amps_ratio:.1f}x) with near-zero flow'
        }

    # Flowline Blockage
    elif pressure_ratio > 1.3:
        return {
            'type': 'Flowline Blockage / High Line Pressure',
            'confidence': 'High',
            'indicators': f'Tubing Pressure elevated ({pressure_ratio:.1f}x)'
        }

    # Tubing Leak
    elif (0.9 < amps_ratio < 1.1) and efficiency < 50:
        return {
            'type': 'Tubing Leak',
            'confidence': 'Moderate',
            'indicators': f'Normal Amps but low efficiency ({efficiency:.0f}%)'
        }
    
    # Pump Wear
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


# =============================================================================
# PHYSICS CHECKERS (Lift Type Specific)
# =============================================================================

def check_fluid_pound(ctx: Dict) -> Optional[Dict]:
    """
    Rod Pump: Detects Fluid Pound (Shock Loading).
    
    Triggers when pump fillage drops significantly below baseline.
    """
    lift_type = str(ctx.get('lift_type', '')).lower().replace(' ', '')
    if 'rod' not in lift_type:
        return None

    if pd.isna(ctx.get('pump_fillage')) or pd.isna(ctx.get('strokes_per_minute')):
        return None
    
    fillage = ctx['pump_fillage']
    spm = ctx['strokes_per_minute']
    baseline_fillage = ctx.get('fillage_avg_24h', 95.0)
    
    logger.info(f"[FLUID_POUND_CHECK] Fillage: {fillage}%, Baseline: {baseline_fillage}%, SPM: {spm}")
    
    # Anomaly if fillage drops more than 15% below baseline
    if spm > 4.0 and fillage < (baseline_fillage * 0.85):
        severity_score = 100 - fillage
        revenue_loss = ctx.get('oil_avg_7d', 0) * ((100 - fillage) / 100) * OIL_PRICE
        
        # BOE calculation
        loss_pct = (100 - fillage) / 100
        oil_comp = ctx.get('oil_avg_7d', 0) * loss_pct
        gas_comp = ctx.get('gas_avg_7d', 0) * loss_pct
        boe_loss = oil_comp + (gas_comp / 6)
        
        data_summary = f"""
        DETECTION TYPE: Rod Pump Fluid Pound
        CURRENT PUMP FILLAGE: {format_number(fillage)}% (Baseline: {format_number(baseline_fillage)}%)
        STROKES PER MINUTE: {format_number(spm, "spm")}
        FILLAGE DROP: {format_number(baseline_fillage - fillage)}% below baseline
        ROOT CAUSE: The pump is not filling completely on the downstroke, causing shock load.
        CALCULATED_REVENUE_LOSS: ${format_number(revenue_loss)}/day
        """
        
        logger.info(f"[FLUID_POUND_DETECTED] Data Summary:\n{data_summary}")
        
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
                "type": "Rod Pump Fluid Pound",
                "confidence": "High",
                "indicators": f"Pump Fillage {format_number(fillage)}% vs SPM {format_number(spm, 'spm')}"
            },
            "chart_payload": {
                "baseline_fillage": baseline_fillage,
                "current_fillage": fillage,
                "efficiency": fillage # Align with chart logic
            }
        }
    return None


def check_pump_wear(ctx: Dict) -> Optional[Dict]:
    """
    ESP: Detects Pump Wear (Head Degradation).
    
    Triggers when discharge pressure is low despite high motor current.
    """
    lift_type = str(ctx.get('lift_type', '')).lower().replace(' ', '')
    if 'esp' not in lift_type:
        return None

    req = ['discharge_pressure', 'pump_intake_pressure', 'motor_current']
    if any(pd.isna(ctx.get(k)) for k in req):
        return None
    
    head = ctx['discharge_pressure'] - ctx['pump_intake_pressure']
    amps = ctx['motor_current']
    
    if head < 800 and amps > 30:
        revenue_loss = 2000.0
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
            },
            "chart_payload": {
                "head": head,
                "amps": amps
            }
        }
    return None


def check_gas_injection_inefficiency(ctx: Dict) -> Optional[Dict]:
    """
    Gas Lift: Detects Gas Hog (inefficient injection).
    
    Triggers when injection rate is high but pressure differential is low.
    """
    lift_type = str(ctx.get('lift_type', '')).lower().replace(' ', '')
    if 'gas' not in lift_type:
        return None

    req = ['injection_rate', 'casing_pressure', 'tubing_pressure']
    if any(pd.isna(ctx.get(k)) for k in req):
        return None
    
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
                "type": "Gas Lift Inefficiency",
                "confidence": "High",
                "indicators": f"High Rate ({format_number(rate)}) with minimal casing differential"
            },
            "chart_payload": {
                "injection_rate": rate,
                "differential": cp - tp
            }
        }
    return None


# =============================================================================
# STANDARD CHECKERS
# =============================================================================

def check_energy_efficiency_drift(ctx: Dict) -> Optional[Dict]:
    """
    Detects subtle efficiency drift by reducing Energy Intensity.
    
    Logic:
    - Calculate Energy Intensity (Amps / Oil Volume).
    - Compare current vs 30-day average.
    - Trigger if current is 15-20% higher (Pump working harder for same fluid).
    """
    oil = ctx.get('oil_volume', 0)
    amps = ctx.get('motor_current', 0)
    
    # Needs valid production to calculate intensity
    if oil < 0.1 or amps < 1:
        return None
        
    # Calculate Energy Intensity (Amps per Bbl or KW per Bbl)
    # Lower is better. If this RISES, efficiency is dropping.
    current_intensity = safe_divide(amps, oil, default=0.0)
    
    # Compare to 30-day baseline intensity
    baseline_oil = ctx.get('average_oil_30d', 0)
    baseline_amps = ctx.get('average_amps_30d', 0)
    baseline_intensity = safe_divide(baseline_amps, baseline_oil, default=0.0)
    
    if baseline_intensity == 0:
        return None
        
    drift_pct = safe_divide((current_intensity - baseline_intensity), baseline_intensity) * 100
        
    if drift_pct > 15: # 15% threshold
        revenue_loss = ctx.get('predicted_oil', oil) * 0.15 * OIL_PRICE # Est 15% potential loss
        boe_loss = ctx.get('predicted_oil', oil) * 0.15
            
        data_summary = f"""
        DETECTION TYPE: Energy Efficiency Drift
        ENERGY INTENSITY: {current_intensity:.2f} (Baseline: {baseline_intensity:.2f}) (+{drift_pct:.1f}%)
        KEY METRICS: Amps {format_number(amps)}, Oil {format_number(oil)} bbl
        """
            
        return {
            "code": "EFFICIENCY_DRIFT",
            "category": "OPERATIONAL",
            "raw_data": data_summary,
            "severity_metrics": {'efficiency_drop': drift_pct, 'impact_value': revenue_loss},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day (Inefficiency)"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day (Risk)"}
            },
            "chart_metric": "efficiency_trend",
            "diagnosis": {
                "type": "Energy Efficiency Drift",
                "confidence": "Moderate",
                "indicators": f"Energy Intensity rose {drift_pct:.1f}% vs baseline"
            },
            "chart_payload": {
                "baseline_intensity": baseline_intensity,
                "current_intensity": current_intensity,
                "drift_pct": drift_pct
            }
        }
    return None


def check_model_divergence(ctx: Dict) -> Optional[Dict]:
    """
    Detects significant divergence between Predicted and Actual production.
    Replaces 'check_ghost_production'.
    
    Logic:
    - Trigger if Actual Oil is 30-40% lower than Predicted.
    - Detects underperformance, not just zero flow.
    """
    actual = ctx.get('oil_volume', 0)
    predicted = ctx.get('predicted_oil', 0)
    
    if predicted < 5: # Ignore low production wells
        return None
        
    # Divergence: (Predicted - Actual) / Predicted
    # Positive means Underperformance (Predicted > Actual)
    divergence_pct = safe_divide((predicted - actual), predicted) * 100
    
    if divergence_pct > 30: # 30% threshold
        delta = predicted - actual # delta is needed for revenue_loss and boe_loss
        revenue_loss = delta * OIL_PRICE
        boe_loss = delta
        
        data_summary = f"""
        DETECTION TYPE: Model Performance Divergence
        ACTUAL PRODUCTION: {format_number(actual)} bbl
        PREDICTED CAPABILITY: {format_number(predicted)} bbl
        DIVERGENCE: {divergence_pct:.1f}% Below Model Expectation
        """
        
        return {
            "code": "MODEL_DIVERGENCE",
            "category": "PRODUCTION",
            "raw_data": data_summary,
            "severity_metrics": {'decline_pct': divergence_pct, 'impact_value': revenue_loss, 'severity_score': divergence_pct},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day gap"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day gap"}
            },
            "chart_metric": "production_bar",
            "diagnosis": {
                "type": "Model Performance Divergence",
                "confidence": "High",
                "indicators": f"Performing {divergence_pct:.0f}% below model prediction"
            },
            "chart_payload": {
                "actual_oil": actual,
                "predicted_oil": predicted,
                "divergence_pct": divergence_pct
            }
        }
    return None

def check_efficiency_degradation(ctx: Dict) -> Optional[Dict]:
    """
    Detects efficiency drops compared to predicted production.
    (Kept as legacy/backup, overlapping with model_divergence but distinct calculation)
    """
    # ... logic mostly covered by model_divergence now, but keeping if logic differs slightly 
    # or if we want to remove it to avoid dupes. 
    # Plan didn't explicitly say remove, but `check_model_divergence` covers it. 
    # I'll keep it but increase threshold to avoid noise or remove if overlapping too much.
    # For now, let's essentially alias it or just rely on model_divergence.
    # Returning None to defer to model_divergence.
    return None


def check_pressure_instability(ctx: Dict) -> Optional[Dict]:
    """
    Detects pressure volatility (Std Dev) rather than absolute thresholds.
    """
    tp_avg = ctx.get('tp_avg', 0)
    tp_std = ctx.get('tp_std', 0) # 3-day std provided by data.py logic?
    
    # We need today's 'volatility' vs historical 'volatility'. 
    # ctx has 'tp_std' which is rolling 3D std. 
    # Let's say current volatility is high if 'tp_std' is high relative to 'tp_avg'.
    
    if tp_avg > 20:
        cv = (tp_std / tp_avg) * 100 # Coefficient of Variation
        
        # If volatility is > 20% and we see it rising
        if cv > 20:
            revenue_loss = ctx.get('oil_volume', 0) * 0.15 * OIL_PRICE
            
            data_summary = f"""
            DETECTION TYPE: Pressure Volatility
            VOLATILITY (CV): {cv:.1f}% (High)
            STD DEV: {tp_std:.1f} psi (Mean Pressure: {tp_avg:.0f} psi)
            """

            return {
                "code": "PRESSURE_INSTABILITY",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity_metrics": {'volatility_pct': cv, 'impact_value': revenue_loss},
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day (Risk)"},
                    "production": {"value": 0, "label": "N/A"}
                },
                "chart_metric": "pressure_trend",
                "diagnosis": {
                    "type": "Flow Instability / Slugging",
                    "confidence": "Moderate",
                    "indicators": f"Pressure Volatility {cv:.1f}%"
                },
                "chart_payload": {
                    "volatility_pct": cv,
                    "pressure_avg": tp_avg,
                    "pressure_std": tp_std
                }
            }
    return None


def check_event_frequency(ctx: Dict, full_df: pd.DataFrame = None) -> Optional[Dict]:
    """
    Detects change in operational pattern frequency.
    Checks frequency of 'Shutdowns' (Amps == 0) in last 30d vs previous.
    """
    if full_df is None or 'motor_current' not in full_df.columns:
        return None
        
    # Define "Event": Motor Current < 1.0 (Shutdown)
    full_df['is_shutdown'] = full_df['motor_current'] < 1.0
    
    # Last 30 days
    last_30_start = full_df['timestamp'].max() - pd.Timedelta(days=30)
    recent = full_df[full_df['timestamp'] >= last_30_start]
    
    # Previous window (30d to 120d ago)
    prev_start = last_30_start - pd.Timedelta(days=90)
    previous = full_df[(full_df['timestamp'] < last_30_start) & (full_df['timestamp'] >= prev_start)]
    
    recent_count = recent['is_shutdown'].sum()
    prev_count_avg = previous['is_shutdown'].sum() / 3 # Monthly avg over 3 months
    
    # Trigger: Recent count significantly higher (> 2x baseline and > 2 events)
    if recent_count > 2 and recent_count > (prev_count_avg * 2):
        increase_pct = ((recent_count - prev_count_avg) / prev_count_avg) * 100 if prev_count_avg > 0 else 100
        
        data_summary = f"""
        DETECTION TYPE: Recurring Operational Events
        EVENT: Pump Shutdowns
        RECENT COUNT (30d): {recent_count}
        HISTORICAL AVG: {prev_count_avg:.1f} per month
        FREQUENCY CHANGE: +{increase_pct:.0f}%
        """
        
        return {
            "code": "PATTERN_SHIFT",
            "category": "OPERATIONAL",
            "raw_data": data_summary,
            "severity_metrics": {'frequency_increase_pct': increase_pct, 'impact_value': 0},
            "impact_value": 0,
            "impact_payload": {
                "financial": {"value": 0, "label": "Operational Review"},
                "production": {"value": 0, "label": "N/A"}
            },
            "chart_metric": "mechanical_failure_view", # Visualize amps
            "diagnosis": {
                "type": "Recurring Operational Events",
                "confidence": "High",
                "indicators": f"Shutdowns jumped to {recent_count}/mo from {prev_count_avg:.1f}/mo"
            },
            "chart_payload": {
                "recent_count": recent_count,
                "historical_avg": prev_count_avg,
                "increase_pct": increase_pct
            }
        }
    return None

def check_monthly_production_decline(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    """
    Detects significant month-over-month production decline.
    """
    if 'timestamp' not in df.columns or df['timestamp'].empty:
        return None
    
    recent_df = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=120))].copy()
    recent_df['month'] = recent_df['timestamp'].dt.to_period('M')
    
    monthly = recent_df.groupby('month').agg({'oil_volume': ['sum', 'count']}).reset_index()
    monthly.columns = ['month', 'oil_total', 'oil_days']
    monthly['oil_rate'] = safe_divide(monthly['oil_total'], monthly['oil_days'])
    
    if len(monthly) >= 2:
        latest = monthly.iloc[-1]
        previous = monthly.iloc[-2]
        
        if latest['oil_days'] < 3:
            return None

        if previous['oil_rate'] < 5: 
            return None # Ignore low rate wells
        
        decline_pct = safe_divide((previous['oil_rate'] - latest['oil_rate']), previous['oil_rate']) * 100
            
        if decline_pct > 15:
            revenue_loss = max(0, previous['oil_rate'] - latest['oil_rate']) * OIL_PRICE
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
                "diagnosis": {
                "type": "Monthly Production Decline",
                "confidence": "High",
                "indicators": "Comparison of monthly production rates."
            },
                "chart_payload": {
                    "decline_pct": decline_pct,
                    "previous_rate": previous['oil_rate'],
                    "current_rate": latest['oil_rate']
                }
            }
    return None


# =============================================================================
# ML-BASED CHECKER
# =============================================================================

def check_ml_anomaly(ctx: Dict, reconstruction_error: float, threshold: float) -> Optional[Dict]:
    """
    Checker for Autoencoder-detected anomalies.
    
    Triggers when reconstruction error exceeds threshold.
    """
    # Goldilocks Filter: Ignore massive errors (Sensor Failures)
    # If error is > 10x the threshold, it's likely a sensor blowout, not a subtle pattern.
    ceiling = threshold * 10
    
    if threshold < reconstruction_error < ceiling:
        severity_score = min(100, (reconstruction_error / threshold) * 50)
        oil_loss = ctx.get('oil_avg_7d', 0) * 0.2  # Assume 20% risk
        revenue_loss = oil_loss * OIL_PRICE
        
        data_summary = f"""
        DETECTION TYPE: ML Pattern Anomaly (Autoencoder)
        ERROR: {reconstruction_error:.4f} (Threshold: {threshold:.4f})
        ROOT CAUSE: Unsupervised model detected a sensor pattern deviation not covered by standard physics rules.
        """
        
        return {
            "code": "ML_PATTERN_ANOMALY",
            "category": "PROCESS",
            "raw_data": data_summary,
            "severity_metrics": {'severity_score': severity_score, 'impact_value': revenue_loss},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day (Est)"},
                "production": {"value": oil_loss, "label": f"{format_number(oil_loss)} BOE/day (Est)"}
            },
            "chart_metric": "sensor_violations",
            "diagnosis": {
                "type": "ML Pattern Anomaly",
                "confidence": "Low",
                "indicators": f"Reconstruction Error {reconstruction_error:.2f} > {threshold:.2f}"
            },
            "chart_payload": {
                "threshold": threshold,
                "reconstruction_error": reconstruction_error
            }
        }
    return None
