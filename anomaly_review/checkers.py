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
    efficiency = (oil / pred_oil) * 100 if pred_oil > 0.1 else 100
    amps_ratio = amps / amps_avg if amps_avg > 0.1 else 1.0
    pressure_ratio = pressure / pressure_avg if pressure_avg > 0.1 else 1.0
    
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
    
    Signature of seized pump, locked rotor, or parted rods.
    """
    oil_current = ctx.get('oil_volume', 0)
    oil_baseline = ctx.get('oil_baseline_90d', 1)
    amps_current = ctx.get('motor_current', 0)
    amps_avg = ctx.get('amps_avg', 1)

    # Case: Seized/Stuck (High Amps, No Oil)
    if amps_current > (1.5 * amps_avg) and oil_current < (0.2 * oil_baseline):
        oil_loss = oil_baseline
        revenue_loss = oil_baseline * OIL_PRICE
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


def check_efficiency_degradation(ctx: Dict) -> Optional[Dict]:
    """
    Detects efficiency drops compared to predicted production.
    """
    pred = ctx.get('predicted_oil', 0)
    actual = ctx.get('oil_volume', 0)
    
    if pred > 5 and actual >= 0:
        eff = (actual / pred) * 100
        avg_eff = ctx.get('efficiency_avg_30d', 100)
        drop = avg_eff - eff
        
        if drop > 15:
            revenue_loss = (pred - actual) * OIL_PRICE
            boe_loss = pred - actual

            return {
                "code": "EFFICIENCY_DEGRADATION",
                "category": "PRODUCTION",
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
    Detects pressure volatility/slugging.
    
    Ignores massive step-changes (crashes) which are handled elsewhere.
    """
    tp_avg = ctx.get('tp_avg', 0)
    tp_std = ctx.get('tp_std', 0)
    tp_current = ctx.get('tubing_pressure', 0)
    
    if tp_avg > 50:
        volatility_pct = (tp_std / tp_avg) * 100
        
        # Ignore step changes (>50% shift from mean)
        deviation_from_mean = abs(tp_current - tp_avg) / tp_avg
        if deviation_from_mean > 0.5:
            return None

        if volatility_pct > 22 and tp_std > 40:
            revenue_loss = ctx.get('oil_volume', 0) * 0.10 * OIL_PRICE
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
                "diagnosis": {
                    "type": "Slug Flow / Instability",
                    "confidence": "Moderate",
                    "indicators": f"High Volatility {format_number(volatility_pct)}%"
                }
            }
    return None


def check_ghost_production(ctx: Dict) -> Optional[Dict]:
    """
    Detects ghost production: pump active but zero output.
    
    Indicates meter failure or parted rods.
    """
    is_active = (ctx.get('strokes_per_minute', 0) > 0.1) or (ctx.get('motor_current', 0) > 5)
    total_vol = ctx.get('oil_volume', 0) + ctx.get('water_volume', 0)
    baseline_vol = ctx.get('oil_baseline_90d', 1)
    
    if is_active and total_vol < 0.05 and baseline_vol > 0.5:
        revenue_loss = ctx.get('oil_avg_30d', 0) * OIL_PRICE
        oil_comp = ctx.get('oil_avg_30d', 0)
        gas_comp = ctx.get('gas_avg_30d', 0)
        boe_loss = oil_comp + (gas_comp / 6)

        return {
            "code": "GHOST_PROD",
            "category": "PRODUCTION",
            "raw_data": "Pump Active, Zero Volume",
            "severity_metrics": {'impact_value': revenue_loss, 'severity_score': 100},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
            },
            "chart_metric": "production_bar",
            "diagnosis": {
                "type": "Meter Failure or Parted Rods",
                "confidence": "High",
                "indicators": "Active Pump, 0 Vol"
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
    monthly['oil_rate'] = monthly['oil_total'] / monthly['oil_days']
    
    if len(monthly) >= 2:
        latest = monthly.iloc[-1]
        previous = monthly.iloc[-2]
        
        if latest['oil_days'] < 3:
            return None

        if previous['oil_rate'] > 5:
            decline_pct = ((previous['oil_rate'] - latest['oil_rate']) / previous['oil_rate']) * 100
            
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
                        "type": "Accelerated Monthly Decline",
                        "confidence": "Low",
                        "indicators": "Comparison of monthly production rates."
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
    if reconstruction_error > threshold:
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
                "type": "Unclassified Operational Shift",
                "confidence": "Low",
                "indicators": f"Reconstruction Error {reconstruction_error:.2f} > {threshold:.2f}"
            }
        }
    return None
