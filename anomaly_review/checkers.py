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
    MODEL_DIVERGENCE_MIN_PCT,
    ENERGY_DIV_AMPS_RATIO,
    ENERGY_DIV_OIL_MAX_RATIO,
    FLUID_POUND_FILLAGE_BELOW_BASELINE,
    FLUID_POUND_FILLAGE_ABSOLUTE_MAX,
    PRESSURE_INSTABILITY_CV_MIN_PCT,
    MONTHLY_DECLINE_MIN_PCT,
    PATTERN_SHIFT_MULTIPLIER,
    PATTERN_SHIFT_MIN_EVENTS,
    PATTERN_SHIFT_SHUTDOWN_AMPS_MAX,
    PATTERN_SHIFT_MIN_SHUTDOWN_HOURS,
    PATTERN_SHIFT_BASELINE_DAYS,
    PATTERN_SHIFT_RECENT_DAYS,
    PATTERN_SHIFT_MIN_EVENT_HOURS_PRESSURE_SPIKE,
    PATTERN_SHIFT_MIN_EVENT_HOURS_FILLAGE_DROP,
    PATTERN_SHIFT_MIN_EVENT_HOURS_SPM_DROP,
    PATTERN_SHIFT_PRESSURE_SPIKE_ABOVE_BASELINE_RATIO,
    PATTERN_SHIFT_FILLAGE_DROP_BELOW_BASELINE_RATIO,
    PATTERN_SHIFT_SPM_DROP_BELOW,
    DATA_MISMATCH_AMPS_ACTIVE_MIN,
    DATA_MISMATCH_ZERO_VOL_MAX,
    DATA_MISMATCH_BASELINE_VOL_MIN,
    normalize_lift_type,
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
    discharge_pressure = ctx.get('discharge_pressure', 0)
    
    if discharge_pressure > 10:
        pressure = discharge_pressure
        pressure_avg = ctx.get('dp_avg', 0)
    else:
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

    # Seized Pump / Locked Rotor (High Amps + Zero SPM)
    spm = ctx.get('strokes_per_minute', 0)
    if amps > 5 and (spm is None or float(spm) < 0.1):
        return {
            'type': 'Seized Pump / Locked Rotor',
            'confidence': 'High',
            'indicators': f'High Amps ({amps:.1f}) with Zero SPM'
        }

    # Seized Pump / Locked Rotor (high amps ratio, near-zero flow)
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

def _lift_type_key(lift_type) -> str:
    return normalize_lift_type(lift_type).lower().replace(" ", "")


def check_fluid_pound(ctx: Dict) -> Optional[Dict]:
    """
    Rod Pump: Detects Fluid Pound (Shock Loading).
    
    Triggers when pump fillage drops significantly below baseline.
    """
    lift_type = _lift_type_key(ctx.get("lift_type", ""))
    if "rod" not in lift_type:
        logger.debug(f"[FLUID_POUND] skip: lift_type='{lift_type}' (not rod pump)")
        return None

    if pd.isna(ctx.get('pump_fillage')) or pd.isna(ctx.get('strokes_per_minute')):
        logger.debug(f"[FLUID_POUND] skip: pump_fillage or strokes_per_minute missing/NaN")
        return None
    
    # Use rolling average fillage to avoid false positives from transient sensor noise
    # If 24h average is available, use it; otherwise fall back to instantaneous value
    fillage = ctx.get('fillage_avg_24h', ctx.get('pump_fillage'))
    spm = ctx['strokes_per_minute']
    
    # STRICT CHECK: If we have no fillage data at all, skip the check
    if fillage is None or pd.isna(fillage):
        return None
    
    # STRICT CHECK: If we have no history baseline, do not guess.
    # This prevents new wells (or those with missing history) from defaulting to 95% and triggering immediately.
    baseline_fillage = ctx.get('fillage_avg_3d', ctx.get('fillage_avg_24h'))
    
    if baseline_fillage is None:
        logger.debug(f"[FLUID_POUND] skip: no fillage baseline (fillage_avg_3d/24h)")
        return None

    # Anomaly if fillage drops below baseline band AND is below absolute max (configurable "unusual" range)
    threshold_fillage = baseline_fillage * FLUID_POUND_FILLAGE_BELOW_BASELINE
    would_trigger = spm >= 0.0 and fillage < threshold_fillage and fillage < FLUID_POUND_FILLAGE_ABSOLUTE_MAX
    logger.debug(f"[FLUID_POUND_CHECK] Fillage: {fillage}%, Baseline: {baseline_fillage}%, SPM: {spm} (trigger={would_trigger})")

    if not would_trigger:
        logger.debug(
            f"[FLUID_POUND] skip: fillage={fillage}% >= threshold {threshold_fillage:.0f}% or >= absolute max {FLUID_POUND_FILLAGE_ABSOLUTE_MAX}"
        )
    if would_trigger:
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
        
        logger.debug(f"[FLUID_POUND_DETECTED] Data Summary:\n{data_summary}")
        
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
    lift_type = _lift_type_key(ctx.get("lift_type", ""))
    if "esp" not in lift_type:
        return None

    req = ['discharge_pressure', 'pump_intake_pressure', 'motor_current']
    if any(pd.isna(ctx.get(k)) for k in req):
        return None
    
    head = ctx['discharge_pressure'] - ctx['pump_intake_pressure']
    amps = ctx['motor_current']
    
    if head < 200 and amps > (ctx.get('amps_avg', 0) * 0.5):
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
    lift_type = _lift_type_key(ctx.get("lift_type", ""))
    if "gas" not in lift_type:
        return None

    req = ['injection_rate', 'casing_pressure', 'tubing_pressure']
    if any(pd.isna(ctx.get(k)) for k in req):
        return None
    
    rate = ctx['injection_rate']
    cp = ctx['casing_pressure']
    tp = ctx['tubing_pressure']
    
    if rate > (ctx.get('injection_avg_7d', 0) * 1.2) and (cp - tp) < 50:
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
    if pd.isna(oil) or pd.isna(amps) or oil < 0.1 or amps < 1:
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
    
    if pd.isna(predicted) or predicted < 5: # Ignore low production wells
        return None
        
    # Divergence: (Predicted - Actual) / Predicted
    # Positive means Underperformance (Predicted > Actual)
    divergence_pct = safe_divide((predicted - actual), predicted) * 100
    
    if divergence_pct > MODEL_DIVERGENCE_MIN_PCT:
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
        cv = (tp_std / tp_avg) * 100  # Coefficient of Variation
        if cv > PRESSURE_INSTABILITY_CV_MIN_PCT:
            revenue_loss = ctx.get('oil_volume', 0) * 0.15 * OIL_PRICE
            boe_loss = ctx.get('oil_volume', 0) * 0.15
            
            data_summary = f"""
            DETECTION TYPE: Pressure Volatility
            VOLATILITY (CV): {cv:.1f}% (High)
            STD DEV: {tp_std:.1f} psi (Mean Pressure: {tp_avg:.0f} psi)
            CALCULATED_REVENUE_LOSS: ${format_number(revenue_loss)}/day
            """

            return {
                "code": "PRESSURE_INSTABILITY",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity_metrics": {'volatility_pct': cv, 'impact_value': revenue_loss},
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day (Risk)"},
                    "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day"}
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


def _count_events_min_hours(series: pd.Series, min_hours: int) -> int:
    """
    Count contiguous runs where value is True, only if run length >= min_hours.
    Use this so we don't count sensor glitches (e.g. 1 hour of 0) as shutdown;
    only sustained low reading (e.g. 4+ or 24+ hours) counts as one event.
    """
    if series.empty or not series.any():
        return 0
    padded = np.concatenate([[False], series.values.astype(bool), [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) != len(ends):
        return 0
    lengths = ends - starts
    if min_hours < 1:
        return len(lengths)
    return int(np.sum(lengths >= min_hours))


def check_event_frequency(ctx: Dict, full_df: pd.DataFrame = None) -> Optional[Dict]:
    """
    Detects change in operational pattern by comparing counts: e.g. "it used to happen
    once (in baseline window), now it happened 3 times (in recent window)". When the
    anomaly happened (calendar date) does not matter; the count does.

    Shutdown: We only count "shutdown" when motor_current is low for at least 24 hours
    (PATTERN_SHIFT_MIN_SHUTDOWN_HOURS). So sensor down for a day = one event; brief
    glitch or hourly blip does not count.

    Baseline = previous PATTERN_SHIFT_BASELINE_DAYS (90d). Recent = last PATTERN_SHIFT_RECENT_DAYS (30d).
    We compare event counts (recent count vs baseline count normalized to per-30d).
    """
    if full_df is None or 'motor_current' not in full_df.columns:
        return None

    full_df = full_df.sort_values('timestamp').copy()
    last_ts = full_df['timestamp'].max()
    recent_start = last_ts - pd.Timedelta(days=PATTERN_SHIFT_RECENT_DAYS)
    prev_start = recent_start - pd.Timedelta(days=PATTERN_SHIFT_BASELINE_DAYS)
    baseline_label = f"previous {PATTERN_SHIFT_BASELINE_DAYS}d"

    # Add derived columns to full_df before slicing so recent/previous have them
    full_df['is_shutdown'] = full_df['motor_current'] < PATTERN_SHIFT_SHUTDOWN_AMPS_MAX
    if 'tubing_pressure' in full_df.columns:
        tp_median = full_df['tubing_pressure'].median()
        thresh = tp_median * PATTERN_SHIFT_PRESSURE_SPIKE_ABOVE_BASELINE_RATIO
        full_df['is_pressure_spike'] = full_df['tubing_pressure'] > thresh
    if 'pump_fillage' in full_df.columns:
        fill_median = full_df['pump_fillage'].median()
        thresh = fill_median * PATTERN_SHIFT_FILLAGE_DROP_BELOW_BASELINE_RATIO
        full_df['is_fillage_drop'] = full_df['pump_fillage'] < thresh
    if 'strokes_per_minute' in full_df.columns:
        full_df['is_spm_drop'] = full_df['strokes_per_minute'] < PATTERN_SHIFT_SPM_DROP_BELOW

    recent = full_df[full_df['timestamp'] >= recent_start]
    previous = full_df[(full_df['timestamp'] < recent_start) & (full_df['timestamp'] >= prev_start)]

    # 1) Shutdown: only count when low amps last >= 24h (sensor down for a day)
    recent_shutdown = _count_events_min_hours(recent['is_shutdown'], PATTERN_SHIFT_MIN_SHUTDOWN_HOURS)
    prev_shutdown = _count_events_min_hours(previous['is_shutdown'], PATTERN_SHIFT_MIN_SHUTDOWN_HOURS)
    prev_shutdown_per_30d = prev_shutdown / 3.0  # 90d = 3 × 30d

    # 2) Pressure spike, fillage drop, SPM drop (sustained events)
    events_detected = []
    if 'is_pressure_spike' in full_df.columns:
        r_ps = _count_events_min_hours(recent['is_pressure_spike'], PATTERN_SHIFT_MIN_EVENT_HOURS_PRESSURE_SPIKE)
        p_ps = _count_events_min_hours(previous['is_pressure_spike'], PATTERN_SHIFT_MIN_EVENT_HOURS_PRESSURE_SPIKE)
        p_ps_per = p_ps / 3.0
        if r_ps >= 1 and (p_ps_per == 0 or r_ps > PATTERN_SHIFT_MULTIPLIER * p_ps_per):
            events_detected.append(("pressure spike", r_ps, p_ps_per))
    if 'is_fillage_drop' in full_df.columns:
        r_fd = _count_events_min_hours(recent['is_fillage_drop'], PATTERN_SHIFT_MIN_EVENT_HOURS_FILLAGE_DROP)
        p_fd = _count_events_min_hours(previous['is_fillage_drop'], PATTERN_SHIFT_MIN_EVENT_HOURS_FILLAGE_DROP)
        p_fd_per = p_fd / 3.0
        if r_fd >= 1 and (p_fd_per == 0 or r_fd > PATTERN_SHIFT_MULTIPLIER * p_fd_per):
            events_detected.append(("fillage drop", r_fd, p_fd_per))
    if 'is_spm_drop' in full_df.columns:
        r_spm = _count_events_min_hours(recent['is_spm_drop'], PATTERN_SHIFT_MIN_EVENT_HOURS_SPM_DROP)
        p_spm = _count_events_min_hours(previous['is_spm_drop'], PATTERN_SHIFT_MIN_EVENT_HOURS_SPM_DROP)
        p_spm_per = p_spm / 3.0
        if r_spm >= 1 and (p_spm_per == 0 or r_spm > PATTERN_SHIFT_MULTIPLIER * p_spm_per):
            events_detected.append(("SPM drop", r_spm, p_spm_per))

    # Estimate operational risk impact (so every anomaly has a non-zero impact for UI/clarity)
    recent_oil = float(recent["oil_volume"].mean()) if "oil_volume" in recent.columns and not recent["oil_volume"].empty else 0.0
    pattern_impact = recent_oil * 0.10 * OIL_PRICE if recent_oil > 0 else 50.0 * OIL_PRICE  # 10% risk or ~$3750 placeholder
    boe_loss = pattern_impact / OIL_PRICE if OIL_PRICE else 0.0
    impact_label = f"${format_number(pattern_impact)}/day (Operational Risk)" if pattern_impact > 0 else "Operational Review"
    production_impact_label = f"{format_number(boe_loss)} BOE/day (Operational Risk)" if boe_loss > 0 else f"{format_number(boe_loss)} BOE/day"

    # Trigger on shutdown increase (sustained events only)
    shutdown_trigger = recent_shutdown >= 1 and (
        (prev_shutdown_per_30d == 0 and recent_shutdown >= PATTERN_SHIFT_MIN_EVENTS)
        or (prev_shutdown_per_30d > 0 and recent_shutdown > PATTERN_SHIFT_MULTIPLIER * prev_shutdown_per_30d)
    )
    if shutdown_trigger:
        increase_pct = ((recent_shutdown - prev_shutdown_per_30d) / prev_shutdown_per_30d) * 100 if prev_shutdown_per_30d > 0 else 100
        event_desc = "Pump shutdowns (sustained low amps ≥{}h)".format(PATTERN_SHIFT_MIN_SHUTDOWN_HOURS)
        data_summary = f"""
        DETECTION TYPE: Operational pattern change (event frequency)
        EVENT: {event_desc}
        RECENT (30d): {recent_shutdown} event(s)
        BASELINE ({baseline_label}): {prev_shutdown_per_30d:.2f} per 30d
        CHANGE: Activity that was rare is now happening more often (e.g. twice a year → monthly).
        CALCULATED_REVENUE_LOSS: ${format_number(pattern_impact)}/day (operational risk estimate)
        """
        if events_detected:
            data_summary += "\n        Also increased: " + ", ".join(f"{n} (recent {r}, baseline {p:.1f})" for n, r, p in events_detected)
        return {
            "code": "PATTERN_SHIFT",
            "category": "OPERATIONAL",
            "raw_data": data_summary.strip(),
            "severity_metrics": {'frequency_increase_pct': increase_pct, 'impact_value': pattern_impact},
            "impact_value": pattern_impact,
            "impact_payload": {
                "financial": {"value": pattern_impact, "label": impact_label},
                "production": {"value": boe_loss, "label": production_impact_label},
            },
            "chart_metric": "mechanical_failure_view",
            "diagnosis": {"type": "Recurring Operational Events", "confidence": "High", "indicators": f"Shutdown: {recent_shutdown}/30d vs {prev_shutdown_per_30d:.2f}/30d"},
            "chart_payload": {"recent_count": recent_shutdown, "historical_avg": prev_shutdown_per_30d, "increase_pct": increase_pct, "event_types": [e[0] for e in events_detected]},
        }
    # Trigger on other event types only (no shutdown increase)
    if events_detected:
        # Pick the primary event by highest recent count so the UI title isn't biased.
        name, r, p = max(events_detected, key=lambda e: e[1])
        increase_pct = ((r - p) / p) * 100 if p > 0 else 100
        data_summary = f"""
        DETECTION TYPE: Operational pattern change
        EVENT: {name} (sustained events)
        RECENT (30d): {r} | BASELINE ({baseline_label}): {p:.1f} per 30d
        CALCULATED_REVENUE_LOSS: ${format_number(pattern_impact)}/day (operational risk estimate)
        Also: """ + ", ".join(f"{n}=recent {r2}" for n, r2, _ in events_detected)
        return {
            "code": "PATTERN_SHIFT",
            "category": "OPERATIONAL",
            "raw_data": data_summary.strip(),
            "severity_metrics": {'frequency_increase_pct': increase_pct, 'impact_value': pattern_impact},
            "impact_value": pattern_impact,
            "impact_payload": {
                "financial": {"value": pattern_impact, "label": impact_label},
                "production": {"value": boe_loss, "label": production_impact_label},
            },
            "chart_metric": "mechanical_failure_view",
            "diagnosis": {"type": "Recurring Operational Events", "confidence": "Moderate", "indicators": f"Increased: {', '.join(e[0] for e in events_detected)}"},
            "chart_payload": {"recent_count": r, "historical_avg": p, "increase_pct": increase_pct, "event_types": [e[0] for e in events_detected]},
        }
    return None

def check_monthly_production_decline(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    """
    Detects significant month-over-month production decline.
    """
    if 'timestamp' not in df.columns or df['timestamp'].empty:
        return None

    recent_df = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=120))].copy()
    # Use YYYY-MM string to avoid Period timezone warning and keep groupby stable
    recent_df['month'] = recent_df['timestamp'].dt.strftime('%Y-%m')
    
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
            
        if decline_pct > MONTHLY_DECLINE_MIN_PCT:
            revenue_loss = max(0, previous['oil_rate'] - latest['oil_rate']) * OIL_PRICE
            boe_loss = max(0, previous['oil_rate'] - latest['oil_rate'])
            data_summary = f"""
        DETECTION TYPE: Monthly Production Decline
        PREVIOUS MONTH RATE: {format_number(previous['oil_rate'])} bbl/day
        CURRENT MONTH RATE: {format_number(latest['oil_rate'])} bbl/day
        DECLINE: {format_number(decline_pct)}%
        CALCULATED_REVENUE_LOSS: ${format_number(revenue_loss)}/day. Production gap: {format_number(boe_loss)} BOE/day.
        ROOT CAUSE: Cannot be determined from production data alone. Field investigation recommended (pump, inflow, or surface equipment).
        """
            
            return {
                "code": "MONTHLY_DECLINE",
                "category": "PRODUCTION",
                "raw_data": data_summary.strip(),
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
                    "indicators": f"Month-over-month decline {decline_pct:.1f}%. Previous: {format_number(previous['oil_rate'])} bbl/day, current: {format_number(latest['oil_rate'])} bbl/day."
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
    if reconstruction_error is None:
        return None
    try:
        rec_error = float(reconstruction_error)
    except (TypeError, ValueError):
        return None
    if rec_error != rec_error:  # NaN
        return None
    # Goldilocks Filter: Ignore only absurdly massive errors (sensor blowout).
    # Use a large multiple so normal "high" errors (e.g. 25k when threshold=95) still get flagged.
    ceiling = threshold * 1000

    if threshold < rec_error < ceiling:
        severity_score = min(100, (rec_error / threshold) * 50)
        oil_loss = ctx.get('oil_avg_7d', 0) * 0.2  # Assume 20% risk
        revenue_loss = oil_loss * OIL_PRICE

        top_sensors = (ctx.get('top_contributing_features') or "").strip()
        if top_sensors:
            sensor_line = f"\nTOP CONTRIBUTING SENSORS (by reconstruction error): {top_sensors}\nRecommend reviewing trends for these sensors to identify root cause."
            root_cause_section = f"""
        *** ROOT CAUSE ANALYSIS (Feature Contribution) ***
        {top_sensors}
        """
        else:
            sensor_line = "\nRecommend trend review to identify which sensors drove the deviation."
            root_cause_section = "\n*** ROOT CAUSE ANALYSIS (Feature Contribution) ***\n(No feature attribution available.)"

        times_above = (rec_error / threshold) if threshold and threshold > 0 else 0
        readable_summary = f"In plain numbers: the error is **{times_above:.0f}× above threshold** ({format_number(rec_error)} vs {format_number(threshold)}). The model does NOT diagnose root cause—it only shows which sensors contributed most. Use the ROOT CAUSE ANALYSIS list below to know what to check. Do NOT invent causes (e.g. 'gas lift', 'pressure management', 'flow instability'); only report the sensor list and recommend reviewing those trends."

        data_summary = f"""
        DETECTION TYPE: ML Pattern Anomaly (Autoencoder)

        WHAT THIS ERROR MEANS (explain this to the user in plain language):
        The model learned the normal pattern of how this well's sensors (pressure, amps, production, etc.) typically move together. "Reconstruction error" measures how much the current readings deviate from that normal; a high value means the well is behaving differently than usual. This metric does not tell you the root cause—only which sensors contributed most.

        READABLE SUMMARY (use this in your description):
        {readable_summary}

        ERROR: {rec_error:.4f} (Threshold: {threshold:.4f})  →  {times_above:.0f}× above threshold
        {sensor_line}
        {root_cause_section}
        """

        diagnosis_indicators = f"Reconstruction Error {rec_error:.2f} > {threshold:.2f}"
        if top_sensors:
            diagnosis_indicators += f". Root cause (feature contribution): {top_sensors}"

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
                "indicators": diagnosis_indicators,
                "top_contributing_sensors": top_sensors or None,
            },
            "chart_payload": {
                "threshold": threshold,
                "reconstruction_error": rec_error,
                "top_contributing_features": top_sensors or None,
            }
        }
    return None

def check_energy_production_divergence(ctx: Dict) -> Optional[Dict]:
    """
    Detects significant divergence between Energy Input (Amps) and Production Output (Oil).

    Previously "Mechanical Failure". Now "Energy/Production Divergence".
    Uses baselines as of 1 day ago when available so the current day's anomaly does not dilute the baseline.
    """
    # Prefer lagged baselines (1d ago) so current anomalous values don't inflate/deflate the baseline
    oil_baseline = ctx.get('oil_baseline_90d_1d_ago') if not pd.isna(ctx.get('oil_baseline_90d_1d_ago')) else ctx.get('oil_baseline_90d')
    amps_avg = ctx.get('amps_avg_1d_ago') if not pd.isna(ctx.get('amps_avg_1d_ago')) else ctx.get('amps_avg')
    if any(pd.isna(ctx.get(k)) for k in ['motor_current', 'oil_volume']) or pd.isna(oil_baseline) or pd.isna(amps_avg):
        logger.debug(
            f"[ENERGY_DIV] skip: motor_current={ctx.get('motor_current')}, oil_volume={ctx.get('oil_volume')}, "
            f"oil_baseline={oil_baseline}, amps_avg={amps_avg} (missing or NaN)"
        )
        return None

    amps_current = ctx['motor_current']
    oil_current = ctx['oil_volume']
    amps_ratio = amps_current / amps_avg if amps_avg and amps_avg > 0 else 0
    oil_ratio = oil_current / oil_baseline if oil_baseline and oil_baseline > 0 else 1

    # High Amps (working hard) but Low Production (no results)
    # Thresholds from config: only flag when clearly outside normal range
    if amps_current > (ENERGY_DIV_AMPS_RATIO * amps_avg) and oil_current < (ENERGY_DIV_OIL_MAX_RATIO * oil_baseline):
        revenue_loss = oil_baseline * OIL_PRICE
        boe_loss = oil_baseline
        
        data_summary = f"""
        DETECTION TYPE: Energy/Production Divergence
        ENERGY INPUT: {format_number(amps_current)} Amps (Baseline: {format_number(amps_avg)})
        PRODUCTION OUTPUT: {format_number(oil_current)} bbl (Baseline: {format_number(oil_baseline)})
        DIVERGENCE: Input +{format_number((amps_current/amps_avg - 1)*100)}%, Output -{format_number((1 - oil_current/oil_baseline)*100)}%
        CALCULATED_REVENUE_LOSS: ${format_number(revenue_loss)}/day (Potential). Production gap: {format_number(boe_loss)} BOE/day.
        ROOT CAUSE: High energy input with low production suggests mechanical inefficiency (pump, drive, or wellbore). Use diagnosis below.
        """

        diagnosis = diagnose_mechanics(ctx)

        return {
            "code": "ENERGY_DIV",
            "category": "OPERATIONAL",
            "raw_data": data_summary.strip(),
            "severity_metrics": {'severity_score': 70, 'impact_value': revenue_loss},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day (Potential)"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day (Potential)"}
            },
            "chart_metric": "mechanical_failure_view",
            "diagnosis": diagnosis,
            "chart_payload": {
                "amps": amps_current,
                "oil": oil_current,
                "efficiency": (oil_current / amps_current) if amps_current > 0 else 0
            }
        }
    logger.debug(
        f"[ENERGY_DIV] skip: amps_ratio={amps_ratio:.2f} (need >{ENERGY_DIV_AMPS_RATIO}) or "
        f"oil_ratio={oil_ratio:.2f} (need <{ENERGY_DIV_OIL_MAX_RATIO})"
    )
    return None

def check_sensor_state_mismatch(ctx: Dict) -> Optional[Dict]:
    """
    Detects when operational state (Amps/SPM) says 'Running' but Production says 'Zero'.
    Previously "Ghost Production". Now "Sensor State Mismatch".
    """
    # Coerce None to 0 so comparisons don't raise (ctx can have None for missing/NaN)
    _f = lambda x: 0.0 if x is None or (isinstance(x, float) and pd.isna(x)) else float(x)
    total_vol = _f(ctx.get('oil_volume')) + _f(ctx.get('water_volume'))
    baseline_vol = _f(ctx.get('oil_baseline_90d')) + _f(ctx.get('water_avg_30d'))
    amps = _f(ctx.get('motor_current'))
    spm = _f(ctx.get('strokes_per_minute'))

    # Active if Amps or SPM above configurable minimum
    is_active = amps > DATA_MISMATCH_AMPS_ACTIVE_MIN or spm > 1.0
    if is_active and total_vol < DATA_MISMATCH_ZERO_VOL_MAX and baseline_vol > DATA_MISMATCH_BASELINE_VOL_MIN:
        revenue_loss = _f(ctx.get('oil_baseline_90d')) * OIL_PRICE
        boe_loss = _f(ctx.get('oil_baseline_90d'))
        
        data_summary = f"""
        DETECTION TYPE: Sensor State Mismatch (Ghost Production)
        STATUS: Pump is Running (Amps {format_number(amps)}, SPM {format_number(spm)})
        FLOW: No Flow Recorded ({format_number(total_vol)} bbl). Baseline expected: {format_number(baseline_vol)} bbl.
        ROOT CAUSE: Operational state indicates active pump but flow meter reads zero. Likely meter calibration, data sync, or sensor fault—not a mechanical well failure.
        CALCULATED_REVENUE_LOSS: ${format_number(revenue_loss)}/day (Risk)
        """
        
        return {
            "code": "DATA_MISMATCH",
            "category": "PRODUCTION",
            "raw_data": data_summary.strip(),
            "severity_metrics": {'impact_value': revenue_loss, 'severity_score': 60},
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${format_number(revenue_loss)}/day (Risk)"},
                "production": {"value": boe_loss, "label": f"{format_number(boe_loss)} BOE/day (Risk)"}
            },
            "chart_metric": "ghost_production_view",
            "diagnosis": {
                "type": "Data Mismatch",
                "confidence": "High",
                "indicators": "Active Pump State vs Zero Flow"
            },
            "chart_payload": {
                "amps": amps,
                "total_volume": total_vol
            }
        }
    return None
