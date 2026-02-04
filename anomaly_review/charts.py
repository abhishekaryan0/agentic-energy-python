"""
Chart generation for anomaly visualization.

Provides Chart.js configuration for various anomaly types:
- Efficiency trends
- Pressure instability
- Monthly decline
- Mechanical failure
- Ghost production
- Pattern shifts
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger('AnomalyReviewCharts')


# =============================================================================
# CHART CONFIGURATION GENERATION
# =============================================================================

def generate_chart_config(
    anomaly_data: str,
    chart_metric: str,
    df: pd.DataFrame,
    anomaly_code: str = None,
    severity_metrics: Dict = None
) -> Dict:
    """
    Generate Chart.js configuration for anomaly visualization.
    
    Args:
        anomaly_data: Raw anomaly data string
        chart_metric: Type of chart to generate
        df: DataFrame with well data
        anomaly_code: Anomaly type code
        severity_metrics: Severity metrics for annotations
        
    Returns:
        Chart.js configuration dict
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_detection_date(anomaly_data: str) -> Optional[str]:
    """Extract detection date from anomaly data string."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', anomaly_data)
    return match.group(1) if match else None


def prepare_daily_aggregation(df: pd.DataFrame, days: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate data to daily level for charting.
    
    Args:
        df: Input DataFrame
        days: Number of days to include
        
    Returns:
        Tuple of (aggregated DataFrame, labels list)
    """
    start_date = df['timestamp'].max() - pd.Timedelta(days=days)
    recent_df = df[df['timestamp'] >= start_date].copy()
    
    agg_dict = {
        'oil_volume': 'mean',
        'water_volume': 'mean',
        'gas_volume': 'mean'
    }
    
    for col in ['tubing_pressure', 'motor_current', 'strokes_per_minute', 
                'pump_fillage', 'injection_rate', 'casing_pressure', 'predicted_oil']:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    daily_df = recent_df.groupby('date_only').agg(agg_dict).reset_index()
    daily_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in daily_df.columns]
    
    for col in daily_df.columns:
        if col.endswith('_mean'):
            daily_df.rename(columns={col: col.replace('_mean', '')}, inplace=True)

    labels = [d.strftime('%Y-%m-%d') for d in daily_df['date_only']]
    
    return daily_df, labels


def create_anomaly_zone_annotation(detection_date: str, labels: List[str]) -> Dict:
    """Create annotation for anomaly detection point."""
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
    """Create threshold line annotation."""
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


def get_common_options() -> Dict:
    """Get common Chart.js options."""
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


# =============================================================================
# ANOMALY-SPECIFIC CHART GENERATORS
# =============================================================================

def generate_mechanical_failure_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate chart showing divergence between energy input and production output."""
    daily_df, labels = prepare_daily_aggregation(df, days=30)
    
    oil_data = daily_df['oil_volume'].fillna(0).round(1).tolist()
    
    if 'motor_current' in daily_df.columns:
        sensor_data = daily_df['motor_current'].fillna(0).round(1).tolist()
        sensor_label = "Motor Current (Amps)"
        sensor_color = "#ef4444"
    else:
        sensor_data = daily_df.get('tubing_pressure', [0] * len(labels)).tolist()
        sensor_label = "Tubing Pressure (PSI)"
        sensor_color = "#f59e0b"

    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Critical Failure Analysis: Input vs Output', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': 'High energy input with collapsed production indicates mechanical seizure.', 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {
        'type': 'linear', 'position': 'left',
        'title': {'display': True, 'text': 'Oil Production (bbl/day)'},
        'grid': {'color': 'rgba(200, 200, 200, 0.1)'}
    }
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
                {"label": "Oil Production", "data": oil_data, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.2)", "borderWidth": 2, "fill": True, "yAxisID": "y"},
                {"label": sensor_label, "data": sensor_data, "borderColor": sensor_color, "borderWidth": 3, "yAxisID": "y1", "tension": 0.1}
            ]
        },
        "options": options
    }


def generate_efficiency_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate efficiency trend chart."""
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    
    if 'efficiency' not in daily_df.columns and 'predicted_oil' in daily_df.columns and 'oil_volume' in daily_df.columns:
        daily_df['efficiency'] = (daily_df['oil_volume'] / daily_df['predicted_oil'].replace(0, 1)) * 100
        daily_df['efficiency'] = daily_df['efficiency'].clip(0, 120)
    
    actual_oil = daily_df['oil_volume'].fillna(0).round(1).tolist()
    predicted_oil = daily_df.get('predicted_oil', daily_df['oil_volume']).fillna(0).round(1).tolist()
    efficiency = daily_df.get('efficiency', [100] * len(labels)).fillna(100).round(1).tolist()
    
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
        try:
            baseline_efficiency = float(match.group(1))
        except:
            pass
    
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


def generate_baseline_shift_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate baseline shift chart."""
    daily_df, labels = prepare_daily_aggregation(df, days=90)
    actual_data = daily_df['oil_volume'].round(1).fillna(0).tolist()
    
    baseline_val = None
    match = re.search(r"90-DAY HISTORICAL AVG.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try:
            baseline_val = float(match.group(1))
        except:
            pass
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
        current_level = baseline_val * (1 - (severity_metrics.get('decline_pct') / 100))
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


def generate_pressure_instability_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate pressure instability chart."""
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    
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


def generate_monthly_decline_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate monthly production decline chart."""
    start_date = df['timestamp'].max() - pd.Timedelta(days=180)
    monthly_raw = df[df['timestamp'] >= start_date].copy()
    monthly_raw['month_str'] = monthly_raw['timestamp'].dt.to_period('M').astype(str)
    
    monthly_df = monthly_raw.groupby('month_str').agg({
        'oil_volume': 'sum',
        'timestamp': 'count'
    }).reset_index()
    
    monthly_df['days_count'] = (monthly_df['timestamp'] / 24).round(0).clip(lower=1)
    if monthly_df['days_count'].mean() < 32:
        monthly_df['days_count'] = monthly_df['timestamp']
    
    monthly_df['rate'] = (monthly_df['oil_volume'] / monthly_df['days_count']).round(1)
    monthly_df['change_pct'] = monthly_df['rate'].pct_change() * 100
    
    labels = monthly_df['month_str'].tolist()
    rates = monthly_df['rate'].tolist()
    changes = monthly_df['change_pct'].fillna(0).round(1).tolist()
    
    bar_colors = ['#ef4444' if c < -15 else '#f59e0b' if c < 0 else '#10b981' for c in changes]
    
    options = get_common_options()
    options['plugins']['title'] = {'display': True, 'text': 'Monthly Production Decline Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Overall Decline: {severity_metrics.get('decline_pct', 0):.1f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Avg Daily Rate (bbl/d)", "data": rates, "backgroundColor": bar_colors, "borderRadius": 6, "borderWidth": 2, "borderColor": "#ffffff"}
            ]
        },
        "options": options
    }


def generate_pattern_shift_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate pattern shift chart."""
    daily_df, labels = prepare_daily_aggregation(df, days=90)
    
    sensor_col = 'motor_current'
    sensor_label = 'Motor Current (Amps)'
    violation_label = 'Current Anomalies'
    
    if 'tubing' in anomaly_data.lower() or 'pressure' in anomaly_data.lower():
        sensor_col = 'tubing_pressure'
        sensor_label = 'Tubing Pressure (PSI)'
        violation_label = 'Pressure Spikes'
    
    sensor_data = daily_df.get(sensor_col, [0] * len(labels)).fillna(0).round(1).tolist()
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


def generate_ghost_production_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str, 
    severity_metrics: Dict
) -> Dict:
    """Generate ghost production chart."""
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


def generate_default_enhanced_chart(
    df: pd.DataFrame, 
    anomaly_data: str, 
    detection_date: str
) -> Dict:
    """Generate default production trend chart."""
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


def create_fallback_chart(df: pd.DataFrame, error_msg: str) -> Dict:
    """Create fallback chart when generation fails."""
    if df.empty:
        return {
            "type": "line",
            "data": {"labels": [], "datasets": []},
            "options": {
                "plugins": {
                    "title": {"display": True, "text": "No Data Available", "font": {"size": 14, "weight": "bold"}, "color": "#ef4444"},
                    "subtitle": {"display": True, "text": "Insufficient data to generate chart", "font": {"size": 11}, "color": "#64748b"}
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
                "title": {"display": True, "text": "Production Data (Limited View)", "font": {"size": 14}},
                "subtitle": {"display": True, "text": f"Chart generation issue: {error_msg[:50]}", "font": {"size": 10}, "color": "#f59e0b"}
            }
        }
    }
