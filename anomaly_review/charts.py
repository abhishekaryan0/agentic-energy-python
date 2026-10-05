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
    severity_metrics: Dict = None,
    chart_payload: Dict = None
) -> Dict:
    """
    Generate Chart.js configuration for anomaly visualization.
    
    Args:
        anomaly_data: Raw anomaly data string
        chart_metric: Type of chart to generate
        df: DataFrame with well data
        anomaly_code: Anomaly type code
        severity_metrics: Severity metrics for annotations
        chart_payload: Structured data from checkers (replaces regex)
        
    Returns:
        Chart.js configuration dict
    """
    if df.empty:
        return create_fallback_chart(df, "No Data")
    
    if chart_payload is None:
        chart_payload = {}

    try:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date_only'] = df['timestamp'].dt.date
        
        # Prefer event_date from payload (set by service), then extract from raw text
        detection_date = chart_payload.get('event_date')
        if detection_date is not None:
            detection_date = str(detection_date)
        else:
            detection_date = extract_detection_date(anomaly_data)

        # Never chart "future" days after the recommendation/detection date.
        df = _clip_dataframe_to_as_of(df, detection_date)
        
        # Route to anomaly-specific chart generators
        if anomaly_code == "MECHANICAL_FAILURE" or chart_metric == "mechanical_failure_view":
            return generate_mechanical_failure_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)

        # ML pattern anomalies: show the sensors that drive the finding (not only reconstruction error).
        if anomaly_code == "ML_PATTERN_ANOMALY":
            return generate_ml_driver_sensor_chart(df, anomaly_data, detection_date, chart_payload)

        if chart_metric == "monthly_production":
            return generate_monthly_decline_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)
        elif chart_metric == "baseline_shift":
            return generate_baseline_shift_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)
        elif chart_metric == "efficiency_trend":
            return generate_efficiency_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)
        elif chart_metric == "tubing_pressure" or chart_metric == "pressure_trend":
            return generate_pressure_instability_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)
        elif chart_metric == "sensor_violations":
            return generate_pattern_shift_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)
        elif chart_metric == "production_bar":
            return generate_ghost_production_chart(df, anomaly_data, detection_date, severity_metrics, chart_payload)
        else:
            return generate_default_enhanced_chart(df, anomaly_data, detection_date)
            
    except Exception as e:
        logger.error(f"Enhanced chart generation failed: {e}")
        return create_fallback_chart(df, str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _clip_dataframe_to_as_of(df: pd.DataFrame, as_of_date: Optional[str]) -> pd.DataFrame:
    """Keep only rows on/before as_of_date so charts cannot show data after the recommendation."""
    if df is None or df.empty or not as_of_date:
        return df
    try:
        as_of = pd.to_datetime(str(as_of_date)[:10])
        ts = pd.to_datetime(df["timestamp"])
        return df.loc[ts.dt.normalize() <= as_of.normalize()].copy()
    except Exception:
        return df


def extract_detection_date(anomaly_data: str) -> Optional[str]:
    """Extract detection date from anomaly data string."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', anomaly_data or "")
    return match.group(1) if match else None


def _parse_driver_sensor_columns(text: str, available_columns) -> List[str]:
    """
    Map narrative / feature-attribution text to dataframe column names.
    Handles forms like: 'Discharge Pressure (42%), Motor Temp (30%)' or plain sensor names.
    """
    available = {str(c).lower(): str(c) for c in available_columns}
    aliases = {
        "discharge pressure": "discharge_pressure",
        "motor temperature": "motor_temp",
        "motor temp": "motor_temp",
        "drive frequency": "drive_frequency",
        "freq hz": "freq_hz",
        "frequency": "drive_frequency",
        "motor current": "motor_current",
        "amps": "motor_current",
        "injection rate": "injection_rate",
        "injection pressure": "injection_pressure",
        "water cut": "water_cut",
        "pump fillage": "pump_fillage",
        "fillage": "pump_fillage",
        "intake pressure": "intake_pressure",
        "tubing pressure": "tubing_pressure",
        "casing pressure": "casing_pressure",
        "surface pressure": "surface_pressure",
        "strokes per minute": "strokes_per_minute",
        "spm": "strokes_per_minute",
        "oil volume": "oil_volume",
        "water volume": "water_volume",
        "gas volume": "gas_volume",
    }
    found: List[str] = []
    raw = (text or "").lower()

    # Prefer explicit "Name (NN%)" attribution fragments first.
    for match in re.finditer(r"([a-z0-9][a-z0-9\s/_-]{1,40}?)\s*\(\d{1,3}%\)", raw):
        phrase = match.group(1).strip()
        col = aliases.get(phrase) or phrase.replace(" ", "_").replace("/", "_")
        if col in available and available[col] not in found:
            found.append(available[col])

    for phrase, col in aliases.items():
        if phrase in raw and col in available and available[col] not in found:
            found.append(available[col])

    return found[:4]


def prepare_daily_aggregation(df: pd.DataFrame, days: int = 60, custom_agg: Dict = None, as_of_date: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate data to daily level for charting.
    
    Args:
        df: Input DataFrame
        days: Number of days to include
        custom_agg: Optional dictionary to override default aggregation (e.g., {'pump_fillage': 'min'})
        as_of_date: Optional YYYY-MM-DD; chart window ends on this date (no future points)
        
    Returns:
        Tuple of (aggregated DataFrame, labels list)
    """
    df = _clip_dataframe_to_as_of(df, as_of_date)
    if df.empty:
        return df, []

    end_ts = df['timestamp'].max()
    if as_of_date:
        try:
            end_ts = min(end_ts, pd.to_datetime(str(as_of_date)[:10]) + pd.Timedelta(hours=23, minutes=59))
        except Exception:
            pass
    start_date = end_ts - pd.Timedelta(days=days)
    recent_df = df[df['timestamp'] >= start_date].copy()
    
    agg_dict = {}
    for col in ['oil_volume', 'water_volume', 'gas_volume']:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    for col in ['tubing_pressure', 'motor_current', 'strokes_per_minute', 
                'pump_fillage', 'injection_rate', 'casing_pressure', 'predicted_oil', 'tp_std',
                'discharge_pressure', 'motor_temp', 'drive_frequency', 'freq_hz',
                'intake_pressure', 'surface_pressure', 'injection_pressure']:
        if col in df.columns:
            agg_dict[col] = 'mean'
            
    # Apply custom overrides
    if custom_agg:
        for col, method in custom_agg.items():
            if col in df.columns:
                agg_dict[col] = method
    
    if not agg_dict:
        return pd.DataFrame(), []

    daily_df = recent_df.groupby('date_only').agg(agg_dict).reset_index()
    daily_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in daily_df.columns]
    
    for col in daily_df.columns:
        if col.endswith('_mean'):
            daily_df.rename(columns={col: col.replace('_mean', '')}, inplace=True)

    # Smart renaming for min/max: Only rename to base if no other variant exists
    for col in list(daily_df.columns):
        if col.endswith('_min'):
            base = col[:-4] # strip _min
            siblings = [c for c in daily_df.columns if c.startswith(base) and c != col]
            if base not in daily_df.columns and not any(c.startswith(base) for c in siblings):
                daily_df.rename(columns={col: base}, inplace=True)
        elif col.endswith('_max'):
            base = col[:-4] # strip _max
            siblings = [c for c in daily_df.columns if c.startswith(base) and c != col]
            if base not in daily_df.columns and not any(c.startswith(base) for c in siblings):
                daily_df.rename(columns={col: base}, inplace=True)

    labels = [d.strftime('%Y-%m-%d') for d in daily_df['date_only']]
    
    return daily_df, labels


def _filter_annotations(annotations: List[Dict]) -> List[Dict]:
    """Remove empty annotation dicts so Chart.js plugin does not receive [{}]."""
    return [a for a in annotations if a]


def create_anomaly_zone_annotation(detection_date: str, labels: List[str]) -> Dict:
    """Vertical line and label at the date the anomaly was detected."""
    if not detection_date or detection_date not in labels:
        return {}
    detection_index = labels.index(detection_date)
    return {
        "type": "line",
        "xMin": detection_index,
        "xMax": detection_index,
        "borderColor": CHART_COLORS["alert"],
        "borderWidth": 2,
        "borderDash": [6, 4],
        "label": {
            "display": True,
            "content": "Anomaly",
            "position": "start",
            "backgroundColor": CHART_COLORS["alert"],
            "color": "#ffffff",
            "font": {"size": 11, "weight": "bold"},
        },
    }


def create_threshold_band(labels: List[str], threshold_value: float, label: str, color: str = None) -> Dict:
    """Horizontal reference line (e.g. baseline or threshold)."""
    color = color or CHART_COLORS["baseline"]
    return {
        "type": "line",
        "yMin": threshold_value,
        "yMax": threshold_value,
        "borderColor": color,
        "borderWidth": 2,
        "borderDash": [8, 4],
        "label": {
            "display": True,
            "content": label,
            "position": "end",
            "backgroundColor": color,
            "color": "#ffffff",
            "font": {"size": 10},
        },
    }


# Shared palette for consistent, accessible charts
CHART_COLORS = {
    "oil": "#2563eb",           # Blue - production
    "water": "#0ea5e9",        # Light blue
    "gas": "#f59e0b",          # Amber
    "baseline": "#059669",      # Green - normal/baseline
    "alert": "#dc2626",         # Red - anomaly/alert
    "warning": "#d97706",      # Amber - warning
    "predicted": "#64748b",    # Slate - model/secondary
    "grid": "rgba(148, 163, 184, 0.12)",
}

def get_common_options() -> Dict:
    """Common Chart.js options: readable fonts, clear tooltips, subtle grid."""
    return {
        "responsive": True,
        "maintainAspectRatio": False,
        "aspectRatio": 2.2,
        "interaction": {"mode": "index", "intersect": False},
        "plugins": {
            "legend": {
                "position": "top",
                "labels": {
                    "usePointStyle": True,
                    "padding": 18,
                    "font": {"size": 12},
                    "boxWidth": 14,
                },
            },
            "tooltip": {
                "usePointStyle": True,
                "backgroundColor": "rgba(15, 23, 42, 0.96)",
                "titleColor": "#f8fafc",
                "bodyColor": "#e2e8f0",
                "borderColor": "#475569",
                "borderWidth": 1,
                "padding": 14,
                "titleFont": {"size": 13, "weight": "bold"},
                "bodyFont": {"size": 12},
                "displayColors": True,
            },
        },
        "scales": {
            "x": {
                "grid": {"display": False, "drawBorder": True},
                "ticks": {
                    "maxTicksLimit": 12,
                    "maxRotation": 40,
                    "minRotation": 0,
                    "font": {"size": 11},
                    "autoSkip": True,
                },
                "title": {"display": False},
            },
            "y": {
                "grid": {"color": CHART_COLORS["grid"]},
                "beginAtZero": False,
                "ticks": {"font": {"size": 11}},
            },
        },
        "elements": {
            "point": {"radius": 0, "hitRadius": 24, "hoverRadius": 6, "hoverBorderWidth": 2},
            "line": {"borderJoinStyle": "round", "tension": 0.2},
        },
    }


# =============================================================================
# ANOMALY-SPECIFIC CHART GENERATORS
# =============================================================================

def generate_mechanical_failure_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Energy input (amps/pressure) vs production output — high input with low output suggests a mechanical issue."""
    custom_agg = {"motor_current": "max", "tubing_pressure": "max", "strokes_per_minute": "max"}
    daily_df, labels = prepare_daily_aggregation(df, days=30, custom_agg=custom_agg)

    oil_data = pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist()
    use_motor = (
        "motor_current" in daily_df.columns
        and pd.to_numeric(daily_df["motor_current"], errors="coerce").fillna(0).gt(0).any()
    )
    if use_motor:
        sensor_data = pd.to_numeric(daily_df["motor_current"], errors="coerce").fillna(0).round(1).tolist()
        sensor_label = "Peak motor current (A)"
        sensor_color = CHART_COLORS["alert"]
    else:
        sensor_data = (
            pd.to_numeric(daily_df["tubing_pressure"], errors="coerce").fillna(0).round(1).tolist()
            if "tubing_pressure" in daily_df.columns else [0] * len(labels)
        )
        sensor_label = "Peak tubing pressure (psi)"
        sensor_color = CHART_COLORS["warning"]

    annotations = _filter_annotations([create_anomaly_zone_annotation(detection_date, labels)])
    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {
        "display": True,
        "text": "Energy input vs production",
        "font": {"size": 16, "weight": "bold"},
    }
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": "Left: oil (bbl/day). Right: peak amps or pressure. Divergence (high input, low output) may indicate a mechanical problem.",
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {
        "type": "linear",
        "position": "left",
        "beginAtZero": True,
        "title": {"display": True, "text": "Oil (bbl/day)"},
        "grid": {"color": CHART_COLORS["grid"]},
    }
    options["scales"]["y1"] = {
        "type": "linear",
        "position": "right",
        "title": {"display": True, "text": sensor_label},
        "grid": {"drawOnChartArea": False},
    }

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Oil (bbl/day)", "data": oil_data, "borderColor": CHART_COLORS["oil"], "backgroundColor": "rgba(37, 99, 235, 0.15)", "borderWidth": 2, "fill": True, "yAxisID": "y"},
                {"label": sensor_label, "data": sensor_data, "borderColor": sensor_color, "borderWidth": 2, "yAxisID": "y1", "tension": 0.2},
            ],
        },
        "options": options,
    }


def generate_efficiency_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Pump fillage or efficiency vs production. Fillage below the green line may indicate fluid pound."""
    custom_agg = {"pump_fillage": "min"}
    daily_df, labels = prepare_daily_aggregation(df, days=45, custom_agg=custom_agg)

    if "efficiency" not in daily_df.columns and "predicted_oil" in daily_df.columns and "oil_volume" in daily_df.columns:
        daily_df["efficiency"] = (daily_df["oil_volume"] / daily_df["predicted_oil"].replace(0, 1)) * 100
        daily_df["efficiency"] = daily_df["efficiency"].clip(0, 120)

    actual_oil = pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist()
    predicted_oil = (
        pd.to_numeric(daily_df["predicted_oil"], errors="coerce").fillna(0).round(1).tolist()
        if "predicted_oil" in daily_df.columns else actual_oil
    )
    efficiency = (
        pd.to_numeric(daily_df["efficiency"], errors="coerce").fillna(100).round(1).tolist()
        if "efficiency" in daily_df.columns else [100] * len(labels)
    )

    if "FLUID POUND" in anomaly_data.upper() or chart_payload.get("baseline_fillage") is not None:
        if "pump_fillage" in daily_df.columns:
            efficiency = pd.to_numeric(daily_df["pump_fillage"], errors="coerce").fillna(0).round(1).tolist()
        title_text = "Pump fillage & production"
        y_axis_text = "Pump fillage (%)"
    else:
        title_text = "Efficiency & production"
        y_axis_text = "Efficiency (%)"

    baseline_efficiency = 100
    if chart_payload and chart_payload.get("baseline_fillage"):
        baseline_efficiency = chart_payload["baseline_fillage"]
    else:
        match = re.search(r"HISTORICAL EFFICIENCY.*?:.*?([\d\.]+)", anomaly_data)
        if match:
            try:
                baseline_efficiency = float(match.group(1))
            except Exception:
                pass

    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    if 0 < baseline_efficiency < 900:
        annotations.append(create_threshold_band(labels, baseline_efficiency, f"Normal ({baseline_efficiency:.0f}%)"))
    annotations = _filter_annotations(annotations)

    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {"display": True, "text": title_text, "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": "Green dashed line = normal level. Values below it may indicate fluid pound or efficiency loss.",
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {
        "type": "linear",
        "position": "left",
        "min": 0,
        "max": 120,
        "title": {"display": True, "text": y_axis_text},
        "grid": {"color": CHART_COLORS["grid"]},
    }
    options["scales"]["y1"] = {
        "type": "linear",
        "position": "right",
        "beginAtZero": True,
        "title": {"display": True, "text": "Oil (bbl/day)"},
        "grid": {"drawOnChartArea": False},
    }

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Actual oil (bbl/day)", "data": actual_oil, "borderColor": CHART_COLORS["oil"], "backgroundColor": "rgba(37, 99, 235, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y1"},
                {"label": "Predicted oil (bbl/day)", "data": predicted_oil, "borderColor": CHART_COLORS["predicted"], "borderDash": [5, 5], "borderWidth": 1.5, "yAxisID": "y1"},
                {"label": y_axis_text, "data": efficiency, "borderColor": "#7c3aed", "borderWidth": 2, "yAxisID": "y"},
            ],
        },
        "options": options,
    }


def generate_baseline_shift_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Daily production vs historical baseline. Decline below the green line suggests a performance shift."""
    custom_agg = {"oil_volume": ["mean", "min", "max"]}
    daily_df, labels = prepare_daily_aggregation(df, days=90, custom_agg=custom_agg)

    actual_data = pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist()
    min_data = (
        pd.to_numeric(daily_df["oil_volume_min"], errors="coerce").fillna(0).round(1).tolist()
        if "oil_volume_min" in daily_df.columns else actual_data
    )
    max_data = (
        pd.to_numeric(daily_df["oil_volume_max"], errors="coerce").fillna(0).round(1).tolist()
        if "oil_volume_max" in daily_df.columns else actual_data
    )

    baseline_val = None
    match = re.search(r"90-DAY HISTORICAL AVG.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try:
            baseline_val = float(match.group(1))
        except Exception:
            pass
    if baseline_val is None or baseline_val == 0:
        baseline_val = float(daily_df["oil_volume"].quantile(0.75))

    baseline_line = [baseline_val] * len(labels)
    rolling_avg = (
        daily_df["oil_volume"].rolling(7, min_periods=3).mean().bfill().round(1).tolist()
        if len(daily_df) >= 7 else actual_data
    )

    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    annotations.append(create_threshold_band(labels, baseline_val, f"Baseline ({baseline_val:.0f} bbl/day)"))
    decline_pct = (severity_metrics or {}).get("decline_pct", 0) or 0
    if decline_pct > 25:
        current_level = baseline_val * (1 - (decline_pct / 100))
        annotations.append(create_threshold_band(labels, current_level, "New level", CHART_COLORS["alert"]))
    annotations = _filter_annotations(annotations)

    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {"display": True, "text": "Production vs baseline", "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": f"Green line = historical baseline. Decline: {decline_pct:.1f}%.",
        "font": {"size": 11},
        "color": "#64748b" if decline_pct < 15 else CHART_COLORS["alert"],
    }
    options["scales"]["y"] = {"beginAtZero": True, "title": {"display": True, "text": "Oil (bbl/day)"}, "grid": {"color": CHART_COLORS["grid"]}}

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Daily max", "data": max_data, "backgroundColor": "rgba(37, 99, 235, 0.08)", "borderColor": "transparent", "borderWidth": 0, "fill": "+1", "pointRadius": 0},
                {"label": "Daily min", "data": min_data, "backgroundColor": "transparent", "borderWidth": 0, "fill": False, "pointRadius": 0},
                {"label": "Daily production", "data": actual_data, "borderColor": CHART_COLORS["oil"], "backgroundColor": "rgba(37, 99, 235, 0.12)", "borderWidth": 2, "fill": False},
                {"label": "7-day average", "data": rolling_avg, "borderColor": "#0ea5e9", "borderWidth": 2, "pointRadius": 0},
                {"label": "Baseline", "data": baseline_line, "borderColor": CHART_COLORS["baseline"], "borderWidth": 2, "borderDash": [8, 4], "pointRadius": 0},
            ],
        },
        "options": options,
    }


def generate_pressure_instability_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Pressure (avg and swing range) vs oil production. Shaded area = pressure volatility (avg ± 2× std)."""
    daily_df, labels = prepare_daily_aggregation(df, days=45, custom_agg={"tp_std": "max"})

    if "discharge_pressure" in daily_df.columns:
        pressure_avg = pd.to_numeric(daily_df["discharge_pressure"], errors="coerce").fillna(0).round(1).tolist()
        label_text = "Discharge pressure"
        pressure_std = pd.to_numeric(daily_df["tp_std"], errors="coerce").fillna(0).tolist() if "tp_std" in daily_df.columns else [0] * len(labels)
    else:
        pressure_avg = (
            pd.to_numeric(daily_df["tubing_pressure"], errors="coerce").fillna(0).round(1).tolist()
            if "tubing_pressure" in daily_df.columns else [0] * len(labels)
        )
        label_text = "Tubing pressure"
        pressure_std = pd.to_numeric(daily_df["tp_std"], errors="coerce").fillna(0).tolist() if "tp_std" in daily_df.columns else [0] * len(labels)

    upper_band = [avg + 2 * std for avg, std in zip(pressure_avg, pressure_std)]
    lower_band = [max(0, avg - 2 * std) for avg, std in zip(pressure_avg, pressure_std)]
    oil_production = pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist()

    annotations = _filter_annotations([create_anomaly_zone_annotation(detection_date, labels)])
    volatility_pct = (severity_metrics or {}).get("volatility_pct", 0) or (chart_payload or {}).get("volatility_pct", 0)

    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {"display": True, "text": "Pressure stability & production", "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": f"Left: {label_text} (red line = avg, shaded = swing range). Right: oil. Volatility: {volatility_pct:.1f}%.",
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {"type": "linear", "position": "left", "beginAtZero": True, "title": {"display": True, "text": f"{label_text} (psi)"}, "grid": {"color": CHART_COLORS["grid"]}}
    options["scales"]["y1"] = {"type": "linear", "position": "right", "beginAtZero": True, "title": {"display": True, "text": "Oil (bbl/day)"}, "grid": {"drawOnChartArea": False}}

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Pressure range (high)", "data": upper_band, "backgroundColor": "rgba(220, 38, 38, 0.12)", "borderColor": "transparent", "borderWidth": 0, "fill": "+1", "pointRadius": 0, "yAxisID": "y"},
                {"label": "Pressure range (low)", "data": lower_band, "backgroundColor": "transparent", "borderWidth": 0, "fill": False, "pointRadius": 0, "yAxisID": "y"},
                {"label": f"Avg {label_text}", "data": pressure_avg, "borderColor": CHART_COLORS["alert"], "borderWidth": 2, "yAxisID": "y", "tension": 0.2},
                {"label": "Oil (bbl/day)", "data": oil_production, "borderColor": CHART_COLORS["oil"], "backgroundColor": "rgba(37, 99, 235, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y1"},
            ],
        },
        "options": options,
    }


def generate_monthly_decline_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Monthly average daily oil rate. Bar color: green = stable, amber = slight drop, red = >15% decline."""
    start_date = df["timestamp"].max() - pd.Timedelta(days=180)
    monthly_raw = df[df["timestamp"] >= start_date].copy()
    monthly_raw["month_str"] = monthly_raw["timestamp"].dt.strftime("%Y-%m")

    monthly_df = monthly_raw.groupby("month_str").agg({"oil_volume": "sum", "timestamp": "count"}).reset_index()
    monthly_df["days_count"] = (monthly_df["timestamp"] / 24).round(0).clip(lower=1)
    if monthly_df["days_count"].mean() < 32:
        monthly_df["days_count"] = monthly_df["timestamp"]
    monthly_df["rate"] = (monthly_df["oil_volume"] / monthly_df["days_count"]).round(1)
    monthly_df["change_pct"] = monthly_df["rate"].pct_change() * 100

    labels = monthly_df["month_str"].tolist()
    rates = monthly_df["rate"].tolist()
    changes = monthly_df["change_pct"].fillna(0).round(1).tolist()

    if len(rates) >= 3:
        x_vals = np.arange(len(rates))
        coeffs = np.polyfit(x_vals, rates, 1)
        trend_line = [coeffs[0] * x + coeffs[1] for x in x_vals]
    else:
        trend_line = rates

    bar_colors = [CHART_COLORS["alert"] if c < -15 else CHART_COLORS["warning"] if c < 0 else CHART_COLORS["baseline"] for c in changes]
    overall_decline = (severity_metrics or {}).get("decline_pct", 0) or (chart_payload or {}).get("decline_pct", 0)

    options = get_common_options()
    options["plugins"]["title"] = {"display": True, "text": "Monthly oil production", "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": f"Bars = avg daily rate (bbl/day). Color: green = stable, amber = slight drop, red = >15% decline. Trend line = overall direction. Decline: {overall_decline:.1f}%.",
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {"beginAtZero": True, "title": {"display": True, "text": "Oil (bbl/day)"}, "grid": {"color": CHART_COLORS["grid"]}}

    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Avg daily rate (bbl/day)", "data": rates, "backgroundColor": bar_colors, "borderRadius": 6, "borderWidth": 1, "borderColor": "#e2e8f0"},
                {"label": "Trend", "data": trend_line, "type": "line", "borderColor": CHART_COLORS["alert"], "borderWidth": 2, "borderDash": [6, 4], "pointRadius": 0, "fill": False},
            ],
        },
        "options": options,
    }


def generate_ml_driver_sensor_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    chart_payload: Dict = None,
) -> Dict:
    """
    ML pattern anomaly chart: plot the sensors that drive the finding.

    Priority:
    1) Top contributing sensors from model attribution / narrative
    2) Oil / water for context
    3) Reconstruction error only as a secondary reference series (not the only story)
    Chart window ends on detection_date (no future points).
    """
    chart_payload = chart_payload or {}
    df = _clip_dataframe_to_as_of(df, detection_date)
    if df.empty:
        return create_fallback_chart(df, "No data as of recommendation date")

    attribution = (
        chart_payload.get("top_contributing_features")
        or chart_payload.get("top_contributing_sensors")
        or ""
    )
    search_text = f"{attribution}\n{anomaly_data or ''}"
    driver_cols = _parse_driver_sensor_columns(search_text, df.columns)

    custom_agg = {c: "mean" for c in driver_cols}
    for prod in ("oil_volume", "water_volume"):
        if prod in df.columns:
            custom_agg[prod] = "mean"
    if "reconstruction_error" in df.columns:
        custom_agg["reconstruction_error"] = "max"

    daily_df, labels = prepare_daily_aggregation(
        df, days=60, custom_agg=custom_agg or None, as_of_date=detection_date
    )
    if not labels:
        return create_fallback_chart(df, "No chartable daily data")

    palette = [
        CHART_COLORS.get("alert", "#dc2626"),
        CHART_COLORS.get("warning", "#f59e0b"),
        CHART_COLORS.get("predicted", "#64748b"),
        "#7c3aed",
        "#0891b2",
    ]
    datasets = []
    highlighted = []
    for i, col in enumerate(driver_cols):
        if col not in daily_df.columns:
            continue
        label = col.replace("_", " ").title()
        highlighted.append(label)
        datasets.append(
            {
                "label": label,
                "data": pd.to_numeric(daily_df[col], errors="coerce").fillna(0).round(2).tolist(),
                "borderColor": palette[i % len(palette)],
                "backgroundColor": "transparent",
                "borderWidth": 2,
                "yAxisID": "y",
                "tension": 0.2,
            }
        )

    # Production context
    if "oil_volume" in daily_df.columns:
        datasets.append(
            {
                "label": "Oil (bbl/day)",
                "data": pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist(),
                "borderColor": CHART_COLORS.get("oil", "#2563eb"),
                "backgroundColor": "rgba(37, 99, 235, 0.08)",
                "borderWidth": 2,
                "fill": False,
                "yAxisID": "y1",
                "tension": 0.2,
            }
        )
    if "water_volume" in daily_df.columns and (
        "water" in search_text.lower() or "injection" in search_text.lower() or not datasets
    ):
        datasets.append(
            {
                "label": "Water (bbl/day)",
                "data": pd.to_numeric(daily_df["water_volume"], errors="coerce").fillna(0).round(1).tolist(),
                "borderColor": "#0ea5e9",
                "backgroundColor": "transparent",
                "borderWidth": 1.5,
                "borderDash": [4, 3],
                "yAxisID": "y1",
                "tension": 0.2,
            }
        )

    # If we still have no driver sensors, fall back to reconstruction-only view.
    if not any(d.get("yAxisID") == "y" for d in datasets) and "reconstruction_error" in daily_df.columns:
        return generate_ml_reconstruction_chart(df, anomaly_data, detection_date, chart_payload)

    if "reconstruction_error" in daily_df.columns and datasets:
        datasets.append(
            {
                "label": "Reconstruction error (reference)",
                "data": pd.to_numeric(daily_df["reconstruction_error"], errors="coerce").fillna(0).round(1).tolist(),
                "borderColor": "#94a3b8",
                "borderWidth": 1,
                "borderDash": [6, 4],
                "pointRadius": 0,
                "yAxisID": "y2",
                "fill": False,
            }
        )

    annotations = _filter_annotations([create_anomaly_zone_annotation(detection_date, labels)])
    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}

    driver_text = ", ".join(highlighted) if highlighted else "key sensors"
    as_of = detection_date or "recommendation date"
    options["plugins"]["title"] = {
        "display": True,
        "text": f"Sensors driving this recommendation ({driver_text})",
        "font": {"size": 16, "weight": "bold"},
    }
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": (
            f"Chart data ends on {as_of} (recommendation/detection date). "
            "Series match the sensors named in the finding so you can verify the relationship visually."
        ),
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {
        "type": "linear",
        "position": "left",
        "title": {"display": True, "text": "Sensor values"},
        "grid": {"color": CHART_COLORS["grid"]},
    }
    options["scales"]["y1"] = {
        "type": "linear",
        "position": "right",
        "title": {"display": True, "text": "Production (bbl/day)"},
        "grid": {"drawOnChartArea": False},
    }
    if any(d.get("yAxisID") == "y2" for d in datasets):
        options["scales"]["y2"] = {
            "type": "linear",
            "position": "right",
            "display": False,
            "grid": {"drawOnChartArea": False},
        }

    return {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": options,
        "meta": {
            "highlighted_parameters": highlighted,
            "recommendation_date": detection_date,
            "data_as_of": labels[-1] if labels else detection_date,
            "continuously_updated": False,
            "note": "Snapshot as of recommendation/detection date; not live-updating after creation.",
        },
    }


def generate_ml_reconstruction_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    chart_payload: Dict = None,
) -> Dict:
    """Fallback: reconstruction error only when driver sensors are unavailable."""
    chart_payload = chart_payload or {}
    threshold = chart_payload.get("threshold")
    if threshold is None:
        threshold = 0.0
    threshold = float(threshold)

    df = _clip_dataframe_to_as_of(df, detection_date)
    df = df.copy()
    df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date
    if "reconstruction_error" not in df.columns:
        return create_fallback_chart(df, "No reconstruction error data")
    daily = (
        df.groupby("date_only")["reconstruction_error"]
        .max()
        .reset_index()
        .sort_values("date_only")
    )
    if daily.empty:
        return create_fallback_chart(df, "No reconstruction error data")
    labels = [d.strftime("%Y-%m-%d") for d in daily["date_only"]]
    rec_data = pd.to_numeric(daily["reconstruction_error"], errors="coerce").fillna(0).round(1).tolist()

    options = get_common_options()
    options["plugins"]["title"] = {
        "display": True,
        "text": "How much does the sensor pattern match 'normal'?",
        "font": {"size": 16, "weight": "bold"},
    }
    as_of = detection_date or (labels[-1] if labels else "n/a")
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": (
            f"Data through {as_of}. Reconstruction error = deviation from learned normal. "
            f"Values above the red line ({threshold:,.0f}) mean unusual behavior."
        ),
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {
        "type": "linear",
        "position": "left",
        "beginAtZero": True,
        "title": {"display": True, "text": "Reconstruction error (deviation from normal pattern)"},
        "grid": {"color": CHART_COLORS["grid"]},
    }

    annotations = []
    if detection_date and detection_date in labels:
        ann = create_anomaly_zone_annotation(detection_date, labels)
        if ann:
            annotations.append(ann)
    if threshold is not None and threshold > 0:
        annotations.append(
            create_threshold_band(labels, threshold, f"Threshold ({threshold:,.0f})", CHART_COLORS["alert"])
        )
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Reconstruction error",
                    "data": rec_data,
                    "borderColor": CHART_COLORS["predicted"],
                    "backgroundColor": "rgba(100, 116, 139, 0.12)",
                    "borderWidth": 2,
                    "fill": True,
                },
            ],
        },
        "options": options,
        "meta": {
            "recommendation_date": detection_date,
            "data_as_of": labels[-1] if labels else detection_date,
            "continuously_updated": False,
        },
    }


def generate_pattern_shift_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Daily peak motor current or pressure. Shows when operational pattern (e.g. shutdowns) changed."""
    custom_agg = {"motor_current": "max", "tubing_pressure": "max"}
    daily_df, labels = prepare_daily_aggregation(df, days=90, custom_agg=custom_agg)

    sensor_col = "motor_current"
    sensor_label = "Peak motor current (A)"
    if "tubing" in anomaly_data.lower() or "pressure" in anomaly_data.lower():
        sensor_col = "tubing_pressure"
        sensor_label = "Peak tubing pressure (psi)"
    # If motor_current is present but all zeros, use tubing_pressure for a meaningful chart
    if (
        sensor_col == "motor_current"
        and "motor_current" in daily_df.columns
        and "tubing_pressure" in daily_df.columns
    ):
        mc_series = pd.to_numeric(daily_df["motor_current"], errors="coerce").fillna(0)
        if not mc_series.gt(0).any():
            sensor_col = "tubing_pressure"
            sensor_label = "Peak tubing pressure (psi)"

    sensor_data = (
        pd.to_numeric(daily_df[sensor_col], errors="coerce").fillna(0).round(1).tolist()
        if sensor_col in daily_df.columns else [0] * len(labels)
    )

    annotations = _filter_annotations([create_anomaly_zone_annotation(detection_date, labels)])
    freq_increase = (severity_metrics or {}).get("frequency_increase_pct", 0) or (chart_payload or {}).get("increase_pct", 0)

    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {"display": True, "text": "Operational pattern (daily peak)", "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": f"{sensor_label} per day. Frequency increase: {freq_increase:.0f}%.",
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {"type": "linear", "position": "left", "beginAtZero": True, "title": {"display": True, "text": sensor_label}, "grid": {"color": CHART_COLORS["grid"]}}

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": sensor_label, "data": sensor_data, "borderColor": CHART_COLORS["predicted"], "backgroundColor": "rgba(100, 116, 139, 0.08)", "borderWidth": 2, "fill": True, "yAxisID": "y"},
            ],
        },
        "options": options,
    }


def generate_ghost_production_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
    severity_metrics: Dict,
    chart_payload: Dict = None,
) -> Dict:
    """Stacked daily production (oil, water, gas). Red line = days when pump was active (amps or SPM). Mismatch suggests meter or data issue."""
    custom_agg = {"motor_current": "max", "strokes_per_minute": "max"}
    daily_df, labels = prepare_daily_aggregation(df, days=30, custom_agg=custom_agg)

    def _to_num_list(ser: pd.Series) -> list:
        return pd.to_numeric(ser, errors="coerce").fillna(0).round(1).tolist()

    oil_data = _to_num_list(daily_df["oil_volume"])
    water_data = _to_num_list(daily_df["water_volume"]) if "water_volume" in daily_df.columns else [0] * len(labels)
    gas_data = _to_num_list(daily_df["gas_volume"]) if "gas_volume" in daily_df.columns else [0] * len(labels)
    motor_current = _to_num_list(daily_df["motor_current"]) if "motor_current" in daily_df.columns else [0] * len(labels)
    spm = _to_num_list(daily_df["strokes_per_minute"]) if "strokes_per_minute" in daily_df.columns else [0] * len(labels)

    pump_running = [1 if (mc > 5 or s > 0.1) else 0 for mc, s in zip(motor_current, spm)]
    max_vol = max(sum([oil_data, water_data, gas_data], []) + [10])
    pump_running_scaled = [pr * max_vol * 0.12 for pr in pump_running]

    annotations = _filter_annotations([create_anomaly_zone_annotation(detection_date, labels)])
    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {"display": True, "text": "Production vs equipment activity", "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {
        "display": True,
        "text": "Stacked bars = daily production (oil, water, gas). Red dashed line = days when pump was active. No bars but red line = possible meter or data issue.",
        "font": {"size": 11},
        "color": "#64748b",
    }
    options["scales"]["y"] = {"stacked": True, "beginAtZero": True, "title": {"display": True, "text": "Volume (bbl/day)"}, "grid": {"color": CHART_COLORS["grid"]}}

    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Oil", "data": oil_data, "backgroundColor": CHART_COLORS["baseline"], "borderRadius": 4, "stack": "production"},
                {"label": "Water", "data": water_data, "backgroundColor": CHART_COLORS["water"], "borderRadius": 4, "stack": "production"},
                {"label": "Gas", "data": gas_data, "backgroundColor": CHART_COLORS["gas"], "borderRadius": 4, "stack": "production"},
                {"label": "Pump active (indicator)", "data": pump_running_scaled, "type": "line", "borderColor": CHART_COLORS["alert"], "borderWidth": 2, "borderDash": [6, 4], "fill": False, "pointRadius": 0, "yAxisID": "y"},
            ],
        },
        "options": options,
    }


def generate_default_enhanced_chart(
    df: pd.DataFrame,
    anomaly_data: str,
    detection_date: str,
) -> Dict:
    """Default: daily oil production over the last 45 days."""
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    oil_data = pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist()

    annotations = _filter_annotations([create_anomaly_zone_annotation(detection_date, labels)])
    options = get_common_options()
    if annotations:
        options["plugins"]["annotation"] = {"annotations": annotations}
    options["plugins"]["title"] = {"display": True, "text": "Oil production trend", "font": {"size": 16, "weight": "bold"}}
    options["plugins"]["subtitle"] = {"display": True, "text": "Daily oil (bbl/day). Red line = when anomaly was detected.", "font": {"size": 11}, "color": "#64748b"}
    options["scales"]["y"] = {"beginAtZero": True, "title": {"display": True, "text": "Oil (bbl/day)"}, "grid": {"color": CHART_COLORS["grid"]}}

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Oil (bbl/day)", "data": oil_data, "borderColor": CHART_COLORS["oil"], "backgroundColor": "rgba(37, 99, 235, 0.12)", "borderWidth": 2, "fill": True, "tension": 0.2},
            ],
        },
        "options": options,
    }


def create_fallback_chart(df: pd.DataFrame, error_msg: str) -> Dict:
    """Fallback when the main chart cannot be generated."""
    if df.empty:
        return {
            "type": "line",
            "data": {"labels": [], "datasets": []},
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": "No data available", "font": {"size": 15, "weight": "bold"}, "color": CHART_COLORS["alert"]},
                    "subtitle": {"display": True, "text": "Insufficient data to generate this chart.", "font": {"size": 12}, "color": "#64748b"},
                },
            },
        }

    daily_df, labels = prepare_daily_aggregation(df, days=30)
    data = pd.to_numeric(daily_df["oil_volume"], errors="coerce").fillna(0).round(1).tolist()

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Oil (bbl/day)",
                "data": data,
                "borderColor": CHART_COLORS["predicted"],
                "backgroundColor": "rgba(100, 116, 139, 0.1)",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.2,
            }],
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {"display": True, "text": "Production (limited view)", "font": {"size": 15}},
                "subtitle": {"display": True, "text": error_msg[:80] + ("..." if len(error_msg) > 80 else ""), "font": {"size": 11}, "color": CHART_COLORS["warning"]},
            },
            "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Oil (bbl/day)"}}},
        },
    }
