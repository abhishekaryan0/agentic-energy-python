"""
Configuration for the anomaly review module.

Contains all constants, thresholds, and configuration values used
throughout the anomaly review package.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# =============================================================================
# PRICING CONSTANTS (Keep static as requested)
# =============================================================================

OIL_PRICE = 75.00
WATER_DISPOSAL_COST = 1.50
GAS_PRICE = 2.50

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================

DETECTION_WINDOW_DAYS = 5
TRAINING_CUTOFF_DAYS = 7
ROLLING_BASELINE_DAYS = 90
MODEL_CACHE_DAYS = 7
MODEL_DIR = "models"

# =============================================================================
# DETECTION THRESHOLDS (What counts as "unusual")
# =============================================================================
# Narrow these so we only flag clear deviations from normal. If ranges are too
# wide, everything falls "in range" and anomalies never trigger.

# Model divergence: trigger when actual production is this far below predicted (%).
MODEL_DIVERGENCE_MIN_PCT = 25.0

# Energy/Production divergence: high amps + low oil vs baseline.
# Trigger when amps > (AMPS_RATIO * baseline) AND oil < (OIL_MAX_RATIO * baseline).
ENERGY_DIV_AMPS_RATIO = 1.35
ENERGY_DIV_OIL_MAX_RATIO = 0.28

# Fluid pound: trigger when fillage < (FILLAGE_BELOW_BASELINE * baseline) and < FILLAGE_ABSOLUTE_MAX.
FLUID_POUND_FILLAGE_BELOW_BASELINE = 0.72
FLUID_POUND_FILLAGE_ABSOLUTE_MAX = 82.0

# Pressure instability: trigger when coefficient of variation (std/mean) > this %.
PRESSURE_INSTABILITY_CV_MIN_PCT = 18.0

# Monthly decline: trigger when recent vs previous decline exceeds this %.
MONTHLY_DECLINE_MIN_PCT = 15.0

# Pattern shift: trigger when recent event count > (PATTERN_SHIFT_MULTIPLIER * baseline).
PATTERN_SHIFT_MULTIPLIER = 2.0
PATTERN_SHIFT_MIN_EVENTS = 2

# Sensor state mismatch: "active" if amps > this or SPM > 1; "zero flow" if total_vol < this.
DATA_MISMATCH_AMPS_ACTIVE_MIN = 5.0
DATA_MISMATCH_ZERO_VOL_MAX = 0.05
DATA_MISMATCH_BASELINE_VOL_MIN = 0.5

# =============================================================================
# SENSOR METADATA
# =============================================================================

SENSOR_DISPLAY_NAMES: Dict[str, str] = {
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

# =============================================================================
# SEVERITY THRESHOLDS
# =============================================================================

SEVERITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
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
    "ML_PATTERN_ANOMALY": {"critical": 80, "high": 60, "moderate": 40},
    "MECHANICAL_FAILURE": {"critical": 100, "high": 80, "moderate": 60},
}

# =============================================================================
# SUPPRESSION RULES (Hierarchy of Truth)
# =============================================================================

# This system prevents alert fatigue by suppressing "Symptom" anomalies 
# when a specific "Root Cause" anomaly is detected on the same day.

SUPPRESSION_RULES: Dict[str, List[str]] = {
    # Top-level rule
    'MECHANICAL_FAILURE': [
        'FLUID_POUND', 'PUMP_WEAR', 'EFFICIENCY_DEGRADATION', 
        'MONTHLY_DECLINE', 'PRESSURE_INSTABILITY', 'ML_PATTERN_ANOMALY'
    ],
    # Existing rules
    'FLUID_POUND': [
        'EFFICIENCY_DEGRADATION', 'MONTHLY_DECLINE', 
        'PRESSURE_INSTABILITY', 'ML_PATTERN_ANOMALY'
    ],
    'PUMP_WEAR': [
        'EFFICIENCY_DEGRADATION', 'MONTHLY_DECLINE', 'ML_PATTERN_ANOMALY'
    ],
    'GAS_HOG': [
        'EFFICIENCY_DEGRADATION', 'MONTHLY_DECLINE', 
        'PRESSURE_INSTABILITY', 'ML_PATTERN_ANOMALY'
    ],
    # Updated rules
    'PRESSURE_INSTABILITY': ['MONTHLY_DECLINE', 'EFFICIENCY_DEGRADATION'],
    'EFFICIENCY_DEGRADATION': ['MONTHLY_DECLINE'],
}

# Priority Ranking (1 = Highest Priority / Most Specific)
ANOMALY_PRIORITY: Dict[str, int] = {
    'GHOST_PROD': 1,
    'MECHANICAL_FAILURE': 1,
    'FLUID_POUND': 2,
    'PUMP_WEAR': 2,
    'GAS_HOG': 2,
    'EFFICIENCY_DEGRADATION': 3,
    'PRESSURE_INSTABILITY': 3,
    'ML_PATTERN_ANOMALY': 4,
    'MONTHLY_DECLINE': 5
}

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

LIFT_TYPE_FEATURES: Dict[str, List[str]] = {
    "Rod Pump": [
        "strokes_per_minute", "pump_fillage", "motor_current", 
        "tubing_pressure", "casing_pressure", "torque",
        "polish_rod_load", "surface_stroke_length", "pump_intake_pressure",
        "wellhead_temp"
    ],
    "ESP": [
        "drive_frequency", "motor_current", "motor_temp", "motor_voltage",
        "pump_intake_pressure", "discharge_pressure", "vibration_x", "vibration_y",
        "intake_fluid_temp", "wellhead_temp", "surface_pressure"
    ],
    "Gas Lift": [
        "injection_rate", "tubing_pressure", "casing_pressure",
        "injection_pressure", "surface_pressure", "bottomhole_pressure"
    ]
}

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    test_mode: bool = True
    test_well_limit: int = 1
    test_well_id: str = "Well_AnomalyTest_RodPump"
    max_workers: int = 10
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.test_well_limit < 1:
            raise ValueError(f"test_well_limit must be >= 1, got {self.test_well_limit}")


@dataclass
class AnomalyReviewConfig:
    """Main configuration for anomaly review."""
    detection_window_days: int = DETECTION_WINDOW_DAYS
    training_cutoff_days: int = TRAINING_CUTOFF_DAYS
    rolling_baseline_days: int = ROLLING_BASELINE_DAYS
    model_cache_days: int = MODEL_CACHE_DAYS
    model_dir: str = MODEL_DIR
    
    # Pricing (static)
    oil_price: float = OIL_PRICE
    water_disposal_cost: float = WATER_DISPOSAL_COST
    gas_price: float = GAS_PRICE
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.detection_window_days < 1:
            raise ValueError(f"detection_window_days must be >= 1")
        if self.oil_price <= 0:
            raise ValueError(f"oil_price must be > 0")


# Default configuration instances
DEFAULT_CONFIG = AnomalyReviewConfig()
DEFAULT_BATCH_CONFIG = BatchConfig()


def format_number(value, metric_name=None) -> str:
    """
    Format numbers for display with proper formatting.
    
    Rules:
    1. Values < 1: Use 1 decimal place (e.g., 0.5 bbl)
    2. Strokes Per Minute: Use 1 decimal place (e.g., 4.5 SPM)
    3. All others: Comma delimited, no decimals (e.g., 1,000)
    """
    import pandas as pd
    
    if pd.isna(value):
        return "N/A"
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        return str(value)

    # Rule 1: Values < 1
    if abs(val_float) < 1 and val_float != 0:
        return f"{val_float:,.1f}"
    
    # Rule 2: Strokes Per Minute
    if metric_name and ('stroke' in str(metric_name).lower() or 'spm' in str(metric_name).lower()):
        return f"{val_float:,.1f}"
    
    # Rule 3: Standard format
    return f"{val_float:,.0f}"
