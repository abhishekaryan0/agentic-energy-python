"""
Configuration for the anomaly review module.

Contains all constants, thresholds, and configuration values used
throughout the anomaly review package.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def get_organization_id() -> Optional[str]:
    """Organization UUID from env (from your organizations Postgres). Used for tenant scoping."""
    return os.getenv("ORGANIZATION_ID")

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
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
# Autoencoder anomaly threshold = mean + (this * std) on training reconstruction error. Lower = more sensitive (e.g. 1.0 for digital twin / testing).
AUTOENCODER_THRESHOLD_SIGMA = 1.5
# How many days of well data we fetch for anomaly review (baselines, models, pattern-shift).
# Pattern-shift compares last 30d vs previous 90d within this window (no full-year history).
ANOMALY_REVIEW_FETCH_DAYS = int(os.getenv("ANOMALY_REVIEW_FETCH_DAYS", "180"))
# S3 layout (aligned with operations and deploy): input/output prefixes
# Input: s3://BUCKET/{ORG}/sensor_data/well_sensor_readings.parquet, daily_production.parquet
# Output: s3://BUCKET/{ORG}/detections/anomaly_review/run_ts=.../anomaly_reviews.parquet
def get_anomaly_review_s3_bucket() -> str:
    """S3 bucket for anomaly review input/output (same as operations)."""
    return os.getenv("S3_BUCKET", "")


def get_anomaly_review_input_prefix() -> str:
    """S3 prefix for sensor/production parquet (e.g. org_id/sensor_data)."""
    org = get_organization_id()
    if org:
        return f"{org}/sensor_data"
    return os.getenv("ANOMALY_REVIEW_INPUT_S3_PREFIX", "sensor_data")


def get_anomaly_review_output_prefix() -> str:
    """S3 prefix for anomaly review output parquet (e.g. org_id/detections/anomaly_review)."""
    org = get_organization_id()
    if org:
        return f"{org}/detections/anomaly_review"
    return os.getenv("ANOMALY_REVIEW_OUTPUT_S3_PREFIX", "detections/anomaly_review")


def get_anomaly_review_model_s3_base() -> Optional[str]:
    """
    S3 base prefix for anomaly review trained model (SageMaker job output).
    Uses ORGANIZATION_ID from env: s3://BUCKET/{ORGANIZATION_ID}/model/.
    Returns None if S3_BUCKET or ORGANIZATION_ID not set.
    Override with ANOMALY_REVIEW_MODEL_S3_URI to pin a specific model.tar.gz URI.
    """
    override = os.getenv("ANOMALY_REVIEW_MODEL_S3_URI", "").strip()
    if override:
        return override if override.startswith("s3://") else f"s3://{override}"
    bucket = get_anomaly_review_s3_bucket()
    org = get_organization_id()
    if not bucket or not org:
        return None
    return f"s3://{bucket}/{org}/model"

# =============================================================================
# DETECTION THRESHOLDS (What counts as "unusual")
# =============================================================================
# Narrow these so we only flag clear deviations from normal. If ranges are too
# wide, everything falls "in range" and anomalies never trigger.

# Model divergence: trigger when actual production is this far below predicted (%).
MODEL_DIVERGENCE_MIN_PCT = 25.0

# Energy/Production divergence: high amps + low oil vs baseline.
# Trigger when amps > (AMPS_RATIO * baseline) AND oil < (OIL_MAX_RATIO * baseline).
ENERGY_DIV_AMPS_RATIO = 1.15
ENERGY_DIV_OIL_MAX_RATIO = 0.28

# Fluid pound: trigger when fillage < (FILLAGE_BELOW_BASELINE * baseline) and < FILLAGE_ABSOLUTE_MAX.
FLUID_POUND_FILLAGE_BELOW_BASELINE = 0.85
FLUID_POUND_FILLAGE_ABSOLUTE_MAX = 82.0

# Pressure instability: trigger when coefficient of variation (std/mean) > this %.
PRESSURE_INSTABILITY_CV_MIN_PCT = 18.0

# Monthly decline: trigger when recent vs previous decline exceeds this %.
MONTHLY_DECLINE_MIN_PCT = 15.0

# Pattern shift: trigger when recent event count > (PATTERN_SHIFT_MULTIPLIER * baseline).
PATTERN_SHIFT_MULTIPLIER = 2.0
PATTERN_SHIFT_MIN_EVENTS = 2
# What counts as "shutdown" for event counting: motor_current below this (Amps).
PATTERN_SHIFT_SHUTDOWN_AMPS_MAX = 1.0
# Only count a "shutdown event" when motor current is off for at least this many hours (avoids brief glitches).
# 24 = about a day; sensor down for a day counts as one shutdown, not hourly blips.
PATTERN_SHIFT_MIN_SHUTDOWN_HOURS = 24
# Don't create a new row for these codes if one exists in the last N days; instead update that row so it stays visible (still severe).
PATTERN_SHIFT_SUPPRESS_DAYS = 7
ANOMALY_CODES_UPDATE_WHEN_RECENT: List[str] = ["PATTERN_SHIFT"]
# Compare counts: recent window vs baseline window. We care that "it used to happen X times, now it happens Y times" (count, not calendar alignment).
PATTERN_SHIFT_BASELINE_DAYS = 90
PATTERN_SHIFT_RECENT_DAYS = 30
# Other event types: min contiguous hours to count (sustained event, not spike).
PATTERN_SHIFT_MIN_EVENT_HOURS_PRESSURE_SPIKE = 2
PATTERN_SHIFT_MIN_EVENT_HOURS_FILLAGE_DROP = 4
PATTERN_SHIFT_MIN_EVENT_HOURS_SPM_DROP = 4
# Thresholds: pressure spike = tubing_pressure > baseline_30d * (1 + this). Fillage drop = below baseline * this. SPM drop = below this value.
PATTERN_SHIFT_PRESSURE_SPIKE_ABOVE_BASELINE_RATIO = 1.25
PATTERN_SHIFT_FILLAGE_DROP_BELOW_BASELINE_RATIO = 0.75
PATTERN_SHIFT_SPM_DROP_BELOW = 2.0

# Sensor state mismatch: "active" if amps > this or SPM > 1; "zero flow" if total_vol < this.
DATA_MISMATCH_AMPS_ACTIVE_MIN = 5.0
DATA_MISMATCH_ZERO_VOL_MAX = 0.05
DATA_MISMATCH_BASELINE_VOL_MIN = 0.5

# =============================================================================
# SENSOR METADATA
# =============================================================================

SENSOR_DISPLAY_NAMES: Dict[str, str] = {
    "oil_volume": "Oil Volume",
    "water_volume": "Water Volume",
    "gas_volume": "Gas Volume",
    "strokes_per_minute": "Strokes Per Minute (SPM)",
    "torque": "Torque",
    "pump_fillage": "Pump Fillage",
    "tubing_pressure": "Tubing Pressure",
    "pump_intake_pressure": "Pump Intake Pressure",
    "casing_pressure": "Casing Pressure",
    "wellhead_temp": "Wellhead Temperature",
    "motor_current": "Motor Current",
    "discharge_pressure": "Discharge Pressure",
    "intake_fluid_temp": "Intake Fluid Temperature",
    "injection_rate": "Injection Rate",
    "injection_pressure": "Injection Pressure",
    "surface_pressure": "Surface Pressure",
    "vibration_x": "Vibration X",
    "vibration_y": "Vibration Y",
    "downhole_net_stroke": "Downhole Net Stroke",
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
# When anomaly A is present, any code listed in SUPPRESSION_RULES[A] is hidden.
# Order: root cause wins; same-issue or symptom codes are suppressed so you
# get one alert per underlying issue, not duplicates.

SUPPRESSION_RULES: Dict[str, List[str]] = {
    # Most specific / critical first
    "MECHANICAL_FAILURE": [
        "FLUID_POUND",
        "PUMP_WEAR",
        "EFFICIENCY_DEGRADATION",
        "MONTHLY_DECLINE",
        "PRESSURE_INSTABILITY",
        "PATTERN_SHIFT",
        "ML_PATTERN_ANOMALY",
    ],
    "DATA_MISMATCH": ["ML_PATTERN_ANOMALY", "PATTERN_SHIFT"],
    "MODEL_DIVERGENCE": ["ML_PATTERN_ANOMALY"],
    "ENERGY_DIV": ["ML_PATTERN_ANOMALY", "PATTERN_SHIFT"],
    # Physics / operational
    "FLUID_POUND": [
        "EFFICIENCY_DEGRADATION",
        "MONTHLY_DECLINE",
        "PRESSURE_INSTABILITY",
        "PATTERN_SHIFT",
        "ML_PATTERN_ANOMALY",
    ],
    "PUMP_WEAR": [
        "EFFICIENCY_DEGRADATION",
        "MONTHLY_DECLINE",
        "PATTERN_SHIFT",
        "ML_PATTERN_ANOMALY",
    ],
    "GAS_HOG": [
        "EFFICIENCY_DEGRADATION",
        "MONTHLY_DECLINE",
        "PRESSURE_INSTABILITY",
        "PATTERN_SHIFT",
        "ML_PATTERN_ANOMALY",
    ],
    # One pressure alert: volatility (root) hides spike-frequency (pattern shift)
    "PRESSURE_INSTABILITY": [
        "MONTHLY_DECLINE",
        "EFFICIENCY_DEGRADATION",
        "PATTERN_SHIFT",
        "ML_PATTERN_ANOMALY",
    ],
    "EFFICIENCY_DEGRADATION": ["MONTHLY_DECLINE", "ML_PATTERN_ANOMALY"],
    "PATTERN_SHIFT": ["ML_PATTERN_ANOMALY"],
    "MONTHLY_DECLINE": ["ML_PATTERN_ANOMALY"],
}

# Priority Ranking (1 = Highest Priority / Most Specific)
ANOMALY_PRIORITY: Dict[str, int] = {
    "GHOST_PROD": 1,
    "MECHANICAL_FAILURE": 1,
    "DATA_MISMATCH": 1,
    "FLUID_POUND": 2,
    "PUMP_WEAR": 2,
    "GAS_HOG": 2,
    "ENERGY_DIV": 2,
    "EFFICIENCY_DEGRADATION": 3,
    "PRESSURE_INSTABILITY": 3,
    "PATTERN_SHIFT": 4,
    "ML_PATTERN_ANOMALY": 4,
    "MONTHLY_DECLINE": 5,
    "MODEL_DIVERGENCE": 5,
}

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

LIFT_TYPE_FEATURES: Dict[str, List[str]] = {
    "Rod Pump": [
        "strokes_per_minute",
        "pump_fillage",
        "motor_current",
        "tubing_pressure",
        "casing_pressure",
        "torque",
        "polish_rod_load",
        "surface_stroke_length",
        "pump_intake_pressure",
        "wellhead_temp",
    ],
    "ESP": [
        "drive_frequency",
        "motor_current",
        "motor_temp",
        "motor_voltage",
        "pump_intake_pressure",
        "discharge_pressure",
        "vibration_x",
        "vibration_y",
        "intake_fluid_temp",
        "wellhead_temp",
        "surface_pressure",
    ],
    "Gas Lift": [
        "injection_rate",
        "tubing_pressure",
        "casing_pressure",
        "injection_pressure",
        "surface_pressure",
        "bottomhole_pressure",
    ],
}

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================


@dataclass
class BatchConfig:
    """Configuration for batch processing. Default: test_mode=False (process all wells)."""

    test_mode: bool = False
    test_well_limit: int = 1
    test_well_id: str = "Mustang-12-3-V"
    max_workers: int = 10

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.test_well_limit < 1:
            raise ValueError(
                f"test_well_limit must be >= 1, got {self.test_well_limit}"
            )


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
    if metric_name and (
        "stroke" in str(metric_name).lower() or "spm" in str(metric_name).lower()
    ):
        return f"{val_float:,.1f}"

    # Rule 3: Standard format
    return f"{val_float:,.0f}"
