"""
Configuration for the operations module.

This module contains configuration dataclasses and constants used throughout
the operations package for recommendations, predictions, and training.
Uses S3 only. ORGANIZATION_ID from env for model path.

Production-related environment variables (non-exhaustive):

- **S3_BUCKET**, **ORGANIZATION_ID**, **AWS_REGION** — parquet I/O and model tarball resolution.
- **OPERATIONS_MODEL_TARBALL_S3_URI** — SageMaker ``operations-train-.../output/model.tar.gz`` (auto-detected if unset).
- **FINANCIAL_MODEL_TARBALL_S3_URI** — optional financial model tarball (auto-detected if unset).
- **OPERATIONS_MODELS_DIR** — optional local joblib directory (dev fallback when S3 URI not set).
- **DATABASE_URL** or **DATABASE_URL_SECRET_ARN** — PostgreSQL for per-org ``org_settings`` (oil price type).
- **EIA_API_KEY** — U.S. EIA Open Data API key when ``org_settings.oil_price_type`` is Brent; if unset,
  **OPERATIONS_OIL_PRICE_USD** (default 70) is used for revenue-side pricing.
- **EIA_CACHE_TTL_SEC**, **EIA_HTTP_TIMEOUT_SEC**, **EIA_HTTP_RETRIES** — tune EIA HTTP client.
- **FINANCIAL_MODEL_PATH** — optional file or directory for ``financial_implementation_cost.joblib``.
- **IMPLEMENTATION_LUMP_DAYS**, **IMPLEMENTATION_AMORTIZATION_DAYS** — convert financial model daily rate to one-time lump and amortize for net daily benefit in ``recommendations.persistence``.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def get_organization_id() -> Optional[str]:
    """Organization UUID from env."""
    return os.getenv("ORGANIZATION_ID")


def get_operations_s3_bucket() -> str:
    """S3 bucket for operations input/output. From S3_BUCKET env."""
    return os.getenv("S3_BUCKET", "")


def get_operations_output_prefix() -> str:
    """S3 prefix for operations detection output: {ORG}/detections/operations or override."""
    override = os.getenv("OPERATIONS_OUTPUT_S3_PREFIX", "").strip()
    if override:
        return override.rstrip("/")
    org = get_organization_id()
    if org:
        return f"{org}/detections/operations"
    return "detections/operations"


@dataclass
class RecommendationConfig:
    """
    Configuration for operational recommendations.
    
    Attributes:
        confidence_threshold: Minimum confidence to persist recommendations (0-1)
        dedup_days: Days to check for duplicate recommendations
        adaptive_k: Z-score threshold for trend-based detection
        priority_thresholds: Dict mapping deviation % to priority levels
        models_dir: Directory containing trained models
    """
    # Prediction thresholds
    confidence_threshold: float = 0.6
    dedup_days: int = 7
    
    # Trend detection
    adaptive_k: float = 2.0
    
    # Priority calculation thresholds (deviation %)
    priority_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 25.0,
        'medium': 10.0,
        'low': 0.0
    })
    
    # Z-score thresholds for trend detection
    zscore_high: float = 3.0
    zscore_medium: float = 2.0
    
    # Model paths
    models_dir: str = 'models'
    default_model_name: str = 'decision_tree_alldata.joblib'
    category_model_name: str = 'category_classifier.joblib'
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}")
        if self.dedup_days < 0:
            raise ValueError(f"dedup_days must be >= 0, got {self.dedup_days}")
        if self.adaptive_k <= 0:
            raise ValueError(f"adaptive_k must be > 0, got {self.adaptive_k}")


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        test_size: Fraction of data for testing (0-1)
        random_state: Random seed for reproducibility
        n_estimators: Number of trees for RandomForest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in a leaf
        use_calibration: Whether to use CalibratedClassifierCV
        calibration_cv: Cross-validation folds for calibration
    """
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    
    # RandomForest parameters
    n_estimators: int = 200
    max_depth: int = 8
    min_samples_split: int = 3
    min_samples_leaf: int = 2
    
    # Calibration
    use_calibration: bool = True
    calibration_cv: int = 3
    
    # Feature engineering
    zscore_threshold: float = 1.2  # For synthetic label generation
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")


# Default configuration instances
DEFAULT_RECOMMENDATION_CONFIG = RecommendationConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()


# ============================================================================
# LIFT TYPE CONFIGURATION
# ============================================================================

# Lift type to sensor field mapping (matches well_sensor_readings table)
LIFT_TYPE_SENSORS: Dict[str, List[str]] = {
    'Rod Pump': [
        'strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage',
        'surface_stroke_length', 'downhole_gross_stroke', 'downhole_net_stroke',
        'runtime', 'cycles', 'structural_load', 'inferred_production',
        'pump_intake_pressure', 'dry_rod_weight', 'buoyant_rod_weight',
        'pump_friction', 'pump_diameter', 'barrel_length', 'pump_depth',
        'idle_time_setpoint', 'surface_pressure', 'tubing_pressure', 'casing_pressure', 'wellhead_temp'
    ],
    'ESP': [
        'motor_temp', 'motor_current', 'discharge_pressure', 'motor_voltage',
        'intake_fluid_temp', 'vibration_x', 'vibration_y', 'discharge_temp',
        'downhole_flow_rate', 'drive_frequency', 'tubing_pressure', 'casing_pressure',
        'drive_input_voltage', 'output_voltage', 'input_current', 'motor_speed',
        'vsd_temp', 'total_harmonic_distortion', 'motor_load', 'set_frequency',
        'acceleration_ramp_time', 'surface_pressure', 'wellhead_temp', 'pump_intake_pressure'
    ],
    'Gas Lift': [
        'injection_rate', 'injection_temperature', 'bottomhole_pressure',
        'injection_pressure', 'cycle_time', 'bottomhole_temp',
        'plunger_arrival_time', 'plunger_velocity', 'arrival_count', 'missed_arrivals',
        'shutin_time', 'afterflow_time', 'flow_time', 'plunger_drop_time',
        'min_shutin_pressure', 'max_shutin_pressure', 'open_diff_pressure',
        'well_open', 'well_close', 'velocity_limit',
        'surface_pressure', 'tubing_pressure', 'casing_pressure', 'wellhead_temp'
    ]
}

# Common sensors across all lift types
COMMON_SENSORS: List[str] = ['surface_pressure', 'tubing_pressure', 'casing_pressure', 'wellhead_temp']

# Production fields from well_daily_production table
PRODUCTION_FIELDS: List[str] = ['oil_volume', 'gas_volume', 'water_volume', 'boe']

# Legacy canonical features for backward compatibility
CANONICAL_FEATURES: List[str] = [
    'oil_volume_slope', 'gas_volume_slope', 'water_volume_slope',
    'oil_volume_roll_mean', 'gas_volume_roll_mean', 'water_volume_roll_mean',
    'surface_pressure_slope', 'surface_pressure_roll_mean',
    'tubing_pressure_slope', 'tubing_pressure_roll_mean',
    'casing_pressure_slope', 'casing_pressure_roll_mean',
    'missing_rate'
]

# Alias for backward compatibility
FEATURES = CANONICAL_FEATURES


# ============================================================================
# CATEGORY CONFIGURATION
# ============================================================================

# Human-readable category descriptions used for UI/context
CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    'Production': (
        "Recommendations that directly aim to increase oil or gas production. "
        "May increase operating cost or equipment wear; valid if net economic benefit is positive."
    ),
    'Optimization': (
        "Recommendations focused on cost/efficiency optimization (reduce chemical usage, "
        "operating costs) without necessarily increasing production significantly."
    ),
    'Maintenance': (
        "Maintenance / Failure Prevention: extend equipment life, prevent failures, "
        "reduce downtime. These are recommendations, not execution actions."
    ),
}

# Valid categories
VALID_CATEGORIES = {'Production', 'Optimization', 'Maintenance'}

# Valid priorities
VALID_PRIORITIES = {'HIGH', 'MEDIUM', 'LOW'}


def get_sensors_for_lift_type(lift_type: str) -> List[str]:
    """
    Get the list of sensor field names for a specific lift type (tenant-aware).
    """
    from operation_recommendations.tenant_config import get_tenant_sensors_for_lift_type

    return get_tenant_sensors_for_lift_type(lift_type)


def get_features_for_lift_type(lift_type: str, include_production: bool = True) -> List[str]:
    """
    Get the list of feature names (slopes + roll_means) for a specific lift type.
    
    Args:
        lift_type: Lift type name
        include_production: Whether to include production features
        
    Returns:
        List of feature names
    """
    sensors = get_sensors_for_lift_type(lift_type)
    features = []
    
    # Add sensor-based features
    for sensor in sensors:
        features.append(f'{sensor}_slope')
        features.append(f'{sensor}_roll_mean')
    
    # Add production features
    if include_production:
        for prod_field in PRODUCTION_FIELDS:
            features.append(f'{prod_field}_slope')
            features.append(f'{prod_field}_roll_mean')
    
    # Add common derived features
    features.append('missing_rate')
    
    # Add correlations for specific lift types
    if lift_type == 'ESP':
        features.extend(['motor_current_motor_temp_corr', 'motor_current_vibration_corr'])
    elif lift_type == 'Rod Pump':
        features.extend(['spm_load_corr', 'fillage_production_corr'])
    elif lift_type == 'Gas Lift':
        features.extend(['injection_production_corr', 'pressure_rate_corr'])
    
    return features
