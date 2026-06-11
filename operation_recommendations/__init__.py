"""
Operations Module for Oil/Gas Well Recommendations
===================================================

This module provides comprehensive operational recommendations for oil and gas wells,
combining rule-based detection, ML predictions, and OpenAI-enhanced suggestions.

Features:
    - TrendAnalyzer: Compute trends and rolling stats from sensor data
    - RuleEngine: Apply deterministic rules to metrics based on sensor thresholds
    - ImpactEstimator: Estimate potential uplift for recommendations
    - ML-based prediction and training functions
    - OpenAI-enhanced suggestion generation

Quick Start:
    >>> from operation_recommendations import TrendAnalyzer, RuleEngine, predict_category
    >>> 
    >>> # Analyze trends
    >>> analyzer = TrendAnalyzer(sensor_df, lift_type='ESP')
    >>> metrics = analyzer.compute_metrics()
    >>> 
    >>> # Get recommendations from metrics (S3 parquet)
    >>> from operation_recommendations.recommendations.persistence import recommend_from_metrics
    >>> recs = recommend_from_metrics(well_id, lift_type, metrics, confidence_threshold=0.6)

Configuration:
    >>> from operation_recommendations import RecommendationConfig, TrainingConfig
    >>> 
    >>> config = RecommendationConfig(confidence_threshold=0.8)
    >>> training_config = TrainingConfig(n_estimators=200)

For detailed documentation, see individual module docstrings.
"""

# ============================================================================
# VERSION
# ============================================================================

__version__ = '1.0.0'

# ============================================================================
# CONFIGURATION
# ============================================================================

from operation_recommendations.config import (
    # Configuration classes
    RecommendationConfig,
    TrainingConfig,
    DEFAULT_RECOMMENDATION_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    
    # Lift type configuration
    LIFT_TYPE_SENSORS,
    COMMON_SENSORS,
    PRODUCTION_FIELDS,
    CANONICAL_FEATURES,
    FEATURES,
    CATEGORY_DESCRIPTIONS,
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    
    # Configuration functions
    get_sensors_for_lift_type,
    get_features_for_lift_type,
)

# ============================================================================
# UTILITIES
# ============================================================================

from operation_recommendations.utils import (
    # Path utilities
    get_project_root,
    get_models_dir,
    get_model_path,
    retry_with_backoff,
    retry_db_operation,
    
    # Type conversion
    to_native_type,
    safe_float,
    truncate_string,
    features_from_metrics,
)

# ============================================================================
# CORE CLASSES (from recommendations subpackage)
# ============================================================================

from operation_recommendations.recommendations.analyzer import TrendAnalyzer

from operation_recommendations.recommendations.rules import (
    RuleEngine,
    ImpactEstimator,
    ACTION_MAPPING,
    ACTION_CATEGORY_MAP,
    load_sensor_config_from_db,
)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

from operation_recommendations.recommendations.persistence import (
    predict_category,
    load_model,
    get_model_for_lift_type,
    predict_with_model,
    recommend_from_metrics,
    write_operations_recommendations_to_s3,
    derive_production_impact_from_metrics,
    _features_from_metrics,
)

from operation_recommendations.train_models import (
    train_operations_from_parquet,
    train_with_synthetic_labels,
    build_metrics_from_parquet,
)

# ============================================================================
# PUBLIC API (model + I/O use S3 only)
# ============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Configuration classes
    'RecommendationConfig',
    'TrainingConfig',
    'DEFAULT_RECOMMENDATION_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    
    # Configuration constants
    'LIFT_TYPE_SENSORS',
    'COMMON_SENSORS',
    'PRODUCTION_FIELDS',
    'CANONICAL_FEATURES',
    'FEATURES',
    'CATEGORY_DESCRIPTIONS',
    'VALID_CATEGORIES',
    'VALID_PRIORITIES',
    'ACTION_MAPPING',
    'ACTION_CATEGORY_MAP',
    
    # Configuration functions
    'get_sensors_for_lift_type',
    'get_features_for_lift_type',
    
    # Utility functions
    'get_project_root',
    'get_models_dir',
    'get_model_path',
    'retry_with_backoff',
    'retry_db_operation',
    'to_native_type',
    'safe_float',
    'truncate_string',
    'features_from_metrics',
    
    # Core classes
    'TrendAnalyzer',
    'RuleEngine',
    'ImpactEstimator',
    'load_sensor_config_from_db',
    '_features_from_metrics',
    
    # Prediction functions
    'predict_category',
    'load_model',
    'get_model_for_lift_type',
    'predict_with_model',
    'recommend_from_metrics',
    'write_operations_recommendations_to_s3',
    'derive_production_impact_from_metrics',
    
    # Training (S3/parquet only)
    'train_operations_from_parquet',
    'train_with_synthetic_labels',
    'build_metrics_from_parquet',
]
