"""
Recommendations Subpackage
==========================

Split from the original recommendations.py for better maintainability.

Modules:
    - analyzer: TrendAnalyzer for computing metrics from sensor data
    - rules: RuleEngine, ImpactEstimator, and ACTION_MAPPING
    - persistence: ML prediction and S3 parquet output

Usage:
    >>> from operation_recommendations.recommendations import TrendAnalyzer, RuleEngine, recommend_from_metrics
"""

# =============================================================================
# ANALYZER
# =============================================================================

from operation_recommendations.recommendations.analyzer import (
    TrendAnalyzer,
)

# =============================================================================
# RULES
# =============================================================================

from operation_recommendations.recommendations.rules import (
    RuleEngine,
    ImpactEstimator,
    ACTION_MAPPING,
    ACTION_CATEGORY_MAP,
    load_sensor_config_from_db,
)

# =============================================================================
# PERSISTENCE
# =============================================================================

from operation_recommendations.recommendations.persistence import (
    load_model,
    get_model_for_lift_type,
    predict_category,
    predict_with_model,
    _features_from_metrics,
    recommend_from_metrics,
    write_operations_recommendations_to_s3,
)

# =============================================================================
# RE-EXPORT FROM CONFIG (for backward compatibility)
# =============================================================================

from operation_recommendations.config import (
    LIFT_TYPE_SENSORS,
    COMMON_SENSORS,
    PRODUCTION_FIELDS,
    CANONICAL_FEATURES,
    FEATURES,
    CATEGORY_DESCRIPTIONS,
    get_sensors_for_lift_type,
    get_features_for_lift_type,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Analyzer
    'TrendAnalyzer',
    
    # Rules
    'RuleEngine',
    'ImpactEstimator',
    'ACTION_MAPPING',
    'ACTION_CATEGORY_MAP',
    'load_sensor_config_from_db',
    
    # Config (re-exported)
    'LIFT_TYPE_SENSORS',
    'COMMON_SENSORS',
    'PRODUCTION_FIELDS',
    'CANONICAL_FEATURES',
    'FEATURES',
    'CATEGORY_DESCRIPTIONS',
    'get_sensors_for_lift_type',
    'get_features_for_lift_type',
    
    # Model loading
    'load_model',
    'get_model_for_lift_type',
    
    # Prediction
    'predict_category',
    'predict_with_model',
    '_features_from_metrics',
    'recommend_from_metrics',
    'write_operations_recommendations_to_s3',
]
