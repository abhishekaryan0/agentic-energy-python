"""
Recommendations Subpackage
==========================

Split from the original recommendations.py for better maintainability.

Modules:
    - analyzer: TrendAnalyzer for computing metrics from sensor data
    - rules: RuleEngine, ImpactEstimator, and ACTION_MAPPING
    - persistence: Database persistence and ML prediction functions

Usage:
    >>> from operations.recommendations import TrendAnalyzer, RuleEngine
    >>> from operations.recommendations import save_recommendation_to_snowflake
"""

# =============================================================================
# ANALYZER
# =============================================================================

from operations.recommendations.analyzer import (
    TrendAnalyzer,
)

# =============================================================================
# RULES
# =============================================================================

from operations.recommendations.rules import (
    RuleEngine,
    ImpactEstimator,
    ACTION_MAPPING,
    ACTION_CATEGORY_MAP,
    load_sensor_config_from_db,
)

# =============================================================================
# PERSISTENCE
# =============================================================================

from operations.recommendations.persistence import (
    # Model loading
    load_model,
    get_model_for_lift_type,
    
    # Prediction
    predict_category,
    predict_with_model,
    predict_all_wells,
    
    # Features
    _features_from_metrics,
    
    # Persistence
    save_recommendation_to_snowflake,
    generate_recommendations_for_well,
    recommend_from_metrics,
    detect_and_persist,
    
    # Training (delegated)
    train_with_synthetic_labels,
    train_from_snowflake,
    train_on_two_wells,
)

# =============================================================================
# RE-EXPORT FROM CONFIG (for backward compatibility)
# =============================================================================

from operations.config import (
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
    'predict_all_wells',
    '_features_from_metrics',
    
    # Persistence
    'save_recommendation_to_snowflake',
    'generate_recommendations_for_well',
    'recommend_from_metrics',
    'detect_and_persist',
    
    # Training
    'train_with_synthetic_labels',
    'train_from_snowflake',
    'train_on_two_wells',
]
