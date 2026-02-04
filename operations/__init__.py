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
    >>> from operations import TrendAnalyzer, RuleEngine, predict_category
    >>> 
    >>> # Analyze trends
    >>> analyzer = TrendAnalyzer(sensor_df, lift_type='ESP')
    >>> metrics = analyzer.compute_metrics()
    >>> 
    >>> # Get recommendations
    >>> engine = RuleEngine(snowflake_conn)
    >>> recommendations = engine.apply(well_id, metrics)

Configuration:
    >>> from operations import RecommendationConfig, TrainingConfig
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

from operations.config import (
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

from operations.utils import (
    # Path utilities
    get_project_root,
    get_models_dir,
    get_model_path,
    
    # Database utilities
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

from operations.recommendations.analyzer import TrendAnalyzer

from operations.recommendations.rules import (
    RuleEngine,
    ImpactEstimator,
    ACTION_MAPPING,
    ACTION_CATEGORY_MAP,
    load_sensor_config_from_db,
)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

from operations.recommendations.persistence import (
    predict_category,
    predict_all_wells,
    load_model,
    get_model_for_lift_type,
    predict_with_model,
    generate_recommendations_for_well,
    detect_and_persist,
    save_recommendation_to_snowflake,
    _features_from_metrics,
)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

from operations.recommendations.persistence import (
    train_with_synthetic_labels,
    train_from_snowflake,
    train_on_two_wells,
)

from operations.train_models import (
    train_category_model,
)

# ============================================================================
# SUGGESTIONS (OpenAI Integration)
# ============================================================================

from operations.suggestions import (
    # Table management
    create_operation_suggestion_table,
    
    # OpenAI generation
    generate_operation_suggestion_with_openai,
    infer_action_category_priority,
    refine_suggestion_with_openai,
    
    # CRUD operations
    save_operation_suggestion,
    get_operation_suggestions,
    update_operation_suggestion_status,
    get_operation_suggestion_detail,
    
    # Pipeline streaming
    stream_run_operational_pipeline,
)

# ============================================================================
# PUBLIC API
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
    'predict_all_wells',
    'load_model',
    'get_model_for_lift_type',
    'predict_with_model',
    'generate_recommendations_for_well',
    'detect_and_persist',
    'save_recommendation_to_snowflake',
    
    # Training functions
    'train_with_synthetic_labels',
    'train_from_snowflake',
    'train_on_two_wells',
    'train_category_model',
    
    # Suggestions functions
    'create_operation_suggestion_table',
    'generate_operation_suggestion_with_openai',
    'infer_action_category_priority',
    'refine_suggestion_with_openai',
    'save_operation_suggestion',
    'get_operation_suggestions',
    'update_operation_suggestion_status',
    'get_operation_suggestion_detail',
    'stream_run_operational_pipeline',
]
