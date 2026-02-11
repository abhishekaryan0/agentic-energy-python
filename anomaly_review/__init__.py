"""
Anomaly Review Module
=====================

Comprehensive anomaly detection and review system for oil/gas well operations.

This module provides:
- Physics-based anomaly checkers (Fluid Pound, Pump Wear, Gas Hog, etc.)
- ML-based anomaly detection (LSTM Autoencoder)
- Automated report generation with LLM
- Chart visualization generation
- Batch processing capabilities

Quick Start:
    >>> from anomaly_review import detect_anomalies, run_batch
    >>> 
    >>> # Detect anomalies for a single well
    >>> results = detect_anomalies('Well_001_RodPump')
    >>> 
    >>> # Run batch processing
    >>> run_batch()

Configuration:
    >>> from anomaly_review import AnomalyReviewConfig, BatchConfig
    >>> 
    >>> config = AnomalyReviewConfig(detection_window_days=7)
    >>> batch_config = BatchConfig(max_workers=20)
"""

__version__ = '1.0.0'

# =============================================================================
# CONFIGURATION
# =============================================================================

from anomaly_review.config import (
    # Configuration classes
    AnomalyReviewConfig,
    BatchConfig,
    DEFAULT_CONFIG,
    DEFAULT_BATCH_CONFIG,
    
    # Pricing constants
    OIL_PRICE,
    WATER_DISPOSAL_COST,
    GAS_PRICE,
    
    # Detection configuration
    DETECTION_WINDOW_DAYS,
    TRAINING_CUTOFF_DAYS,
    ROLLING_BASELINE_DAYS,
    MODEL_CACHE_DAYS,
    MODEL_DIR,
    
    # Sensor metadata
    SENSOR_DISPLAY_NAMES,
    SEVERITY_THRESHOLDS,
    
    # Suppression rules
    SUPPRESSION_RULES,
    ANOMALY_PRIORITY,
    
    # Utilities
    format_number,
)

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

from anomaly_review.data import (
    # Database connections
    get_snowflake_conn,
    get_db_url,
    get_openai_client,
    
    # Data fetching
    fetch_well_data,
    fetch_all_active_wells,
    get_recent_anomaly_history,
    
    # Data processing
    calculate_rolling_baseline,
)

# =============================================================================
# CHECKER FUNCTIONS
# =============================================================================

from anomaly_review.checkers import (
    # Severity
    determine_severity,
    diagnose_mechanics,
    
    # Physics checkers
    check_fluid_pound,
    check_pump_wear,
    check_gas_injection_inefficiency,
    check_energy_efficiency_drift,
    check_efficiency_degradation,
    check_pressure_instability,
    check_model_divergence,
    check_monthly_production_decline,
    check_event_frequency,
    check_energy_production_divergence,
    check_sensor_state_mismatch,
    
    # ML checker
    check_ml_anomaly,
)

# =============================================================================
# ML MODELS
# =============================================================================

from anomaly_review.models import (
    # Model classes
    LSTMRegressor,
    LSTMAutoencoder,
    
    # Training
    train_lstm_model,
    train_autoencoder_model,
    get_or_train_model,
    
    # Persistence
    save_models_to_disk,
    load_models_from_disk,
    get_model_dir,
    
    # Prediction
    predict_with_lstm,
    detect_autoencoder_anomalies,
    
    # Utilities
    prepare_sequences,
)

# =============================================================================
# CHART GENERATION
# =============================================================================

from anomaly_review.charts import (
    generate_chart_config,
    generate_mechanical_failure_chart,
    generate_efficiency_chart,
    generate_baseline_shift_chart,
    generate_pressure_instability_chart,
    generate_monthly_decline_chart,
    generate_pattern_shift_chart,
    generate_ghost_production_chart,
    generate_default_enhanced_chart,
    create_fallback_chart,
)

# =============================================================================
# LLM GENERATION
# =============================================================================

from anomaly_review.llm import (
    generate_llm_content,
)

# =============================================================================
# CORE SERVICE
# =============================================================================

from anomaly_review.service import (
    detect_anomalies,
    select_top_anomalies_per_well,
    save_reviews,
)

# =============================================================================
# BATCH RUNNER
# =============================================================================

# from anomaly_review.run import (
#     run_batch,
#     process_single_well,
#     get_all_active_wells,
# )

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Configuration
    'AnomalyReviewConfig',
    'BatchConfig',
    'DEFAULT_CONFIG',
    'DEFAULT_BATCH_CONFIG',
    'OIL_PRICE',
    'WATER_DISPOSAL_COST',
    'GAS_PRICE',
    'DETECTION_WINDOW_DAYS',
    'TRAINING_CUTOFF_DAYS',
    'ROLLING_BASELINE_DAYS',
    'MODEL_CACHE_DAYS',
    'MODEL_DIR',
    'SENSOR_DISPLAY_NAMES',
    'SEVERITY_THRESHOLDS',
    'SUPPRESSION_RULES',
    'ANOMALY_PRIORITY',
    'format_number',
    
    # Data functions
    'get_snowflake_conn',
    'get_db_url',
    'get_openai_client',
    'fetch_well_data',
    'fetch_all_active_wells',
    'get_recent_anomaly_history',
    'calculate_rolling_baseline',
    
    # Checkers
    'determine_severity',
    'diagnose_mechanics',
    'check_fluid_pound',
    'check_pump_wear',
    'check_gas_injection_inefficiency',
    'check_energy_efficiency_drift',
    'check_efficiency_degradation',
    'check_pressure_instability',
    'check_model_divergence',
    'check_monthly_production_decline',
    'check_event_frequency',
    'check_ml_anomaly',
    'check_energy_production_divergence',
    'check_sensor_state_mismatch',
    
    # ML models
    'LSTMRegressor',
    'LSTMAutoencoder',
    'train_lstm_model',
    'train_autoencoder_model',
    'get_or_train_model',
    'save_models_to_disk',
    'load_models_from_disk',
    'get_model_dir',
    'predict_with_lstm',
    'detect_autoencoder_anomalies',
    'prepare_sequences',
    
    # Charts
    'generate_chart_config',
    'generate_mechanical_failure_chart',
    'generate_efficiency_chart',
    'generate_baseline_shift_chart',
    'generate_pressure_instability_chart',
    'generate_monthly_decline_chart',
    'generate_pattern_shift_chart',
    'generate_ghost_production_chart',
    'generate_default_enhanced_chart',
    'create_fallback_chart',
    
    # LLM
    'generate_llm_content',
    
    # Core service
    'detect_anomalies',
    'select_top_anomalies_per_well',
    'save_reviews',
    
    # Batch runner
    # 'run_batch',
    # 'process_single_well',
    # 'get_all_active_wells',
]
