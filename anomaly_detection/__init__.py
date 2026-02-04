"""
Anomaly Detection Module
========================

Multi-method ensemble anomaly detection for oil/gas well sensor data.
Combines statistical rules, Isolation Forest, and LSTM autoencoder.

Usage:
    from anomaly_detection import check_anomaly, get_detector, AnomalyDetector
    
    # Quick check
    result = check_anomaly(readings, lift_type="Rod Pump")
    
    # With historical data for LSTM
    result = check_anomaly(readings, historical_data=history, lift_type="ESP")
    
    # Get detector instance
    detector = get_detector()
    score = detector.ensemble_score(readings, lift_type="Gas Lift")
"""

from anomaly_detection.detector import (
    # Configuration
    AnomalyConfig,
    DEFAULT_CONFIG,
    
    # Main class
    AnomalyDetector,
    
    # Convenience functions
    check_anomaly,
    ensemble_score,
    get_detector,
    reset_detector,
    
    # LSTM model (for training)
    LSTMAutoencoder,
    
    # Utility
    get_safe_value,
    get_well_type_rules,
    
    # Constants (for external configuration)
    ANOMALY_THRESHOLD,
    ML_ONLY_ANOMALY_THRESHOLD,
    WEIGHTS,
    WELL_TYPES,
    WELL_TYPE_FEATURES_FALLBACK,
    CRITICAL_SENSORS,
    SEQ_LEN,
    LSTM_EPOCHS,
    CONTAMINATION_RATIO,
)

from anomaly_detection.train_models import (
    train_all_models,
    train_isolation_forest,
    train_statistical_model,
    train_lstm_autoencoder,
    load_training_data_from_snowflake,
    load_training_data_from_csv,
)

__all__ = [
    # Configuration
    'AnomalyConfig',
    'DEFAULT_CONFIG',
    
    # Main class
    'AnomalyDetector',
    
    # Convenience functions
    'check_anomaly',
    'ensemble_score', 
    'get_detector',
    'reset_detector',
    
    # Model
    'LSTMAutoencoder',
    
    # Training
    'train_all_models',
    'train_isolation_forest',
    'train_statistical_model',
    'train_lstm_autoencoder',
    'load_training_data_from_snowflake',
    'load_training_data_from_csv',
    
    # Utilities
    'get_safe_value',
    'get_well_type_rules',
    
    # Constants
    'ANOMALY_THRESHOLD',
    'ML_ONLY_ANOMALY_THRESHOLD',
    'WEIGHTS',
    'WELL_TYPES',
    'WELL_TYPE_FEATURES_FALLBACK',
    'CRITICAL_SENSORS',
    'SEQ_LEN',
    'LSTM_EPOCHS',
    'CONTAMINATION_RATIO',
]

__version__ = '1.0.0'
