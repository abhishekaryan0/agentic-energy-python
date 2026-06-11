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
    well_id_to_safe,
    
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

_TRAINING_EXPORTS = {
    "train_all_models",
    "train_isolation_forest",
    "train_statistical_model",
    "train_lstm_autoencoder",
}


def __getattr__(name: str):
    """Lazy import training helpers so detection batch jobs do not load train_models at startup."""
    if name in _TRAINING_EXPORTS:
        from anomaly_detection import train_models as _tm

        return getattr(_tm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    
    # Utilities
    'get_safe_value',
    'get_well_type_rules',
    'well_id_to_safe',
    
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
