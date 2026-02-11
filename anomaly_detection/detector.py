"""
Unified Anomaly Detection Module
================================

Integrates statistical rules, Isolation Forest, and LSTM for comprehensive anomaly detection.
Loads sensor configurations dynamically from Snowflake instead of hardcoding.

Features:
    - Rule-based detection using sensor thresholds from database
    - Isolation Forest for statistical outlier detection
    - LSTM Autoencoder for time-series pattern anomalies
    - Weighted ensemble scoring combining all methods

Usage:
    from anomaly_detection import check_anomaly, get_detector
    
    result = check_anomaly(readings, lift_type="Rod Pump")
    print(result['is_anomaly'], result['anomaly_score'])
"""

import pandas as pd
import numpy as np
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib

# ============================================================================
# OPTIONAL DEPENDENCY HANDLING
# ============================================================================

try:
    from config.sensor_config import get_sensor_ranges, get_anomaly_rules, get_features_for_lift_type, get_lift_types
    SENSOR_CONFIG_AVAILABLE = True
except ImportError:
    SENSOR_CONFIG_AVAILABLE = False
    get_features_for_lift_type = None
    get_lift_types = None
    get_sensor_ranges = None
    get_anomaly_rules = None
    logging.warning("sensor_config module not available; using fallback rules")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    StandardScaler = None
    logging.warning("scikit-learn not available; Isolation Forest disabled")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available; LSTM disabled")
    torch = None
    nn = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AnomalyConfig:
    """
    Configuration for anomaly detection thresholds and weights.
    
    Attributes:
        anomaly_threshold: Score threshold to classify as anomaly (0-1)
        ml_only_threshold: Higher threshold when only ML models available
        weights: Dict of weights for each detection method
        seq_len: Sequence length for LSTM input
        lstm_epochs: Training epochs for LSTM
        lstm_lr: Learning rate for LSTM training
        mse_threshold: MSE threshold for LSTM anomaly scoring
        iso_contamination: Expected contamination ratio for Isolation Forest
        z_score_threshold: Z-score threshold for statistical detection
        rule_base_contrib: Base contribution for rule violations
        rule_extra_cap: Maximum extra contribution from deviation
        rule_dev_scale: Scaling factor for deviation contribution
    """
    # Detection thresholds
    anomaly_threshold: float = 0.75
    ml_only_threshold: float = 0.85
    
    # Method weights (must sum to ~1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'rules': 0.4,
        'iso': 0.3,
        'lstm': 0.3
    })
    
    # LSTM configuration
    seq_len: int = 10
    lstm_epochs: int = 20
    lstm_lr: float = 0.001
    mse_threshold: float = 0.05
    
    # Isolation Forest configuration
    iso_contamination: float = 0.05
    contamination_ratio: float = 0.1
    
    # Statistical model configuration
    z_score_threshold: float = 3.0
    
    # Rule scoring configuration
    rule_base_contrib: float = 0.35
    rule_extra_cap: float = 0.7
    rule_dev_scale: float = 0.9
    
    # Models directory
    models_dir: str = 'models'
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 < self.anomaly_threshold <= 1:
            raise ValueError(f"anomaly_threshold must be in (0, 1], got {self.anomaly_threshold}")
        if not 0 < self.iso_contamination < 0.5:
            raise ValueError(f"iso_contamination must be in (0, 0.5), got {self.iso_contamination}")
        weight_sum = sum(self.weights.values())
        if not 0.9 <= weight_sum <= 1.1:
            raise ValueError(f"weights must sum to ~1.0, got {weight_sum}")


# Default configuration instance
DEFAULT_CONFIG = AnomalyConfig()

# Export constants for backward compatibility
ANOMALY_THRESHOLD = DEFAULT_CONFIG.anomaly_threshold
ML_ONLY_ANOMALY_THRESHOLD = DEFAULT_CONFIG.ml_only_threshold
WEIGHTS = DEFAULT_CONFIG.weights
SEQ_LEN = DEFAULT_CONFIG.seq_len
LSTM_EPOCHS = DEFAULT_CONFIG.lstm_epochs
LSTM_LR = DEFAULT_CONFIG.lstm_lr
MSE_THRESHOLD = DEFAULT_CONFIG.mse_threshold
ISO_CONTAMINATION = DEFAULT_CONFIG.iso_contamination
CONTAMINATION_RATIO = DEFAULT_CONFIG.contamination_ratio
Z_SCORE_THRESHOLD = DEFAULT_CONFIG.z_score_threshold
RULE_BASE_CONTRIB = DEFAULT_CONFIG.rule_base_contrib
RULE_EXTRA_CAP = DEFAULT_CONFIG.rule_extra_cap
RULE_DEV_SCALE = DEFAULT_CONFIG.rule_dev_scale

# Well type constants
ROLLING_WINDOWS = [1, 3, 6, 12]
WELL_TYPES = ['Rod Pump', 'ESP', 'Gas Lift']

# Fallback features when sensor_config unavailable (matches sensor_definitions)
WELL_TYPE_FEATURES_FALLBACK: Dict[str, List[str]] = {
    'Rod Pump': ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure'],
    'ESP': ['motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage'],
    'Gas Lift': ['injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time'],
}

# Critical sensors that require immediate attention when out of range
CRITICAL_SENSORS: Dict[str, List[str]] = {
    'Rod Pump': ['tubing_pressure', 'strokes_per_minute'],
    'ESP': ['motor_current', 'motor_temp'],
    'Gas Lift': ['injection_rate', 'injection_pressure'],
}

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback for when PyTorch is not available
class FakeLSTMModule:
    """Dummy module when torch is not available."""
    pass

if nn is None:
    nn = FakeLSTMModule()
    nn.Module = object

if TORCH_AVAILABLE:
    class LSTMAutoencoder(nn.Module):
        """LSTM-based autoencoder for sequence anomaly detection. Uses PyTorch LSTM."""
        def __init__(self, input_dim, hidden_dim=64, num_layers=1):
            super(LSTMAutoencoder, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            # hidden is (1, batch, hidden_dim), we want (batch, seq_len, hidden_dim)
            decoder_input = hidden.permute(1, 0, 2).repeat(1, x.shape[1], 1)
            decoded, _ = self.decoder(decoder_input)
            reconstructed = self.linear(decoded)
            return reconstructed
else:
    class LSTMAutoencoder:
        """Dummy LSTM when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            pass
        
        def forward(self, x):
            return x

def get_safe_value(d: Dict, k: str, default: float = 0.0) -> float:
    """Safely get a float value from dict, handling None and conversion errors."""
    val = d.get(k)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# Dynamic rules loader function
def get_well_type_rules(lift_type: str) -> Dict:
    """Get sensor rules for a well type from dynamic config (Snowflake)."""
    if not SENSOR_CONFIG_AVAILABLE:
        raise ValueError("Dynamic config not available. Ensure Snowflake is configured and tables are created.")
    
    try:
        ranges = get_sensor_ranges(lift_type)
        rules = {}
        for field_name, range_info in ranges.items():
            rules[field_name] = {
                "min": range_info.get('min'),
                "max": range_info.get('max')
            }
        return rules
    except Exception as e:
        logger.error(f"Error loading dynamic rules for {lift_type}: {e}")
        raise

class AnomalyDetector:
    """
    Unified anomaly detection combining rules, Isolation Forest, and LSTM.
    
    This class provides a multi-method ensemble approach to detecting anomalies
    in oil/gas well sensor data. It combines:
    
    1. Rule-based detection: Uses sensor thresholds from database
    2. Isolation Forest: Statistical outlier detection
    3. LSTM Autoencoder: Time-series pattern anomalies
    
    Attributes:
        config: AnomalyConfig instance with detection parameters
        trained_models: Dict of loaded ML models
        
    Example:
        detector = AnomalyDetector()
        result = detector.check_anomaly(
            readings={'motor_temp': 250, 'motor_current': 45},
            lift_type='ESP'
        )
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """
        Initialize detector with rules from dynamic Snowflake config.
        
        Args:
            config: Optional AnomalyConfig instance. Uses DEFAULT_CONFIG if not provided.
            
        Raises:
            RuntimeError: If sensor_config module is not available
        """
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        
        if not SENSOR_CONFIG_AVAILABLE:
            raise RuntimeError(
                "Dynamic config (Snowflake) is required. "
                "Run: python setup_production_db.py\n"
                "Or ensure config.sensor_config module is available."
            )
        
        self.trained_models: Dict[str, Any] = self._load_trained_models()
        
        logger.info(f"✓ Anomaly detector initialized (using dynamic Snowflake config)")
        logger.info(f"  - Threshold: {self.config.anomaly_threshold}")
        logger.info(f"  - Weights: {self.config.weights}")
        if self.trained_models:
            logger.info(f"✓ Loaded trained models: {', '.join(self.trained_models.keys())}")
    
    def _load_trained_models(self) -> Dict[str, Any]:
        """Load pre-trained models from models/ directory."""
        trained_models: Dict[str, Any] = {}
        models_dir = self.config.models_dir
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found at {models_dir}. Run: python -m anomaly_detection.train_models")
            return trained_models
        
        # Load models for each well type
        well_types = ['rod_pump', 'esp', 'gas_lift']
        for well_type in well_types:
            iso_file = os.path.join(models_dir, f'isolation_forest_{well_type}.joblib')
            stat_file = os.path.join(models_dir, f'statistical_model_{well_type}.joblib')
            lstm_file = os.path.join(models_dir, f'lstm_autoencoder_{well_type}.pt') if TORCH_AVAILABLE else None
            
            if os.path.exists(iso_file):
                try:
                    model_data = joblib.load(iso_file)
                    if isinstance(model_data, dict) and 'model' in model_data and 'scaler' in model_data:
                        trained_models[f'iso_{well_type}'] = model_data
                        logger.info(f"✓ Loaded isolation forest model for {well_type}")
                    else:
                        logger.warning(f"Invalid isolation forest model structure for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load {iso_file}: {e}")
            
            if os.path.exists(stat_file):
                try:
                    model_data = joblib.load(stat_file)
                    if isinstance(model_data, dict) and 'mean' in model_data and 'std' in model_data:
                        trained_models[f'stat_{well_type}'] = model_data
                        logger.info(f"✓ Loaded statistical model for {well_type}")
                    else:
                        logger.warning(f"Invalid statistical model structure for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load {stat_file}: {e}")
            
            lstm_meta_file = os.path.join(models_dir, f'lstm_metadata_{well_type}.joblib')
            if lstm_file and os.path.exists(lstm_file) and os.path.exists(lstm_meta_file):
                try:
                    meta = joblib.load(lstm_meta_file)
                    input_dim = meta['input_dim']
                    features = meta.get('features', [])
                    seq_len_saved = meta.get('seq_len', SEQ_LEN)  # use training-time seq_len
                    scaler = joblib.load(os.path.join(models_dir, f'lstm_scaler_{well_type}.joblib'))
                    model = LSTMAutoencoder(input_dim=input_dim)
                    model.load_state_dict(torch.load(lstm_file, map_location='cpu'))
                    model.eval()
                    trained_models[f'lstm_{well_type}'] = {
                        'model': model, 'scaler': scaler, 'seq_len': seq_len_saved,
                        'features': features
                    }
                    logger.info(f"✓ Loaded LSTM autoencoder for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load LSTM for {well_type}: {e}")
        
        return trained_models
    
    def _compute_rule_based_score(self, readings: Dict, active_rules: Dict, lift_type: str) -> Tuple[List, float]:
        """Compute rule-based anomaly score and violations."""
        violations = []
        rule_anomaly_score = 0.0
        
        for field, value in readings.items():
            if field not in active_rules or value is None:
                continue
            
            rule = active_rules[field]
            if rule is None or not isinstance(rule, dict):
                continue
            
            rule_min = rule.get("min")
            rule_max = rule.get("max")
            
            if rule_min is None or rule_max is None:
                continue
            
            if value < rule_min or value > rule_max:
                range_width = rule_max - rule_min if rule_max != rule_min else 1e-8
                deviation_pct = abs(value - (rule_min if value < rule_min else rule_max)) / range_width * 100
                
                sensor_unit = get_sensor_ranges(lift_type).get(field, {}).get('unit', '')
                
                min_str = self._format_value(rule_min)
                max_str = self._format_value(rule_max)
                val_str = self._format_value(value)
                unit_str = f" {sensor_unit}" if sensor_unit else ""
                violation_text = f"Out of range. Expected {min_str}{unit_str} - {max_str}{unit_str}, got {val_str}{unit_str} ({deviation_pct:.1f}% deviation)"

                logger.info(f"[AnomalyDetector] lift_type={lift_type} field={field} min={rule_min} max={rule_max} unit={sensor_unit} value={value}")

                violations.append({
                    "field": field,
                    "value": value,
                    "min": rule_min,
                    "max": rule_max,
                    "unit": sensor_unit,
                    "deviation_pct": round(deviation_pct, 2),
                    "violation": violation_text
                })
                
                # Calculate contribution using configured scaling factors
                dev_factor = deviation_pct / 100.0
                contrib = self.config.rule_base_contrib + min(
                    self.config.rule_extra_cap,
                    dev_factor * self.config.rule_dev_scale
                )
                rule_anomaly_score += contrib
        
        rule_anomaly_score = min(1.0, rule_anomaly_score)
        return violations, rule_anomaly_score
    
    def _format_value(self, val) -> str:
        """Safely format a value as string."""
        try:
            return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        except Exception:
            return str(val)
    
    def _combine_scores(self, rule_score: float, ml_score: float, lstm_score: float = 0.0) -> float:
        """
        Combine rule, ML (iso+stat), and LSTM scores using configured weights.
        
        Args:
            rule_score: Score from rule-based detection (0-1)
            ml_score: Score from Isolation Forest + statistical model (0-1)
            lstm_score: Score from LSTM autoencoder (0-1)
            
        Returns:
            Combined weighted score normalized to 0-1
        """
        weights = self.config.weights
        rules_w = weights.get('rules', 0.4)
        ml_w = weights.get('iso', 0.3) + weights.get('stat', 0.0)  # Treat as combined ML weight
        lstm_w = weights.get('lstm', 0.3)
        total_w = rules_w + ml_w + lstm_w
        
        if total_w == 0:
            return 0.0
            
        normalized_score = (rules_w * rule_score + ml_w * ml_score + lstm_w * lstm_score) / total_w
        return min(1.0, normalized_score)
    
    def check_anomaly(self, readings: Dict, historical_data: Optional[List[Dict]] = None, well_id: str = None, lift_type: str = None) -> Dict:
        """
        Check for anomalies using statistical rules, trained ML models, and optional LSTM on history.
        
        Args:
            readings: Dict of current sensor values {field: value}
            historical_data: Optional list of previous readings (newest last, excluding current)
            well_id: Well identifier (optional, for logging)
            lift_type: Well type (Rod Pump/ESP/Gas Lift) for context-aware rules
        
        Returns:
            Dict with is_anomaly, anomaly_score, violations, summary, detection_method
        """
        if lift_type is None or (isinstance(lift_type, str) and lift_type.strip() == ''):
            raise ValueError(f"lift_type is required for anomaly detection. Received: {repr(lift_type)}")
        
        violations = []
        rule_anomaly_score = 0.0
        ml_anomaly_score = 0.0
        lstm_anomaly_score = 0.0
        
        active_rules = get_well_type_rules(lift_type)
        
        # Rule-based detection
        violations, rule_anomaly_score = self._compute_rule_based_score(readings, active_rules, lift_type)
        
        # ML-based detection (Isolation Forest + stat)
        detection_method = "statistical_rules"
        if self.trained_models:
            ml_anomaly_score = self._detect_anomaly_with_trained_models(readings, lift_type)
            if ml_anomaly_score > 0.5:
                detection_method += " + ml_model"
        
        # LSTM on history if provided
        seq_len = self.config.seq_len
        if historical_data and TORCH_AVAILABLE and len(historical_data) >= seq_len - 1:
            try:
                lstm_key = f'lstm_{lift_type.lower().replace(" ", "_")}'
                if lstm_key in self.trained_models:
                    lstm_data = self.trained_models[lstm_key]
                    model = lstm_data['model']
                    scaler = lstm_data['scaler']
                    model_seq_len = lstm_data.get('seq_len', seq_len)
                    features = lstm_data.get('features') or self._get_features_for_well_type(lift_type)
                    if features:
                        history_df = pd.DataFrame(historical_data + [readings])
                        for f in features:
                            if f not in history_df.columns:
                                history_df[f] = 0
                        history_arr = history_df[features].fillna(0).values[-model_seq_len:].astype(np.float32)
                        if history_arr.shape[0] == model_seq_len and history_arr.shape[1] == model.input_dim:
                            X_scaled = scaler.transform(history_arr)
                            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
                            with torch.no_grad():
                                reconstructed = model(X_tensor)
                                mse = nn.functional.mse_loss(reconstructed, X_tensor).item()
                            lstm_anomaly_score = min(1.0, mse / self.config.mse_threshold)
                            if lstm_anomaly_score > 0.5:
                                detection_method += " + lstm"
            except Exception as e:
                logger.warning(f"Error in LSTM detection: {e}")
        
        # Combine scores
        combined_score = self._combine_scores(rule_anomaly_score, ml_anomaly_score, lstm_anomaly_score)

        logger.info(f"[AnomalyDetector] rule_score={rule_anomaly_score:.3f} ml_score={ml_anomaly_score:.3f} lstm_score={lstm_anomaly_score:.3f} combined_score={combined_score:.3f}")

        is_anomaly = len(violations) > 0 or combined_score >= self.config.anomaly_threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(combined_score, 3),
            "rule_score": round(rule_anomaly_score, 3),
            "ml_score": round(ml_anomaly_score, 3),
            "lstm_score": round(lstm_anomaly_score, 3),
            "violations": violations,
            "summary": f"{len(violations)} rule violation(s) detected" if violations else "No violations detected",
            "detection_method": detection_method
        }
    
    def _detect_anomaly_with_trained_models(self, readings: Dict, lift_type: str) -> float:
        """Use trained Isolation Forest and statistical models to detect anomalies."""
        well_type_clean = lift_type.lower().replace(" ", "_")
        iso_key = f'iso_{well_type_clean}'
        stat_key = f'stat_{well_type_clean}'
        
        if iso_key not in self.trained_models and stat_key not in self.trained_models:
            return 0.0
        
        features = None
        if iso_key in self.trained_models and 'features' in self.trained_models[iso_key]:
            features = self.trained_models[iso_key]['features']
        if not features and stat_key in self.trained_models and 'features' in self.trained_models[stat_key]:
            features = self.trained_models[stat_key]['features']
        if not features:
            features = WELL_TYPE_FEATURES_FALLBACK.get(lift_type, [])
        if not features:
            return 0.0
        
        values = np.array([[get_safe_value(readings, f) for f in features]])
        
        anomaly_scores = []
        
        if iso_key in self.trained_models:
            model_data = self.trained_models[iso_key]
            model = model_data['model']
            scaler = model_data['scaler']
            model_features = model_data.get('features', features)
            vals = np.array([[get_safe_value(readings, f) for f in model_features]])
            if len(model_features) == getattr(scaler, 'n_features_in_', len(model_features)):
                values_scaled = scaler.transform(vals)
                score = model.decision_function(values_scaled)[0]
                normalized_score = 1 - (score + 1) / 2
                anomaly_scores.append(normalized_score)
        
        if stat_key in self.trained_models:
            stat_model = self.trained_models[stat_key]
            mean = np.array(stat_model['mean'])
            std = np.array(stat_model['std'])
            stat_features = stat_model.get('features', features)
            vals = np.array([get_safe_value(readings, f) for f in stat_features])
            if len(stat_features) == len(mean):
                z_scores = np.abs((vals - mean) / (std + 1e-8))
                max_z = np.max(z_scores)
                stat_anomaly = min(1.0, max_z / self.config.z_score_threshold)
                anomaly_scores.append(stat_anomaly)
        
        return np.mean(anomaly_scores) if anomaly_scores else 0.0
    
    def _get_features_for_well_type(self, lift_type: str) -> List[str]:
        """Get feature list from sensor_definitions (DB) or fallback."""
        if SENSOR_CONFIG_AVAILABLE and get_features_for_lift_type:
            try:
                features = get_features_for_lift_type(lift_type)
                if features:
                    return features
            except Exception as e:
                logger.warning(f"Could not get features from sensor_config for {lift_type}: {e}")
        return WELL_TYPE_FEATURES_FALLBACK.get(lift_type, [])
    
    def train_isolation_forest(self, data: np.ndarray, well_type: str) -> Optional[Dict[str, Any]]:
        """
        Train Isolation Forest on historical sensor data for a well type.
        
        Args:
            data: 2D numpy array of sensor readings (samples x features)
            well_type: Well type identifier for logging
            
        Returns:
            Dict with 'model' and 'scaler' keys, or None if training failed
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available; skipping Isolation Forest training")
            return None
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            model = IsolationForest(
                contamination=self.config.iso_contamination,
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            logger.info(f"Trained Isolation Forest for {well_type}: {data.shape[0]} samples")
            return {'model': model, 'scaler': scaler}
        except Exception as e:
            logger.error(f"Error training Isolation Forest for {well_type}: {e}")
            return None
    
    def train_statistical_model(self, data: np.ndarray) -> Optional[Dict]:
        """Train simple statistical model (mean/std) on data."""
        try:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            return {'mean': mean.tolist(), 'std': std.tolist()}
        except Exception as e:
            logger.error(f"Error training statistical model: {e}")
            return None
    
    def train_lstm_autoencoder(
        self,
        data: np.ndarray,
        well_type: str,
        epochs: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Train LSTM Autoencoder on historical sensor data for a well type.
        
        Args:
            data: 2D numpy array of sensor readings (samples x features)
            well_type: Well type identifier for logging
            epochs: Number of training epochs (uses config default if None)
            
        Returns:
            Dict with 'model', 'scaler', and 'seq_len' keys, or None if failed
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available; skipping LSTM training")
            return None
        
        epochs = epochs or self.config.lstm_epochs
        seq_len = self.config.seq_len
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            
            sequences = [X_scaled[i:i+seq_len] for i in range(len(X_scaled) - seq_len)]
            if not sequences:
                logger.warning(f"Not enough data for LSTM sequences (need {seq_len}+, got {len(X_scaled)})")
                return None
            
            X_tensor = torch.FloatTensor(np.array(sequences))
            input_dim = X_scaled.shape[1]
            
            model = LSTMAutoencoder(input_dim=input_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lstm_lr)
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, X_tensor)
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    logger.info(f"[{well_type}] Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
            
            model.eval()
            logger.info(f"Trained LSTM Autoencoder for {well_type}: {len(sequences)} sequences")
            return {'model': model, 'scaler': scaler, 'seq_len': seq_len}
        except Exception as e:
            logger.error(f"Error training LSTM for {well_type}: {e}")
            return None
    
    def ensemble_score(
        self,
        readings: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None,
        lift_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute ensemble anomaly score combining rules, ISO, and LSTM.
        
        Args:
            readings: Dict of current sensor values
            historical_data: Optional list of previous readings
            lift_type: Well type (required)
            
        Returns:
            Dict with:
                - final_score: Combined anomaly score
                - method_scores: Individual scores by method
                - weights: Configured weights
                - is_anomaly: Boolean classification
                - threshold: Threshold used for classification
        """
        if historical_data is None:
            historical_data = []
        
        result = self.check_anomaly(readings, historical_data, lift_type=lift_type)
        return {
            "final_score": result['anomaly_score'],
            "method_scores": {
                "rules": result['rule_score'],
                "isolation_forest": result['ml_score'],  # Note: ml_score includes iso + stat
                "lstm_autoencoder": result['lstm_score']
            },
            "weights": self.config.weights,
            "is_anomaly": result['is_anomaly'],
            "threshold": self.config.anomaly_threshold
        }

# ============================================================================
# SINGLETON INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================================

_detector: Optional[AnomalyDetector] = None
_detector_config: Optional[AnomalyConfig] = None


def get_detector(config: Optional[AnomalyConfig] = None) -> AnomalyDetector:
    """
    Get or create singleton anomaly detector.
    
    Args:
        config: Optional custom configuration. If provided and different from
                current config, the detector will be recreated.
    
    Returns:
        AnomalyDetector instance
        
    Example:
        detector = get_detector()
        # Or with custom config:
        custom_config = AnomalyConfig(anomaly_threshold=0.8)
        detector = get_detector(config=custom_config)
    """
    global _detector, _detector_config
    
    # Recreate if config changed
    if config is not None and config != _detector_config:
        _detector = None
        _detector_config = config
    
    if _detector is None:
        _detector = AnomalyDetector(config=config or DEFAULT_CONFIG)
        _detector_config = config or DEFAULT_CONFIG
        
    return _detector


def reset_detector() -> None:
    """
    Reset the singleton detector instance.
    
    Useful for testing or when configuration changes.
    Call get_detector() after this to create a new instance.
    """
    global _detector, _detector_config
    _detector = None
    _detector_config = None
    logger.info("Anomaly detector reset")


def check_anomaly(
    readings: Dict[str, Any],
    historical_data: Optional[List[Dict]] = None,
    well_id: Optional[str] = None,
    lift_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function: check for anomalies using default detector.
    
    Args:
        readings: Dict of current sensor values {field: value}
        historical_data: Optional list of previous readings for LSTM
        well_id: Well identifier (optional, for logging)
        lift_type: Well type (Rod Pump/ESP/Gas Lift) - REQUIRED
    
    Returns:
        Dict with keys:
            - is_anomaly: bool
            - anomaly_score: float (0-1)
            - rule_score: float (0-1)
            - ml_score: float (0-1)
            - lstm_score: float (0-1)
            - violations: List of violation dicts
            - summary: str
            - detection_method: str
            
    Example:
        result = check_anomaly(
            readings={'motor_temp': 250, 'motor_current': 45},
            lift_type='ESP'
        )
        if result['is_anomaly']:
            print(f"Anomaly detected! Score: {result['anomaly_score']}")
    """
    detector = get_detector()
    return detector.check_anomaly(readings, historical_data, well_id, lift_type)


def ensemble_score(
    readings: Dict[str, Any],
    historical_data: Optional[List[Dict]] = None,
    lift_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function: compute ensemble score using default detector.
    
    Args:
        readings: Dict of current sensor values
        historical_data: Optional list of previous readings
        lift_type: Well type - REQUIRED
    
    Returns:
        Dict with ensemble scoring details
    """
    detector = get_detector()
    return detector.ensemble_score(readings, historical_data, lift_type=lift_type)