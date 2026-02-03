"""
Unified Anomaly Detection Module
Integrates statistical rules, Isolation Forest, and LSTM for comprehensive anomaly detection.
Loads sensor configurations dynamically from Snowflake instead of hardcoding.
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

try:
    from sensor_config import get_sensor_ranges, get_anomaly_rules, get_features_for_lift_type, get_lift_types
    SENSOR_CONFIG_AVAILABLE = True
except ImportError:
    SENSOR_CONFIG_AVAILABLE = False
    get_features_for_lift_type = None
    get_lift_types = None
    logging.warning("sensor_config module not available; using fallback rules")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
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

# Anomaly detection configuration
ANOMALY_THRESHOLD = 0.75
ML_ONLY_ANOMALY_THRESHOLD = 0.85
WEIGHTS = {'rules': 0.4, 'iso': 0.3, 'lstm': 0.3}
ROLLING_WINDOWS = [1, 3, 6, 12]
WELL_TYPES = ['Rod Pump', 'ESP', 'Gas Lift']
# Fallback features when sensor_config unavailable (matches sensor_definitions)
WELL_TYPE_FEATURES_FALLBACK = {
    'Rod Pump': ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure'],
    'ESP': ['motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage'],
    'Gas Lift': ['injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time'],
}
CONTAMINATION_RATIO = 0.1
CRITICAL_SENSORS = {
    'Rod Pump': ['tubing_pressure', 'strokes_per_minute'],
    'ESP': ['motor_current', 'motor_temp'],
    'Gas Lift': ['injection_rate', 'injection_pressure'],
}

# Constants for ML training and inference
SEQ_LEN = 10
LSTM_EPOCHS = 20  # Increased for better training
LSTM_LR = 0.001
MSE_THRESHOLD = 0.05  # For LSTM anomaly scoring
ISO_CONTAMINATION = 0.05
Z_SCORE_THRESHOLD = 3.0  # For statistical model
RULE_BASE_CONTRIB = 0.35
RULE_EXTRA_CAP = 0.7
RULE_DEV_SCALE = 0.9

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
            decoder_input = hidden.repeat(1, x.shape[1], 1)
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
    """Unified anomaly detection combining rules, Isolation Forest, and LSTM."""
    
    def __init__(self):
        """Initialize detector with rules from dynamic Snowflake config."""
        if not SENSOR_CONFIG_AVAILABLE:
            raise RuntimeError("Dynamic config (Snowflake) is required. Run: python setup_production_db.py")
        
        self.trained_models = self._load_trained_models()
        
        logger.info(f"✓ Anomaly detector initialized (using dynamic Snowflake config)")
        if self.trained_models:
            logger.info(f"✓ Loaded trained models: {', '.join(self.trained_models.keys())}")
    
    def _load_trained_models(self) -> Dict:
        """Load pre-trained models from models/ directory."""
        trained_models = {}
        models_dir = 'models'
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found at {models_dir}. Run: python train_anomaly_models.py")
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
                    scaler = joblib.load(os.path.join(models_dir, f'lstm_scaler_{well_type}.joblib'))
                    model = LSTMAutoencoder(input_dim=input_dim)
                    model.load_state_dict(torch.load(lstm_file, map_location='cpu'))
                    model.eval()
                    trained_models[f'lstm_{well_type}'] = {
                        'model': model, 'scaler': scaler, 'seq_len': SEQ_LEN,
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
                
                dev_factor = deviation_pct / 100.0
                contrib = RULE_BASE_CONTRIB + min(RULE_EXTRA_CAP, dev_factor * RULE_DEV_SCALE)
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
        """Combine rule, ML (iso+stat), and LSTM scores using weights."""
        rules_w = WEIGHTS.get('rules', 0.4)
        ml_w = WEIGHTS.get('iso', 0.3) + WEIGHTS.get('stat', 0.0)  # Treat as combined ML weight
        lstm_w = WEIGHTS.get('lstm', 0.3)
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
        if historical_data and TORCH_AVAILABLE and len(historical_data) >= SEQ_LEN - 1:
            try:
                lstm_key = f'lstm_{lift_type.lower().replace(" ", "_")}'
                if lstm_key in self.trained_models:
                    lstm_data = self.trained_models[lstm_key]
                    model = lstm_data['model']
                    scaler = lstm_data['scaler']
                    seq_len = lstm_data['seq_len']
                    features = lstm_data.get('features') or self._get_features_for_well_type(lift_type)
                    if features:
                        history_df = pd.DataFrame(historical_data + [readings])
                        for f in features:
                            if f not in history_df.columns:
                                history_df[f] = 0
                        history_arr = history_df[features].fillna(0).values[-SEQ_LEN:].astype(np.float32)
                        if history_arr.shape[0] == SEQ_LEN and history_arr.shape[1] == model.input_dim:
                            X_scaled = scaler.transform(history_arr)
                            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
                            with torch.no_grad():
                                reconstructed = model(X_tensor)
                                mse = nn.functional.mse_loss(reconstructed, X_tensor).item()
                            lstm_anomaly_score = min(1.0, mse / MSE_THRESHOLD)
                            if lstm_anomaly_score > 0.5:
                                detection_method += " + lstm"
            except Exception as e:
                logger.warning(f"Error in LSTM detection: {e}")
        
        # Combine scores
        combined_score = self._combine_scores(rule_anomaly_score, ml_anomaly_score, lstm_anomaly_score)

        logger.info(f"[AnomalyDetector] rule_score={rule_anomaly_score:.3f} ml_score={ml_anomaly_score:.3f} lstm_score={lstm_anomaly_score:.3f} combined_score={combined_score:.3f}")

        is_anomaly = len(violations) > 0 or combined_score >= ANOMALY_THRESHOLD

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
                stat_anomaly = min(1.0, max_z / Z_SCORE_THRESHOLD)
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
    
    def train_isolation_forest(self, data: np.ndarray, well_type: str) -> Optional[Dict]:
        """Train Isolation Forest on historical sensor data for a well type."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available; skipping Isolation Forest training")
            return None
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            model = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42)
            model.fit(X_scaled)
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
    
    def train_lstm_autoencoder(self, data: np.ndarray, well_type: str, epochs: int = LSTM_EPOCHS) -> Optional[Dict]:
        """Train LSTM Autoencoder on historical sensor data for a well type."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available; skipping LSTM training")
            return None
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            
            sequences = [X_scaled[i:i+SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]
            if not sequences:
                logger.warning("Not enough data for LSTM sequences")
                return None
            
            X_tensor = torch.FloatTensor(np.array(sequences))
            input_dim = X_scaled.shape[1]
            
            model = LSTMAutoencoder(input_dim=input_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)  # Lower LR for stability
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, X_tensor)
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
            
            model.eval()
            return {'model': model, 'scaler': scaler, 'seq_len': SEQ_LEN}
        except Exception as e:
            logger.error(f"Error training LSTM for {well_type}: {e}")
            return None
    
    def ensemble_score(self, readings: Dict, historical_data: Optional[List[Dict]] = None, lift_type: str = None) -> Dict:
        """Compute ensemble anomaly score combining rules, ISO, and LSTM."""
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
            "weights": WEIGHTS,
            "is_anomaly": result['is_anomaly'],
            "threshold": ANOMALY_THRESHOLD
        }

# Singleton instance
_detector = None

def get_detector() -> AnomalyDetector:
    """Get or create singleton anomaly detector."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector

def check_anomaly(readings: Dict, historical_data: Optional[List[Dict]] = None, well_id: str = None, lift_type: str = None) -> Dict:
    """Convenience function: check for anomalies using default detector."""
    detector = get_detector()
    return detector.check_anomaly(readings, historical_data, well_id, lift_type)

def ensemble_score(readings: Dict, historical_data: Optional[List[Dict]] = None, lift_type: str = None) -> Dict:
    """Convenience function: compute ensemble score using default detector."""
    detector = get_detector()
    return detector.ensemble_score(readings, historical_data, lift_type=lift_type)