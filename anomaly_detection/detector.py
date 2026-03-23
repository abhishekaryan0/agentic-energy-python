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
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import joblib

# ============================================================================
# OPTIONAL DEPENDENCY HANDLING
# ============================================================================

try:
    from config.sensor_config import get_sensor_ranges, get_features_for_lift_type, get_lift_types
    SENSOR_CONFIG_AVAILABLE = True
except ImportError:
    SENSOR_CONFIG_AVAILABLE = False
    get_features_for_lift_type = None
    get_lift_types = None
    get_sensor_ranges = None
    logging.warning("sensor_config module not available; using fallback rules (S3-only / ML-only mode)")

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

def _ensure_models_from_s3(models_dir_abs: str) -> bool:
    """
    If S3_BUCKET (and optionally S3_PREFIX, MODEL_VERSION) are set in env,
    download anomaly_detection tarball from S3 and extract into models_dir_abs.
    Returns True if extraction succeeded and anomaly_detection/ exists, else False.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return False
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed; cannot download models from S3. pip install boto3")
        return False
    prefix = (os.environ.get("S3_PREFIX") or "models/anomaly").rstrip("/")
    version = os.environ.get("MODEL_VERSION") or "v1"
    key = f"{prefix}/anomaly_detection_{version}.tar.gz"
    logger.info("Downloading anomaly detection models from s3://%s/%s", bucket, key)
    try:
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tmp_path = f.name
        try:
            s3.download_file(bucket, key, tmp_path)
        except Exception as e:
            logger.warning("S3 download failed: %s", e)
            return False
        os.makedirs(models_dir_abs, exist_ok=True)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=models_dir_abs)
        logger.info("Extracted models to %s", models_dir_abs)
        base = os.path.join(models_dir_abs, "anomaly_detection")
        return os.path.isdir(base)
    except Exception as e:
        logger.warning("Failed to fetch models from S3: %s", e)
        return False
    finally:
        if "tmp_path" in dir() and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
        rule_deviation_pct: Percentage band around average for rule checks
        rule_use_client_bounds: Whether to clip average band to client min/max
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
    rule_deviation_pct: float = 50.0
    rule_use_client_bounds: bool = True
    
    # Models directory: models/anomaly_detection/wells/<safe_id>/, well_types/<type>/
    models_dir: str = 'models'
    anomaly_detection_subdir: str = 'anomaly_detection'
    wells_subdir: str = 'wells'
    well_types_subdir: str = 'well_types'
    # Fixed filenames inside each well (or well_type) folder
    iso_filename: str = 'isolation_forest.joblib'
    stat_filename: str = 'statistical_model.joblib'
    lstm_pt_filename: str = 'lstm_autoencoder.pt'
    lstm_scaler_filename: str = 'lstm_scaler.joblib'
    lstm_meta_filename: str = 'lstm_metadata.joblib'

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

def well_id_to_safe(well_id: Optional[str], normalize_lower: bool = True) -> str:
    """Convert well_id to a safe filename/key segment (alphanumeric and underscores, max 64 chars)."""
    if not well_id:
        return "well"
    safe = re.sub(r"[^\w\-]", "_", str(well_id).strip())[:64]
    safe = safe.strip("_") or "well"
    return safe.lower() if normalize_lower else safe


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
    """
    Get sensor rules for a well type from dynamic config (Snowflake).
    
    Returns a dict of client-provided ranges that can be used as hard bounds
    around the statistically learned average band. In SageMaker (no Snowflake),
    this returns an empty dict and rule-based checks rely purely on model stats.
    """
    if os.environ.get("SAGEMAKER_REGION"):
        return {}
    # When sensor_config / Snowflake is not available, allow pure ML/S3-based detection.
    if not SENSOR_CONFIG_AVAILABLE:
        logging.warning(
            "Dynamic sensor_config not available; get_well_type_rules returning empty rules "
            "(rules component disabled, ML models + fallback bands only)."
        )
        return {}
    try:
        ranges = get_sensor_ranges(lift_type)
        rules = {}
        for field_name, range_info in ranges.items():
            rules[field_name] = {
                "min": range_info.get('min'),
                "max": range_info.get('max'),
                "unit": range_info.get('unit'),
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
        
        # In S3-only mode, allow running without sensor_config/Snowflake by using fallback rules.
        if not SENSOR_CONFIG_AVAILABLE and not os.environ.get("SAGEMAKER_REGION"):
            logger.warning(
                "sensor_config (Snowflake) not available. "
                "Proceeding with fallback anomaly rules and ML models only."
            )
        
        self.trained_models: Dict[str, Any] = self._load_trained_models()
        
        logger.info(f"✓ Anomaly detector initialized (using dynamic Snowflake config)")
        logger.info(f"  - Threshold: {self.config.anomaly_threshold}")
        logger.info(f"  - Weights: {self.config.weights}")
        if self.trained_models:
            logger.info(f"✓ Loaded trained models: {', '.join(self.trained_models.keys())}")
    
    def _load_trained_models(self) -> Dict[str, Any]:
        """Load pre-trained models from models/anomaly_detection/ (wells/<safe_id>/ and well_types/<type>/).
        Uses ANOMALY_MODELS_DIR env var if set (e.g. /opt/ml/model in SageMaker).
        If local models are missing and S3_BUCKET is set in env, downloads from S3 and uses them."""
        trained_models: Dict[str, Any] = {}
        models_root = os.environ.get("ANOMALY_MODELS_DIR", self.config.models_dir)
        # Resolve to absolute path when using default local models dir (for S3 download target)
        if models_root == self.config.models_dir:
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_root = os.path.join(_project_root, self.config.models_dir)
        # Base path for anomaly detection: models/anomaly_detection/
        base_dir = os.path.join(models_root, self.config.anomaly_detection_subdir)
        wells_dir = os.path.join(base_dir, self.config.wells_subdir)
        well_types_dir = os.path.join(base_dir, self.config.well_types_subdir)

        # When S3_BUCKET is set, always load from S3 (download and use). Skip only in SageMaker (ANOMALY_MODELS_DIR set).
        if os.environ.get("S3_BUCKET") and not os.environ.get("ANOMALY_MODELS_DIR"):
            if not _ensure_models_from_s3(models_root):
                logger.warning("S3 download failed or no object; no models loaded.")
            if not os.path.exists(base_dir):
                return trained_models
        elif not os.path.exists(base_dir):
            logger.warning(f"Models directory not found at {base_dir}. Set S3_BUCKET in .env or run: python -m anomaly_detection.train_models")
            return trained_models

        def _load_well_type_from_folder(type_dir: str, well_type: str) -> None:
            """Load iso/stat/lstm from a folder with fixed filenames."""
            iso_f = os.path.join(type_dir, self.config.iso_filename)
            stat_f = os.path.join(type_dir, self.config.stat_filename)
            if os.path.exists(iso_f):
                try:
                    d = joblib.load(iso_f)
                    if isinstance(d, dict) and 'model' in d and 'scaler' in d:
                        trained_models[f'iso_{well_type}'] = d
                        logger.info(f"✓ Loaded isolation forest model for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load {iso_f}: {e}")
            if os.path.exists(stat_f):
                try:
                    d = joblib.load(stat_f)
                    if isinstance(d, dict) and 'mean' in d and 'std' in d:
                        trained_models[f'stat_{well_type}'] = d
                        logger.info(f"✓ Loaded statistical model for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load {stat_f}: {e}")
            if TORCH_AVAILABLE:
                lstm_pt = os.path.join(type_dir, self.config.lstm_pt_filename)
                lstm_meta = os.path.join(type_dir, self.config.lstm_meta_filename)
                lstm_sc = os.path.join(type_dir, self.config.lstm_scaler_filename)
                if os.path.exists(lstm_pt) and os.path.exists(lstm_meta) and os.path.exists(lstm_sc):
                    try:
                        meta = joblib.load(lstm_meta)
                        scaler = joblib.load(lstm_sc)
                        model = LSTMAutoencoder(input_dim=meta['input_dim'])
                        model.load_state_dict(torch.load(lstm_pt, map_location='cpu'))
                        model.eval()
                        trained_models[f'lstm_{well_type}'] = {
                            'model': model, 'scaler': scaler,
                            'seq_len': meta.get('seq_len', SEQ_LEN),
                            'features': meta.get('features', [])
                        }
                        logger.info(f"✓ Loaded LSTM autoencoder for {well_type}")
                    except Exception as e:
                        logger.error(f"Failed to load LSTM for {well_type}: {e}")

        def _load_well_from_folder(well_dir: str, safe_id: str) -> None:
            """Load iso/stat/lstm for one well from its folder."""
            iso_f = os.path.join(well_dir, self.config.iso_filename)
            stat_f = os.path.join(well_dir, self.config.stat_filename)
            if os.path.exists(iso_f):
                try:
                    d = joblib.load(iso_f)
                    if isinstance(d, dict) and 'model' in d and 'scaler' in d:
                        trained_models[f'iso_well_{safe_id}'] = d
                        logger.info(f"✓ Loaded isolation forest model for well {safe_id}")
                except Exception as e:
                    logger.error(f"Failed to load {iso_f}: {e}")
            if os.path.exists(stat_f):
                try:
                    d = joblib.load(stat_f)
                    if isinstance(d, dict) and 'mean' in d and 'std' in d:
                        trained_models[f'stat_well_{safe_id}'] = d
                        logger.info(f"✓ Loaded statistical model for well {safe_id}")
                except Exception as e:
                    logger.error(f"Failed to load {stat_f}: {e}")
            if TORCH_AVAILABLE:
                lstm_pt = os.path.join(well_dir, self.config.lstm_pt_filename)
                lstm_meta = os.path.join(well_dir, self.config.lstm_meta_filename)
                lstm_sc = os.path.join(well_dir, self.config.lstm_scaler_filename)
                if os.path.exists(lstm_pt) and os.path.exists(lstm_meta) and os.path.exists(lstm_sc):
                    try:
                        meta = joblib.load(lstm_meta)
                        scaler = joblib.load(lstm_sc)
                        model = LSTMAutoencoder(input_dim=meta['input_dim'])
                        model.load_state_dict(torch.load(lstm_pt, map_location='cpu'))
                        model.eval()
                        trained_models[f'lstm_well_{safe_id}'] = {
                            'model': model, 'scaler': scaler,
                            'seq_len': meta.get('seq_len', SEQ_LEN),
                            'features': meta.get('features', [])
                        }
                        logger.info(f"✓ Loaded LSTM autoencoder for well {safe_id}")
                    except Exception as e:
                        logger.error(f"Failed to load LSTM for well {safe_id}: {e}")

        # 1) Load well-type fallback: first from well_types/<type>/, then flat (backward compat)
        well_types = ['rod_pump', 'esp', 'gas_lift']
        for well_type in well_types:
            type_dir = os.path.join(well_types_dir, well_type)
            if os.path.isdir(type_dir):
                _load_well_type_from_folder(type_dir, well_type)
            else:
                # Flat files (backward compat) under models/anomaly_detection/
                iso_file = os.path.join(base_dir, f'isolation_forest_{well_type}.joblib')
                stat_file = os.path.join(base_dir, f'statistical_model_{well_type}.joblib')
                lstm_file = os.path.join(base_dir, f'lstm_autoencoder_{well_type}.pt')
                lstm_meta_file = os.path.join(base_dir, f'lstm_metadata_{well_type}.joblib')
                lstm_scaler_file = os.path.join(base_dir, f'lstm_scaler_{well_type}.joblib')
                if os.path.exists(iso_file):
                    try:
                        d = joblib.load(iso_file)
                        if isinstance(d, dict) and 'model' in d and 'scaler' in d:
                            trained_models[f'iso_{well_type}'] = d
                            logger.info(f"✓ Loaded isolation forest model for {well_type}")
                    except Exception as e:
                        logger.error(f"Failed to load {iso_file}: {e}")
                if os.path.exists(stat_file):
                    try:
                        d = joblib.load(stat_file)
                        if isinstance(d, dict) and 'mean' in d and 'std' in d:
                            trained_models[f'stat_{well_type}'] = d
                            logger.info(f"✓ Loaded statistical model for {well_type}")
                    except Exception as e:
                        logger.error(f"Failed to load {stat_file}: {e}")
                if TORCH_AVAILABLE and os.path.exists(lstm_file) and os.path.exists(lstm_meta_file) and os.path.exists(lstm_scaler_file):
                    try:
                        meta = joblib.load(lstm_meta_file)
                        scaler = joblib.load(lstm_scaler_file)
                        model = LSTMAutoencoder(input_dim=meta['input_dim'])
                        model.load_state_dict(torch.load(lstm_file, map_location='cpu'))
                        model.eval()
                        trained_models[f'lstm_{well_type}'] = {
                            'model': model, 'scaler': scaler,
                            'seq_len': meta.get('seq_len', SEQ_LEN),
                            'features': meta.get('features', [])
                        }
                        logger.info(f"✓ Loaded LSTM autoencoder for {well_type}")
                    except Exception as e:
                        logger.error(f"Failed to load LSTM for {well_type}: {e}")

        # 2) Load per-well models from models/wells/<safe_id>/ (use lowercase key for consistent lookup)
        if os.path.isdir(wells_dir):
            try:
                for name in os.listdir(wells_dir):
                    well_dir = os.path.join(wells_dir, name)
                    if os.path.isdir(well_dir):
                        _load_well_from_folder(well_dir, name.lower())
            except OSError as e:
                logger.warning(f"Could not list wells directory: {e}")

        return trained_models
    
    def _compute_rule_based_score(
        self,
        readings: Dict,
        active_rules: Dict,
        lift_type: str,
        well_id: Optional[str] = None
    ) -> Tuple[List, float]:
        """
        Compute rule-based anomaly score and violations using an average-based band.
        
        Primary logic:
            - Use statistical model mean per sensor as the "average" value.
            - Build a band: mean ± (rule_deviation_pct %).
            - Optionally clip that band to client-provided min/max (active_rules).
            - Flag values that fall outside this band.
        
        Fallback:
            - If no statistical model is available for a sensor, fall back to
              direct client min/max checks (legacy behavior).
        """
        violations: List[Dict[str, Any]] = []
        rule_anomaly_score = 0.0

        # Look up statistical model (prefer per-well, then per lift-type)
        well_type_clean = lift_type.lower().replace(" ", "_")
        safe_well = well_id_to_safe(well_id) if well_id else None
        stat_key = None
        if safe_well and f'stat_well_{safe_well}' in self.trained_models:
            stat_key = f'stat_well_{safe_well}'
        elif f'stat_{well_type_clean}' in self.trained_models:
            stat_key = f'stat_{well_type_clean}'

        stat_model = self.trained_models.get(stat_key) if stat_key else None
        feature_means: Dict[str, float] = {}
        if stat_model is not None and 'mean' in stat_model:
            means = np.array(stat_model['mean'])
            stat_features = stat_model.get('features')
            # Only build mapping when features metadata is available and lengths match
            if stat_features and len(stat_features) == len(means):
                feature_means = {
                    str(f): float(means[idx])
                    for idx, f in enumerate(stat_features)
                }

        deviation_fraction = max(self.config.rule_deviation_pct / 100.0, 0.0)

        for field, value in readings.items():
            if value is None:
                continue

            rule = active_rules.get(field) or {}
            client_min = rule.get("min")
            client_max = rule.get("max")
            sensor_unit = rule.get("unit") or ""

            # Decide average to use for this field
            avg = feature_means.get(field)

            # If we have a valid average, build mean-based band
            if avg is not None and not np.isnan(avg):
                # Handle negative means by always taking numeric min/max of the endpoints
                low_candidate = avg * (1.0 - deviation_fraction)
                high_candidate = avg * (1.0 + deviation_fraction)
                band_min = min(low_candidate, high_candidate)
                band_max = max(low_candidate, high_candidate)

                # Optionally clip band by client min/max
                if self.config.rule_use_client_bounds:
                    if client_min is not None:
                        band_min = max(band_min, client_min)
                    if client_max is not None:
                        band_max = min(band_max, client_max)

                # If band collapsed or invalid, fall back to client-only logic
                if band_max <= band_min:
                    band_min = client_min
                    band_max = client_max
            else:
                # No statistical average available – rely purely on client bounds
                band_min = client_min
                band_max = client_max

            # If we still don't have a usable band, skip this sensor
            if band_min is None or band_max is None:
                continue

            # Check if current value is outside the allowed band
            if value < band_min or value > band_max:
                # Deviation as percentage beyond the band width
                width = band_max - band_min if band_max != band_min else 1e-8
                if value < band_min:
                    deviation_pct = abs(band_min - value) / width * 100.0
                else:
                    deviation_pct = abs(value - band_max) / width * 100.0

                min_str = self._format_value(band_min)
                max_str = self._format_value(band_max)
                val_str = self._format_value(value)
                unit_str = f" {sensor_unit}" if sensor_unit else ""
                if avg is not None and not np.isnan(avg):
                    avg_str = self._format_value(avg)
                    violation_text = (
                        f"Outside expected band around average. "
                        f"Avg {avg_str}{unit_str}, allowed {min_str}{unit_str} - {max_str}{unit_str}, "
                        f"got {val_str}{unit_str} ({deviation_pct:.1f}% beyond band)"
                    )
                else:
                    violation_text = (
                        f"Out of range. Expected {min_str}{unit_str} - {max_str}{unit_str}, "
                        f"got {val_str}{unit_str} ({deviation_pct:.1f}% deviation)"
                    )

                logger.debug(
                    "[AnomalyDetector] lift_type=%s field=%s avg=%s band=[%s, %s] value=%s unit=%s",
                    lift_type,
                    field,
                    avg,
                    band_min,
                    band_max,
                    value,
                    sensor_unit,
                )

                violations.append({
                    "field": field,
                    "value": value,
                    "min": band_min,
                    "max": band_max,
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
        
        # Rule-based detection (average-based band using statistical model + client ranges)
        violations, rule_anomaly_score = self._compute_rule_based_score(
            readings,
            active_rules,
            lift_type,
            well_id=well_id,
        )
        
        # ML-based detection (Isolation Forest + stat); prefer per-well models
        detection_method = "statistical_rules"
        if self.trained_models:
            ml_anomaly_score = self._detect_anomaly_with_trained_models(readings, lift_type, well_id=well_id)
            if ml_anomaly_score > 0.5:
                detection_method += " + ml_model"
        
        # LSTM on history if provided; prefer per-well model then lift_type
        seq_len = self.config.seq_len
        if historical_data and TORCH_AVAILABLE and len(historical_data) >= seq_len - 1:
            try:
                well_type_clean = lift_type.lower().replace(" ", "_")
                lstm_key = (f'lstm_well_{well_id_to_safe(well_id)}' if well_id else None)
                if lstm_key and lstm_key not in self.trained_models:
                    lstm_key = None
                if not lstm_key:
                    lstm_key = f'lstm_{well_type_clean}'
                if lstm_key in self.trained_models:
                    lstm_data = self.trained_models[lstm_key]
                    model = lstm_data['model']
                    scaler = lstm_data['scaler']
                    model_seq_len = lstm_data.get('seq_len', seq_len)
                    features = lstm_data.get('features') or self._get_features_for_well_type(lift_type)
                    if features and hasattr(model, 'input_dim'):
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

        logger.debug(f"[AnomalyDetector] rule_score={rule_anomaly_score:.3f} ml_score={ml_anomaly_score:.3f} lstm_score={lstm_anomaly_score:.3f} combined_score={combined_score:.3f}")

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
    
    def _detect_anomaly_with_trained_models(self, readings: Dict, lift_type: str, well_id: Optional[str] = None) -> float:
        """Use trained Isolation Forest and statistical models; prefer per-well models over lift_type."""
        well_type_clean = lift_type.lower().replace(" ", "_")
        safe_well = well_id_to_safe(well_id) if well_id else None
        # Prefer per-well models, then fall back to lift_type
        iso_key = f'iso_well_{safe_well}' if (well_id and f'iso_well_{safe_well}' in self.trained_models) else f'iso_{well_type_clean}'
        stat_key = f'stat_well_{safe_well}' if (well_id and f'stat_well_{safe_well}' in self.trained_models) else f'stat_{well_type_clean}'
        
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
        if not SKLEARN_AVAILABLE or StandardScaler is None:
            logger.warning("scikit-learn not available; skipping LSTM training (scaler required)")
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
        well_id: Optional[str] = None,
        lift_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute ensemble anomaly score combining rules, ISO, and LSTM.
        
        Args:
            readings: Dict of current sensor values
            historical_data: Optional list of previous readings
            well_id: Well identifier (for per-well model lookup)
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
        
        result = self.check_anomaly(readings, historical_data, well_id=well_id, lift_type=lift_type)
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
    well_id: Optional[str] = None,
    lift_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function: compute ensemble score using default detector.
    
    Args:
        readings: Dict of current sensor values
        historical_data: Optional list of previous readings
        well_id: Well identifier (for per-well model lookup)
        lift_type: Well type - REQUIRED
    
    Returns:
        Dict with ensemble scoring details
    """
    detector = get_detector()
    return detector.ensemble_score(readings, historical_data, well_id=well_id, lift_type=lift_type)