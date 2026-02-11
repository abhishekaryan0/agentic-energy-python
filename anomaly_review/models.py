"""
ML models for anomaly detection in well data.

Provides:
- LSTMRegressor: Time-series forecasting model
- LSTMAutoencoder: Unsupervised anomaly detection model
- Training and prediction functions
- Model persistence (save/load)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from anomaly_review.utils import safe_divide
from anomaly_review.config import MODEL_DIR, MODEL_CACHE_DAYS, LIFT_TYPE_FEATURES

logger = logging.getLogger('AnomalyReviewModels')

# =============================================================================
# TORCH AVAILABILITY CHECK
# =============================================================================

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logger.warning("PyTorch not available; LSTM models disabled")


# =============================================================================
# LSTM MODELS (PyTorch)
# =============================================================================

if TORCH_AVAILABLE:
    class LSTMRegressor(nn.Module):
        """
        LSTM model for time-series regression (production forecasting).
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (default 1 for regression)
        """
        def __init__(self, input_dim: int, hidden_dim: int = 50, num_layers: int = 2, output_dim: int = 1):
            super(LSTMRegressor, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])  # Predict using the last time step
            return out

    class LSTMAutoencoder(nn.Module):
        """
        LSTM Autoencoder for unsupervised anomaly detection.
        
        Learns to reconstruct normal patterns; high reconstruction error indicates anomalies.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
            super(LSTMAutoencoder, self).__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            encoded, (hidden, cell) = self.encoder(x)
            decoder_input = torch.zeros_like(x)
            decoded, _ = self.decoder(decoder_input, (hidden, cell))
            return self.fc(decoded)
else:
    # Placeholder classes when PyTorch not available
    class LSTMRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")
    
    class LSTMAutoencoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")


# =============================================================================
# SEQUENCE PREPARATION
# =============================================================================

def prepare_sequences(
    df: pd.DataFrame, 
    features: List[str], 
    target: str = None, 
    window_size: int = 7
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert DataFrame to 3D tensors (Samples, Window, Features).
    
    Args:
        df: Input DataFrame
        features: List of feature column names
        target: Optional target column for supervised learning
        window_size: Sequence window size
        
    Returns:
        Tuple of (X, y) arrays or (None, None) if insufficient data
    """
    if len(df) <= window_size:
        return None, None
    
    data = df[features].values
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        if target and target in df.columns:
            target_val = df.iloc[i + window_size][target]
            y.append(target_val)
    
    return np.array(X), np.array(y) if y else None


# =============================================================================
# FEATURE ENGINEERING & SELECTION
# =============================================================================

def engineer_physics_features(df: pd.DataFrame, lift_type: str = "Unknown") -> pd.DataFrame:
    """
    Enrich DataFrame with physics-based engineered features.
    
    Uses safe_divide to avoid crashes.
    """
    df = df.copy()
    
    # 1. Energy Intensity (Amps per Barrel)
    # Only if we have liquid production
    total_liquid = df['oil_volume'] + df['water_volume']
    df['energy_intensity'] = safe_divide(df['motor_current'], total_liquid)
    
    # 2. Water Cut (Water / Total Liquid)
    df['water_cut'] = safe_divide(df['water_volume'], total_liquid)
    
    # 3. Gas-Oil Ratio
    df['gas_oil_ratio'] = safe_divide(df['gas_volume'], df['oil_volume'])
    
    # 4. Pressure Differential (Casing - Tubing)
    if 'casing_pressure' in df.columns and 'tubing_pressure' in df.columns:
        df['pressure_diff'] = df['casing_pressure'] - df['tubing_pressure']
        
    return df


def select_top_features_rf(df: pd.DataFrame, candidate_features: List[str], target_col: str = 'oil_volume', top_n: int = 10) -> List[str]:
    """
    Select top features using Random Forest importance.
    """
    if df.empty or len(df) < 50:
        return candidate_features[:top_n]
        
    try:
        # Prepare data
        valid_feats = [f for f in candidate_features if f in df.columns]
        if not valid_feats:
            return []
            
        X = df[valid_feats].fillna(0)
        y = df[target_col].fillna(0)
        
        # Train lightweight RF
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        
        # Get importance
        importances = pd.Series(rf.feature_importances_, index=valid_feats)
        top_features = importances.nlargest(top_n).index.tolist()
        
        logger.info(f"Auto-Selected Top {top_n} Features: {top_features}")
        return top_features
        
    except Exception as e:
        logger.warning(f"Feature selection failed, using defaults: {e}")
        return candidate_features[:top_n]


def safe_prepare_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Prepare DataFrame for prediction by ensuring all required features exist.
    
    If features are missing (e.g., failed sensor), they are created with 0 values
    to prevent model crash.
    """
    df_safe = df.copy()
    missing = [f for f in features if f not in df_safe.columns]
    
    if missing:
        logger.warning(f"Missing features for prediction: {missing}. Imputing with Mean (via NaNs).")
        for f in missing:
            df_safe[f] = np.nan
            
    # Return features in exact order expected by model/scaler
    return df_safe[features]


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lstm_model(df: pd.DataFrame, features: List[str], target: str) -> Optional[Dict]:
    """
    Train LSTM regressor model for production forecasting.
    
    Args:
        df: Training DataFrame with features and target
        features: List of feature column names
        target: Target column name
        
    Returns:
        Dict with model, type, features, and window_size or None if training fails
    """
    if not TORCH_AVAILABLE:
        return None
    
    window_size = 7
    X_np, y_np = prepare_sequences(df, features, target, window_size)
    
    if X_np is None or len(X_np) < 30:
        return None
    
    X_tensor = torch.FloatTensor(X_np)
    y_tensor = torch.FloatTensor(y_np).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = LSTMRegressor(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(50):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    return {
        'model': model,
        'type': 'lstm_regressor',
        'features': features,
        'window_size': window_size
    }


def train_autoencoder_model(df: pd.DataFrame, features: List[str]) -> Optional[Dict]:
    """
    Train LSTM Autoencoder for unsupervised anomaly detection.
    
    Args:
        df: Training DataFrame with features
        features: List of feature column names
        
    Returns:
        Dict with model, type, features, threshold, and window_size or None if training fails
    """
    if not TORCH_AVAILABLE:
        return None
    
    window_size = 7
    X_np, _ = prepare_sequences(df, features, target=None, window_size=window_size)
    
    if X_np is None or len(X_np) < 50:
        return None
    
    X_tensor = torch.FloatTensor(X_np)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = LSTMAutoencoder(input_dim=len(features), hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(50):
        for batch in loader:
            batch_X = batch[0]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
    
    # Calculate threshold (Mean + 2 Sigma on training data for better sensitivity)
    model.eval()
    with torch.no_grad():
        reconstructions = model(X_tensor)
        train_loss = torch.mean((X_tensor - reconstructions) ** 2, dim=[1, 2])
        threshold = torch.mean(train_loss) + 2 * torch.std(train_loss)
        
        # Safety: If threshold is NaN (e.g., from constant input), use default
        if torch.isnan(threshold) or threshold == 0:
            threshold = 0.5
            logger.warning(f"Autoencoder threshold was NaN/Zero. Defaulting to {threshold}")
        else:
            # Ensure minimum threshold to avoid noise sensititivity
            threshold = max(float(threshold), 0.05)
    
    return {
        'model': model,
        'type': 'lstm_autoencoder',
        'features': features,
        'threshold': float(threshold),
        'window_size': window_size
    }


def get_or_train_model(well_id: str, df: pd.DataFrame) -> Dict:
    """
    Get cached model or train new models for a well.
    
    Args:
        well_id: Well identifier
        df: DataFrame with well data
        
    Returns:
        Dict with models, scaler, imputer, and features
    """
    # Crash Prevention: Sparse data
    if len(df) < 60:
        logger.info(f"Skipping model training for {well_id}: insufficient data ({len(df)} rows)")
        return {}

    # Try loading cached model first
    cached_art = load_models_from_disk(well_id)
    if cached_art:
        return cached_art

    logger.info(f"Training new models for {well_id}...")
    
    # Determine lift type for feature selection
    latest_lift_type = df['lift_type'].mode()[0] if 'lift_type' in df.columns and not df['lift_type'].empty else "Rod Pump"

    # 1. Physics Feature Engineering
    df = engineer_physics_features(df, lift_type=latest_lift_type)
    
    # 2. Candidate Selection based on Lift Type
    base_features = LIFT_TYPE_FEATURES.get(latest_lift_type, LIFT_TYPE_FEATURES.get("Rod Pump")) # Default to Rod Pump if unknown
    
    # Add engineered features to candidates if they exist
    engineered = ['energy_intensity', 'water_cut', 'gas_oil_ratio', 'pressure_diff']
    base_features += [f for f in engineered if f in df.columns]

    # Filter for columns that actually exist in DF
    candidate_features = [c for c in base_features if c in df.columns]
    
    # 3. Automated Feature Selection (The "Smart" Filter)
    # Refine the list to the top 10 most predictive features
    selected_features = select_top_features_rf(df, candidate_features, target_col='oil_volume', top_n=10)
    
    if not selected_features:
        logger.warning(f"No valid features found for {well_id}. Using basic fallback.")
        selected_features = ['motor_current', 'tubing_pressure'] # Minimal fallback
        
    logger.info(f"Training model for {well_id} ({latest_lift_type}) using {len(selected_features)} features.")
    
    # Prepare training data with SELECTED features
    # Ensure 'oil_volume' is available for target, even if not a feature
    required_cols = list(set(selected_features + ['oil_volume']))
    train_df = df[required_cols].copy()
    
    # Handle missing values for training
    train_df = train_df.ffill().bfill().fillna(0)  # Fill NaNs before scaling

    # Filter for valid training data (e.g., positive oil volume)
    train_df_filtered = train_df[train_df['oil_volume'] > 0.1].copy()
    
    if len(train_df_filtered) < 30:
        logger.warning(f"Insufficient valid training data for {well_id} after filtering. ({len(train_df_filtered)} rows)")
        return {}

    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    
    # Transform
    X = scaler.fit_transform(imputer.fit_transform(train_df_filtered[selected_features]))
    
    models = {}
    
    # A. Random Forest Regressor (For Production Prediction)
    try:
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X, train_df_filtered['oil_volume'].values) 
        models['rf_model'] = rf
    except Exception as e:
        logger.error(f"RF training failed: {e}")

    # B. LSTM Autoencoder (For Anomaly Detection)
    try:
        # Train Autoencoder on the same filtered/imputed data
        # We re-use 'train_df_filtered' but need to pass the full original DF pattern for sequences
        # ideally. However, train_autoencoder_model takes (df, features).
        ae_artifact = train_autoencoder_model(df, selected_features)
        
        if ae_artifact:
            models['autoencoder'] = ae_artifact['model']
            models['threshold'] = ae_artifact['threshold']
            logger.info(f"Autoencoder trained. Threshold: {ae_artifact['threshold']:.4f}")

    except Exception as e:
        logger.error(f"Autoencoder training failed: {e}")

    # Save artifacts (threshold from autoencoder when available)
    artifacts = {
        'models': models,
        'scaler': scaler,
        'imputer': imputer,
        'features': selected_features,
        'threshold': models.get('threshold', 0.5),
        'trained_date': datetime.now().isoformat()
    }

    save_models_to_disk(well_id, artifacts)

    return artifacts
        



# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def get_model_dir(well_id: str) -> str:
    """Get or create model directory for a well."""
    path = os.path.join(MODEL_DIR, well_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_models_to_disk(well_id: str, artifacts: Dict) -> None:
    """
    Save models, scaler, imputer, and metadata to disk.
    
    Args:
        well_id: Well identifier
        artifacts: Dict with models, scaler, imputer, features
    """
    try:
        model_dir = get_model_dir(well_id)
        
        # Save metadata (threshold is at top level in artifacts from get_or_train_model)
        metadata = {
            'features': artifacts['features'],
            'threshold': artifacts.get('threshold'),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Save sklearn objects
        joblib.dump(artifacts['scaler'], os.path.join(model_dir, f'{well_id}_scaler.joblib'))
        joblib.dump(artifacts['imputer'], os.path.join(model_dir, f'{well_id}_imputer.joblib'))
        
        if 'rf_model' in artifacts['models']:
            joblib.dump(artifacts['models']['rf_model'], os.path.join(model_dir, f'{well_id}_rf_baseline.joblib'))
        
        # Save PyTorch models
        if TORCH_AVAILABLE:
            if 'lstm_regressor' in artifacts['models']:
                torch.save(
                    artifacts['models']['lstm_regressor'].state_dict(),
                    os.path.join(model_dir, f'{well_id}_lstm_forecast.pth')
                )
            if 'autoencoder' in artifacts['models']:
                torch.save(
                    artifacts['models']['autoencoder'].state_dict(),
                    os.path.join(model_dir, f'{well_id}_autoencoder_detect.pth')
                )
        
        logger.info(f"Saved models for {well_id} to {model_dir}")
    except Exception as e:
        logger.error(f"Failed to save models for {well_id}: {e}")


def load_models_from_disk(well_id: str) -> Optional[Dict]:
    """
    Load models if they exist and are recent.
    
    Args:
        well_id: Well identifier
        
    Returns:
        Dict with models, scaler, imputer, features or None if not available
    """
    try:
        model_dir = get_model_dir(well_id)
        meta_path = os.path.join(model_dir, 'metadata.json')
        
        if not os.path.exists(meta_path):
            return None
        
        # Check age
        file_time = datetime.fromtimestamp(os.path.getmtime(meta_path))
        if datetime.now() - file_time > timedelta(days=MODEL_CACHE_DAYS):
            logger.info(f"Cached models for {well_id} are expired. Retraining...")
            return None
        
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        features = metadata['features']
        threshold = metadata.get('threshold')
        
        # Load sklearn objects
        scaler = joblib.load(os.path.join(model_dir, f'{well_id}_scaler.joblib'))
        imputer = joblib.load(os.path.join(model_dir, f'{well_id}_imputer.joblib'))
        
        models = {}
        rf_path = os.path.join(model_dir, f'{well_id}_rf_baseline.joblib')
        if os.path.exists(rf_path):
            models['rf_model'] = joblib.load(rf_path)
        
        # Load PyTorch models
        if TORCH_AVAILABLE:
            lstm_path = os.path.join(model_dir, f'{well_id}_lstm_forecast.pth')
            if os.path.exists(lstm_path):
                lstm = LSTMRegressor(input_dim=len(features))
                lstm.load_state_dict(torch.load(lstm_path))
                lstm.eval()
                models['lstm_regressor'] = lstm
            
            ae_path = os.path.join(model_dir, f'{well_id}_autoencoder_detect.pth')
            if os.path.exists(ae_path):
                ae = LSTMAutoencoder(input_dim=len(features), hidden_dim=64)
                ae.load_state_dict(torch.load(ae_path))
                ae.eval()
                models['autoencoder'] = ae
        
        logger.info(f"Loaded cached models for {well_id} from disk.")
        return {
            'models': models,
            'scaler': scaler,
            'imputer': imputer,
            'features': features,
            'threshold': metadata.get('threshold', 0.5),
        }
        
    except Exception as e:
        logger.warning(f"Failed to load cached models for {well_id}: {e}")
        return None


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_with_lstm(
    model, 
    df: pd.DataFrame, 
    features: List[str], 
    scaler, 
    imputer
) -> pd.Series:
    """
    Generate predictions using LSTM model.
    
    Args:
        model: Trained LSTM model
        df: DataFrame with features
        features: List of feature column names
        scaler: Fitted StandardScaler
        imputer: Fitted SimpleImputer
        
    Returns:
        Series with predictions aligned with df index
    """
    if not TORCH_AVAILABLE:
        return pd.Series(index=df.index, data=np.nan)
    
    window_size = 7
    if len(df) <= window_size:
        return pd.Series(index=df.index, data=np.nan)
    
    try:
        # Create missing columns if needed
        safe_df = safe_prepare_data(df, features)
        scaled_data = scaler.transform(imputer.transform(safe_df))
        
        predictions = pd.Series(index=df.index, data=np.nan, dtype=float)
        X_list = []
        indices = []
        
        for i in range(window_size, len(df)):
            seq = scaled_data[i - window_size:i]
            X_list.append(seq)
            indices.append(df.index[i])
        
        if not X_list:
            return predictions
        
        X_tensor = torch.FloatTensor(np.array(X_list))
        
        with torch.no_grad():
            model.eval()
            preds = model(X_tensor).numpy().flatten()
        
        for idx, val in zip(indices, preds):
            predictions.at[idx] = max(0, val)
        
        return predictions
    except Exception as e:
        logger.error(f"LSTM Prediction failed: {e}")
        return pd.Series(index=df.index, data=np.nan)


def detect_autoencoder_anomalies(
    model, 
    df: pd.DataFrame, 
    features: List[str], 
    scaler, 
    imputer,
    threshold: float
) -> pd.DataFrame:
    """
    Detect anomalies using autoencoder reconstruction error.
    
    Args:
        model: Trained autoencoder model
        df: DataFrame with features
        features: List of feature column names
        scaler: Fitted StandardScaler
        imputer: Fitted SimpleImputer
        threshold: Anomaly threshold
        
    Returns:
        DataFrame with reconstruction_error and is_ml_anomaly columns
    """
    if not TORCH_AVAILABLE:
        return pd.DataFrame()
    
    window_size = 7
    if len(df) <= window_size:
        return pd.DataFrame()
    
    try:
        # Create missing columns if needed
        safe_df = safe_prepare_data(df, features)
        scaled_data = scaler.transform(imputer.transform(safe_df))
        X_list = []
        indices = []
        
        for i in range(window_size, len(df) + 1):
            seq = scaled_data[i - window_size:i]
            X_list.append(seq)
            indices.append(df.index[i - 1])
        
        if not X_list:
            return pd.DataFrame()
        
        X_tensor = torch.FloatTensor(np.array(X_list))
        
        with torch.no_grad():
            model.eval()
            reconstructions = model(X_tensor)
            errors = torch.mean((X_tensor - reconstructions) ** 2, dim=[1, 2]).numpy()
        
        results = pd.DataFrame(index=indices)
        results['reconstruction_error'] = errors
        results['is_ml_anomaly'] = errors > threshold
        return results
        
    except Exception as e:
        logger.error(f"Autoencoder detection failed: {e}")
        return pd.DataFrame()
