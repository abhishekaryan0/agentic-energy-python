"""
Model Training Pipeline for Anomaly Detection
==============================================

Trains per-well anomaly detection models from Snowflake (well_sensor_readings).
Each well is trained on its own last PER_WELL_TRAINING_LIMIT (default 5000) rows.
Uses: Isolation Forest, statistical (mean/std), and LSTM autoencoder.
Saves under models/anomaly_detection/wells/<safe_well_id>/ with fixed filenames.
Requires Snowflake; no CSV fallback.

S3 upload: S3_BUCKET must be set in .env. After training, models are packed and
uploaded to S3 (required). Optional: S3_PREFIX (default models/anomaly), MODEL_VERSION (default v1).
"""

import os
import logging
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Optional: .env only needed for local runs (SageMaker training from parquet does not use it)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    _env_file = os.path.join(_project_root, ".env")
    if os.path.isfile(_env_file):
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available; LSTM training disabled")
    torch = None
    nn = None

# Configure logging (log file next to this module)
_log_dir = os.path.dirname(os.path.abspath(__file__))
_log_file = os.path.join(os.path.dirname(_log_dir), 'train_anomaly_models.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from anomaly_detection.detector import (
    CONTAMINATION_RATIO, SEQ_LEN, LSTM_EPOCHS, LSTM_LR, LSTMAutoencoder,
    well_id_to_safe,
)

# Load well types and features from sensor_config (database)
try:
    from config.sensor_config import get_lift_types, get_features_for_lift_type
    _lift_types = list(get_lift_types().keys())
    WELL_TYPES = _lift_types if _lift_types else ['Rod Pump', 'ESP', 'Gas Lift']
except ImportError:
    WELL_TYPES = ['Rod Pump', 'ESP', 'Gas Lift']
    get_features_for_lift_type = None

WELL_TYPE_FEATURES_FALLBACK = {
    'Rod Pump': ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure'],
    'ESP': ['motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage'],
    'Gas Lift': ['injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time'],
}

# LSTMAutoencoder imported from anomaly_detector

# ===== CONFIGURATION =====
MODELS_DIR = 'models'
# First subfolder under models/ for all anomaly detection artifacts
ANOMALY_DETECTION_SUBDIR = 'anomaly_detection'
# Base path: models/anomaly_detection/
ANOMALY_DETECTION_MODELS_DIR = os.path.join(MODELS_DIR, ANOMALY_DETECTION_SUBDIR)
# Per-well models under models/anomaly_detection/wells/<safe_well_id>/
WELLS_SUBDIR = 'wells'
# Fixed filenames inside each well folder (and well_types folder)
ISO_FILENAME = 'isolation_forest.joblib'
STAT_FILENAME = 'statistical_model.joblib'
LSTM_PT_FILENAME = 'lstm_autoencoder.pt'
LSTM_SCALER_FILENAME = 'lstm_scaler.joblib'
LSTM_META_FILENAME = 'lstm_metadata.joblib'
METRICS_FILENAME = 'metrics.json'
# Per-well: each well is trained on its own last N readings (Snowflake only)
PER_WELL_TRAINING_LIMIT = 5000

# Snowflake config
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

# ===== UTILITY FUNCTIONS =====

def create_models_dir(base_dir=None):
    """Ensure anomaly_detection/ and wells subdir exist. If base_dir set, use base_dir/anomaly_detection/wells/."""
    if base_dir is not None:
        models_root = os.path.join(base_dir, ANOMALY_DETECTION_SUBDIR)
    else:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            logger.info(f"Created {MODELS_DIR} directory")
        models_root = ANOMALY_DETECTION_MODELS_DIR
    if not os.path.exists(models_root):
        os.makedirs(models_root, exist_ok=True)
        logger.info(f"Created {models_root} directory")
    wells_dir = os.path.join(models_root, WELLS_SUBDIR)
    if not os.path.exists(wells_dir):
        os.makedirs(wells_dir)
        logger.info(f"Created {wells_dir} directory")


def get_well_model_dir(safe_well_id, base_dir=None):
    """Return path to a well's model folder. If base_dir set: base_dir/anomaly_detection/wells/<safe_well_id>/."""
    if base_dir is not None:
        return os.path.join(base_dir, ANOMALY_DETECTION_SUBDIR, WELLS_SUBDIR, safe_well_id)
    return os.path.join(ANOMALY_DETECTION_MODELS_DIR, WELLS_SUBDIR, safe_well_id)

def connect_snowflake():
    """Connect to Snowflake (used only for local Snowflake-based training)."""
    try:
        from snowflake.connector import connect as snowflake_connect
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        logger.info("Connected to Snowflake")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise

def _get_sensor_columns():
    """Get all sensor columns from sensor_config (DB) or fallback static list."""
    if get_features_for_lift_type:
        try:
            from config.sensor_config import get_all_sensor_columns
            return get_all_sensor_columns()
        except Exception:
            pass
    return [
        'well_id', 'timestamp', 'lift_type',
        'strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure',
        'motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage',
        'injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time',
        'surface_pressure', 'casing_pressure', 'wellhead_temp'
    ]


def get_well_ids_from_snowflake(conn=None):
    """Return list of distinct well_id from well_sensor_readings. Uses conn if provided."""
    own_conn = conn is None
    if own_conn:
        conn = connect_snowflake()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT well_id FROM well_sensor_readings ORDER BY well_id")
        well_ids = [row[0] for row in cursor.fetchall() if row[0]]
        cursor.close()
        return well_ids
    except Exception as e:
        logger.error(f"Failed to get well IDs from Snowflake: {e}")
        raise
    finally:
        if own_conn and conn:
            conn.close()


def load_training_data_for_well_from_snowflake(well_id, limit=None, conn=None):
    """
    Load one well's last N rows from Snowflake (by timestamp).
    Returns DataFrame ordered by timestamp for training. Uses conn if provided (caller must close).
    """
    if limit is None:
        limit = PER_WELL_TRAINING_LIMIT
    columns = _get_sensor_columns()
    base_cols = ['well_id', 'timestamp', 'lift_type']
    sensor_cols = [c for c in columns if c not in base_cols]
    own_conn = conn is None
    if own_conn:
        conn = connect_snowflake()
    try:
        cursor = conn.cursor()
        sel = ', '.join(base_cols + sensor_cols)
        query = f"""
            SELECT * FROM (
                SELECT {sel}
                FROM well_sensor_readings
                WHERE well_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            ) AS recent
            ORDER BY timestamp
        """
        cursor.execute(query, (well_id, limit))
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=base_cols + sensor_cols)
        cursor.close()
        return df
    except Exception as e:
        logger.error(f"Failed to load data for well {well_id}: {e}")
        raise
    finally:
        if own_conn and conn:
            conn.close()


def load_well_sensor_readings_from_parquet(path):
    """
    Load well_sensor_readings from a parquet file (local or S3).
    path: local path or s3://bucket/key.
    Returns DataFrame with well_id, timestamp, lift_type and sensor columns.
    """
    if path.startswith("s3://"):
        try:
            import boto3
            from urllib.parse import urlparse
            parsed = urlparse(path)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            s3 = boto3.client("s3")
            buf = __import__("io").BytesIO()
            s3.download_fileobj(bucket, key, buf)
            buf.seek(0)
            return pd.read_parquet(buf)
        except Exception as e:
            logger.error(f"Failed to load parquet from S3 {path}: {e}")
            raise
    return pd.read_parquet(path)


def get_well_ids_from_dataframe(df):
    """Return list of distinct well_id from a sensor DataFrame."""
    if "well_id" not in df.columns:
        return []
    return sorted(df["well_id"].dropna().astype(str).unique().tolist())


def load_training_data_for_well_from_df(well_id, full_df, limit=None):
    """
    Get one well's last N rows from a full sensor DataFrame (by timestamp).
    Returns DataFrame ordered by timestamp for training.
    """
    if limit is None:
        limit = PER_WELL_TRAINING_LIMIT
    well_df = full_df[full_df["well_id"].astype(str) == str(well_id)].copy()
    if "timestamp" in well_df.columns:
        well_df = well_df.sort_values("timestamp")
    well_df = well_df.tail(limit)
    if "timestamp" in well_df.columns:
        well_df = well_df.sort_values("timestamp").reset_index(drop=True)
    return well_df


def prepare_data_for_training(df, well_type):
    """
    Prepare data for model training. Does not mutate the input DataFrame.
    
    - Filter by well_type, select features for that type
    - Handle missing values (ffill, bfill, median, 0)
    - Build synthetic labels and train/test split
    
    Args:
        df: DataFrame with sensor data (not modified)
        well_type: Type of well (Rod Pump, ESP, Gas Lift)
        
    Returns:
        X_train, X_test, y_train, y_test, scaler, available_features or None
    """
    logger.info(f"Preparing data for {well_type} training...")
    
    # Filter for this well type
    if 'lift_type' in df.columns:
        df = df[df['lift_type'] == well_type].copy()
    
    logger.info(f"Records for {well_type}: {len(df)}")

    # Get features from sensor_definitions (DB) or fallback
    if get_features_for_lift_type:
        try:
            features = get_features_for_lift_type(well_type)
        except Exception:
            features = WELL_TYPE_FEATURES_FALLBACK.get(well_type, [])
    else:
        features = WELL_TYPE_FEATURES_FALLBACK.get(well_type, [])
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        logger.warning(f"No matching feature columns in data for {well_type}. DataFrame columns: {list(df.columns)[:15]}...")
        return None
    
    logger.info(f"Using features: {available_features}")
    
    df = df[available_features].copy()
    # Handle missing values: ffill/bfill, then fill remaining with column median
    df = df.ffill().bfill()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    # Last resort: fill any remaining NaN with 0
    df = df.fillna(0)
    
    logger.info(f"Records after cleaning: {len(df)}")
    
    if len(df) < 2:
        logger.error(f"Insufficient training records ({len(df)}) for {well_type}")
        return None
    
    if len(df) < 100:
        logger.warning(f"Very few training records ({len(df)}) for {well_type}")
    
    # Create synthetic anomaly labels (for supervised learning evaluation)
    # In real scenario, this would be actual anomaly labels
    # Here we mark statistical outliers as "anomalies"
    X = df[available_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Detect outliers using Isolation Forest (used for synthetic labels)
    iso_forest = IsolationForest(contamination=CONTAMINATION_RATIO, random_state=42)
    y_synthetic = iso_forest.fit_predict(X_scaled)
    y_synthetic = (y_synthetic == -1).astype(int)  # Convert to binary (0=normal, 1=anomaly)
    
    logger.info(f"Synthetic anomalies detected: {y_synthetic.sum()} ({100*y_synthetic.mean():.1f}%)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_synthetic, 
        test_size=0.2, 
        random_state=42,
        stratify=y_synthetic if len(np.unique(y_synthetic)) > 1 else None
    )
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler, available_features


def calculate_metrics(y_true, y_pred, scores=None):
    """Calculate evaluation metrics."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'true_positives': int(((y_pred == 1) & (y_true == 1)).sum()),
        'false_positives': int(((y_pred == 1) & (y_true == 0)).sum()),
        'false_negatives': int(((y_pred == 0) & (y_true == 1)).sum()),
        'true_negatives': int(((y_pred == 0) & (y_true == 0)).sum()),
    }
    if scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, scores)
        except Exception:
            metrics['auc_roc'] = 0.0
    return metrics


def train_isolation_forest(X_train, X_test, y_test, well_type, scaler):
    """
    Train Isolation Forest model for outlier detection.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_test: Test labels (synthetic anomalies)
        well_type: Type of well
        scaler: StandardScaler fitted on training data
        
    Returns:
        Trained model and metrics
    """
    logger.info(f"Training Isolation Forest for {well_type}...")
    
    # Scale training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=CONTAMINATION_RATIO,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    model.fit(X_train_scaled)
    
    # Evaluate
    y_pred_train = (model.predict(X_train_scaled) == -1).astype(int)
    y_pred_test = (model.predict(X_test_scaled) == -1).astype(int)
    
    # Get anomaly scores (lower = more anomalous)
    scores_test = model.score_samples(X_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred_test, scores_test)
    
    logger.info(f"Isolation Forest ({well_type}) - "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, "
                f"F1: {metrics['f1']:.3f}")
    
    return model, scaler, metrics

def train_statistical_model(X_train, X_test, y_test, well_type, features):
    """
    Train statistical model (mean/std based anomaly detection).
    
    Args:
        X_train: Training features
        X_test: Test features
        y_test: Test labels
        well_type: Type of well
        features: Feature names
        
    Returns:
        Statistical model and metrics
    """
    logger.info(f"Training Statistical Model for {well_type}...")
    
    # Calculate statistics from training data
    stats = {
        'mean': np.mean(X_train, axis=0),
        'std': np.std(X_train, axis=0),
        'min': np.min(X_train, axis=0),
        'max': np.max(X_train, axis=0),
        'features': features
    }
    
    # Detect anomalies: values beyond 3 standard deviations
    z_scores = np.abs((X_test - stats['mean']) / (stats['std'] + 1e-8))
    y_pred_test = (z_scores.max(axis=1) > 3).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred_test, z_scores.max(axis=1))
    
    logger.info(f"Statistical Model ({well_type}) - "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, "
                f"F1: {metrics['f1']:.3f}")
    
    return stats, metrics

def train_lstm_autoencoder(X_train, X_test, well_type, features):
    """
    Train LSTM Autoencoder for time-series anomaly detection.
    
    Args:
        X_train: Training sequences
        X_test: Test sequences
        well_type: Type of well
        features: Feature names
        
    Returns:
        Trained model dict and metrics
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available; skipping LSTM training")
        return None, {'error': 'PyTorch not available'}
    
    logger.info(f"Training LSTM Autoencoder for {well_type}...")
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create sequences
        sequences_train = [X_train_scaled[i:i+SEQ_LEN] for i in range(len(X_train_scaled) - SEQ_LEN)]
        sequences_test = [X_test_scaled[i:i+SEQ_LEN] for i in range(len(X_test_scaled) - SEQ_LEN)]
        
        if not sequences_train:
            logger.warning("Not enough training sequences for LSTM")
            return None, {'error': 'Insufficient data'}
        
        X_train_tensor = torch.FloatTensor(np.array(sequences_train))
        X_test_tensor = torch.FloatTensor(np.array(sequences_test))
        
        BATCH_SIZE = 64
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        input_dim = X_train_scaled.shape[1]
        model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        
        model.train()
        for epoch in range(LSTM_EPOCHS):
            total_loss = 0.0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            
            avg_loss = total_loss / len(train_dataset)
                
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{LSTM_EPOCHS}, Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_output = model(X_test_tensor)
            test_loss = criterion(test_output, X_test_tensor).item()
        
        metrics = {
            'test_mse': test_loss,
            'epochs': LSTM_EPOCHS,
            'seq_len': SEQ_LEN,
            'input_dim': input_dim
        }
        
        logger.info(f"LSTM Autoencoder ({well_type}) - Test MSE: {test_loss:.4f}")
        
        return {'model': model, 'scaler': scaler, 'seq_len': SEQ_LEN}, metrics
        
    except Exception as e:
        logger.error(f"Error training LSTM for {well_type}: {e}")
        return None, {'error': str(e)}


def save_model(model, filepath):
    """Save trained model using joblib."""
    try:
        joblib.dump(model, filepath)
        logger.info(f"Saved model to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def save_metrics(metrics, safe_well_id, well_dir=None):
    """Save model training metrics into well folder as metrics.json."""
    if well_dir is None:
        well_dir = get_well_model_dir(safe_well_id)
    filepath = os.path.join(well_dir, METRICS_FILENAME)
    try:
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def train_all_models():
    """
    Main training pipeline: train models per well.
    Prefers S3 parquet (sensor_data/well_sensor_readings.parquet) when available,
    falls back to Snowflake when parquet is not configured.
    """
    logger.info("=" * 60)
    logger.info(
        f"STARTING MODEL TRAINING PIPELINE (per well, last {PER_WELL_TRAINING_LIMIT} rows per well)"
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1) Prefer S3 parquet (same layout as digital twin / operations)
    # ------------------------------------------------------------------
    parquet_path = os.getenv("ANOMALY_SENSOR_PARQUET_S3", "").strip()
    if not parquet_path:
        bucket = os.getenv("S3_BUCKET", "").strip()
        org = os.getenv("ORGANIZATION_ID", "").strip()
        if bucket and org:
            parquet_path = f"s3://{bucket}/{org}/sensor_data/well_sensor_readings.parquet"

    if parquet_path:
        logger.info(
            "Using S3 parquet for anomaly_detection training: %s (S3-only, no Snowflake)",
            parquet_path,
        )
        # Train into local models/anomaly_detection/ (ANOMALY_DETECTION_MODELS_DIR)
        train_all_models_from_parquet(
            parquet_path,
            output_dir=MODELS_DIR,
            per_well_limit=PER_WELL_TRAINING_LIMIT,
        )
    else:
        # ------------------------------------------------------------------
        # 2) Legacy Snowflake path (kept for backward compatibility)
        # ------------------------------------------------------------------
        logger.info("ANOMALY_SENSOR_PARQUET_S3 not set and S3 paths not configured; falling back to Snowflake.")
        create_models_dir()

        try:
            conn = connect_snowflake()
        except Exception as e:
            logger.error("Snowflake connection required for legacy training path: %s", e)
            return

        try:
            well_ids = get_well_ids_from_snowflake(conn=conn)
        except Exception as e:
            logger.error("Failed to get well list from Snowflake: %s", e)
            conn.close()
            return

        if not well_ids:
            logger.error("No wells found in well_sensor_readings (Snowflake)")
            conn.close()
            return

        logger.info(
            f"Training per well from Snowflake: {len(well_ids)} wells (last {PER_WELL_TRAINING_LIMIT} rows each)"
        )

        all_metrics = {}

        for well_id in well_ids:
            try:
                well_df = load_training_data_for_well_from_snowflake(
                    well_id, limit=PER_WELL_TRAINING_LIMIT, conn=conn
                )
            except Exception as e:
                logger.warning("Skip well %s: %s", well_id, e)
                all_metrics[f"well {well_id}"] = {"error": str(e)}
                continue
            if well_df.empty or len(well_df) < 2:
                continue
            lift_type = (
                well_df["lift_type"].dropna().iloc[0]
                if "lift_type" in well_df.columns
                else None
            )
            if not lift_type or str(lift_type).strip() == "":
                continue
            safe_id = well_id_to_safe(
                well_id
            )  # lowercase for consistent folder/key with detector
            label = f"well {well_id} ({lift_type})"

            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {label}")
            logger.info(f"{'='*60}")

            try:
                prepared = prepare_data_for_training(well_df, lift_type)
                if prepared is None:
                    logger.warning(f"Skipping {label} - no training data")
                    all_metrics[label] = {"error": "No training data available"}
                    continue
                X_train, X_test, y_train, y_test, scaler, features = prepared

                iso_model, scaler_iso, iso_metrics = train_isolation_forest(
                    X_train, X_test, y_test, label, scaler
                )
                stat_model, stat_metrics = train_statistical_model(
                    X_train, X_test, y_test, label, features
                )
                lstm_model, lstm_metrics = train_lstm_autoencoder(
                    X_train, X_test, label, features
                )

                # Save under models/wells/<safe_id>/ with fixed filenames
                well_dir = get_well_model_dir(safe_id)
                os.makedirs(well_dir, exist_ok=True)
                save_model(
                    {"model": iso_model, "scaler": scaler_iso, "features": features},
                    os.path.join(well_dir, ISO_FILENAME),
                )
                save_model(
                    stat_model,
                    os.path.join(well_dir, STAT_FILENAME),
                )
                if lstm_model:
                    torch.save(
                        lstm_model["model"].state_dict(),
                        os.path.join(well_dir, LSTM_PT_FILENAME),
                    )
                    joblib.dump(
                        lstm_model["scaler"],
                        os.path.join(well_dir, LSTM_SCALER_FILENAME),
                    )
                    meta = {
                        "input_dim": lstm_model["model"].input_dim,
                        "features": features,
                        "seq_len": SEQ_LEN,
                    }
                    joblib.dump(
                        meta, os.path.join(well_dir, LSTM_META_FILENAME)
                    )
                    logger.info(f"Saved LSTM model for {label}")

                all_metrics[label] = {
                    "isolation_forest": iso_metrics,
                    "statistical": stat_metrics,
                    "lstm": lstm_metrics
                    if lstm_model
                    else {"error": "Not trained"},
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": features,
                    "well_id": str(well_id),
                    "lift_type": str(lift_type),
                    "timestamp": datetime.now().isoformat(),
                }
                save_metrics(all_metrics[label], safe_id, well_dir=well_dir)

            except Exception as e:
                logger.error(f"Error training {label}: {e}", exc_info=True)
                all_metrics[label] = {"error": str(e)}

        try:
            conn.close()
        except Exception:
            pass

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")

        for label, metrics in all_metrics.items():
            logger.info(f"\n{label}:")
            if "error" in metrics:
                logger.info(f"  ❌ Error: {metrics['error']}")
            else:
                logger.info(f"  ✓ Training samples: {metrics['training_samples']}")
                logger.info(f"  ✓ Test samples: {metrics['test_samples']}")
                logger.info(f"  ✓ Features: {', '.join(metrics['features'])}")
                logger.info(
                    f"  ✓ Isolation Forest - F1: {metrics['isolation_forest']['f1']:.3f}"
                )
                logger.info(
                    f"  ✓ Statistical Model - F1: {metrics['statistical']['f1']:.3f}"
                )
                if "lstm" in metrics and metrics["lstm"] and "test_mse" in metrics["lstm"]:
                    logger.info(
                        f"  ✓ LSTM Autoencoder - Test MSE: {metrics['lstm']['test_mse']:.4f}"
                    )
                else:
                    logger.info("  ✓ LSTM Autoencoder - Not trained")

        summary_file = os.path.join(
            ANOMALY_DETECTION_MODELS_DIR, "training_summary.json"
        )

        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        with open(summary_file, "w") as f:
            json.dump(convert_types(all_metrics), f, indent=2)
        logger.info(f"\nSaved training summary to {summary_file}")

    # ------------------------------------------------------------------
    # Upload trained models to S3 for inference (same as before)
    # ------------------------------------------------------------------
    s3_bucket = os.environ.get("S3_BUCKET")
    if not s3_bucket:
        logger.error(
            "S3_BUCKET is required. Set S3_BUCKET in .env (e.g. S3_BUCKET=your-models-bucket)"
        )
        sys.exit(1)
    if not _upload_anomaly_models_to_s3(s3_bucket):
        logger.error(
            "S3 upload failed. Training saved locally but deploy will not have the latest models."
        )
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - Per-well models saved and uploaded to S3")
    logger.info("=" * 60)


def train_all_models_from_parquet(parquet_path, output_dir, per_well_limit=None):
    """
    Train per-well anomaly models from a parquet file (local or s3://), write to output_dir.
    Used by SageMaker: data at parquet_path, artifacts under output_dir (e.g. /opt/ml/model).
    output_dir will contain anomaly_detection/wells/<safe_well_id>/ and training_summary.json.
    """
    if per_well_limit is None:
        per_well_limit = PER_WELL_TRAINING_LIMIT
    logger.info("="*60)
    logger.info(f"TRAINING FROM PARQUET: {parquet_path} -> {output_dir} (last {per_well_limit} rows/well)")
    logger.info("="*60)

    create_models_dir(base_dir=output_dir)

    full_df = load_well_sensor_readings_from_parquet(parquet_path)
    logger.info(f"Loaded {len(full_df):,} rows from parquet")

    well_ids = get_well_ids_from_dataframe(full_df)
    if not well_ids:
        logger.error("No well_id column or no wells in parquet")
        return

    logger.info(f"Training per well: {len(well_ids)} wells")

    all_metrics = {}
    for well_id in well_ids:
        well_df = load_training_data_for_well_from_df(well_id, full_df, limit=per_well_limit)
        if well_df.empty or len(well_df) < 2:
            continue
        lift_type = well_df["lift_type"].dropna().iloc[0] if "lift_type" in well_df.columns else None
        if not lift_type or str(lift_type).strip() == "":
            continue
        safe_id = well_id_to_safe(well_id)
        label = f"well {well_id} ({lift_type})"

        logger.info(f"\n{'='*60}\nTraining models for {label}\n{'='*60}")

        try:
            prepared = prepare_data_for_training(well_df, lift_type)
            if prepared is None:
                all_metrics[label] = {"error": "No training data available"}
                continue
            X_train, X_test, y_train, y_test, scaler, features = prepared

            iso_model, scaler_iso, iso_metrics = train_isolation_forest(
                X_train, X_test, y_test, label, scaler
            )
            stat_model, stat_metrics = train_statistical_model(
                X_train, X_test, y_test, label, features
            )
            lstm_model, lstm_metrics = train_lstm_autoencoder(
                X_train, X_test, label, features
            )

            well_dir = get_well_model_dir(safe_id, base_dir=output_dir)
            os.makedirs(well_dir, exist_ok=True)
            save_model(
                {"model": iso_model, "scaler": scaler_iso, "features": features},
                os.path.join(well_dir, ISO_FILENAME),
            )
            save_model(stat_model, os.path.join(well_dir, STAT_FILENAME))
            if lstm_model:
                torch.save(lstm_model["model"].state_dict(), os.path.join(well_dir, LSTM_PT_FILENAME))
                joblib.dump(lstm_model["scaler"], os.path.join(well_dir, LSTM_SCALER_FILENAME))
                meta = {"input_dim": lstm_model["model"].input_dim, "features": features, "seq_len": SEQ_LEN}
                joblib.dump(meta, os.path.join(well_dir, LSTM_META_FILENAME))

            all_metrics[label] = {
                "isolation_forest": iso_metrics,
                "statistical": stat_metrics,
                "lstm": lstm_metrics if lstm_model else {"error": "Not trained"},
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features": features,
                "well_id": str(well_id),
                "lift_type": str(lift_type),
                "timestamp": datetime.now().isoformat(),
            }
            save_metrics(all_metrics[label], safe_id, well_dir=well_dir)
        except Exception as e:
            logger.error(f"Error training {label}: {e}", exc_info=True)
            all_metrics[label] = {"error": str(e)}

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        return obj

    summary_dir = os.path.join(output_dir, ANOMALY_DETECTION_SUBDIR)
    summary_file = os.path.join(summary_dir, "training_summary.json")
    with open(summary_file, "w") as f:
        json.dump(_convert(all_metrics), f, indent=2)
    logger.info(f"Saved training summary to {summary_file}")
    logger.info("TRAINING FROM PARQUET COMPLETE - artifacts in %s", output_dir)


def _upload_anomaly_models_to_s3(bucket: str) -> bool:
    """
    Pack models/anomaly_detection/ and upload to S3 using the existing pack script.
    Uses env S3_PREFIX (default models/anomaly) and MODEL_VERSION (default v1).
    Returns True if upload succeeded, False otherwise.
    """
    prefix = os.environ.get("S3_PREFIX", "models/anomaly").rstrip("/")
    version = os.environ.get("MODEL_VERSION", "v1")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(project_root, "scripts", "pack_and_upload_sagemaker_model.py")
    if not os.path.isfile(script):
        logger.error("Pack script not found: %s", script)
        return False
    cmd = [
        sys.executable,
        script,
        "--bucket", bucket,
        "--prefix", prefix,
        "--version", version,
    ]
    logger.info("Uploading anomaly detection models to S3: %s (prefix=%s, version=%s)", bucket, prefix, version)
    try:
        r = subprocess.run(cmd, cwd=project_root)
        if r.returncode != 0:
            logger.error("S3 upload failed (exit code %s)", r.returncode)
            return False
        logger.info("Models uploaded to s3://%s/%s/anomaly_detection_%s.tar.gz", bucket, prefix, version)
        return True
    except Exception as e:
        logger.error("Failed to run pack/upload script: %s", e)
        return False


if __name__ == '__main__':
    train_all_models()
