"""
Model Training Pipeline for Anomaly Detection
==============================================

Trains per-well anomaly detection models from org sensor data on S3.

Input layout (default):
  s3://{S3_BUCKET}/{ORGANIZATION_ID}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet

Legacy single-file fallback:
  s3://{S3_BUCKET}/{ORGANIZATION_ID}/sensor_data/well_sensor_readings.parquet

Each well is trained on its own last PER_WELL_TRAINING_LIMIT (default 5000) rows.
Uses: Isolation Forest, statistical (mean/std), and LSTM autoencoder.
Saves under models/anomaly_detection/wells/<safe_well_id>/ with fixed filenames.

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

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "PyTorch is required for anomaly detection training (LSTM autoencoder). "
        "Install with: pip install torch>=2.0.0"
    ) from e

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
from anomaly_detection.sensor_data import (
    load_well_sensor_readings_from_parquet,
    resolve_training_sensor_data,
)

# Load well types and features from sensor_config (S3 parquet) or fallback
try:
    from anomaly_detection.sensor_config import get_lift_types, get_features_for_lift_type
    _lift_types = list(get_lift_types().keys())
    WELL_TYPES = _lift_types if _lift_types else ['Rod Pump', 'ESP', 'Gas Lift']
except ImportError:
    WELL_TYPES = ['Rod Pump', 'ESP', 'Gas Lift']
    get_features_for_lift_type = None

WELL_TYPE_FEATURES_FALLBACK = {
    'Rod Pump': ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure'],
    'Pumping Unit': ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure'],
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
# Per-well: each well is trained on its own last N readings from S3 parquet
PER_WELL_TRAINING_LIMIT = 5000

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


def _get_sensor_columns():
    """Get all sensor columns from sensor_config (DB) or fallback static list."""
    if get_features_for_lift_type:
        try:
            from anomaly_detection.sensor_config import get_all_sensor_columns
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
    from anomaly_detection.detector import normalize_lift_type_for_models

    feature_lift_type = normalize_lift_type_for_models(well_type)
    logger.info(f"Preparing data for {well_type} training (features from {feature_lift_type})...")

    logger.info(f"Records for {well_type}: {len(df)}")

    # Get features from sensor_definitions (DB) or fallback
    if get_features_for_lift_type:
        try:
            features = get_features_for_lift_type(feature_lift_type)
        except Exception:
            features = WELL_TYPE_FEATURES_FALLBACK.get(feature_lift_type, [])
    else:
        features = WELL_TYPE_FEATURES_FALLBACK.get(
            feature_lift_type, WELL_TYPE_FEATURES_FALLBACK.get(well_type, [])
        )
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
    logger.info(f"Training LSTM Autoencoder for {well_type}...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sequences_train = [X_train_scaled[i:i + SEQ_LEN] for i in range(len(X_train_scaled) - SEQ_LEN)]
    sequences_test = [X_test_scaled[i:i + SEQ_LEN] for i in range(len(X_test_scaled) - SEQ_LEN)]

    if not sequences_train:
        raise ValueError(
            f"Not enough rows for LSTM sequences (need > {SEQ_LEN}, got {len(X_train_scaled)})"
        )

    X_train_tensor = torch.FloatTensor(np.array(sequences_train))
    X_test_tensor = torch.FloatTensor(np.array(sequences_test))

    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    model.eval()
    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_loss = criterion(test_output, X_test_tensor).item()

    metrics = {
        "test_mse": test_loss,
        "epochs": LSTM_EPOCHS,
        "seq_len": SEQ_LEN,
        "input_dim": input_dim,
    }
    logger.info(f"LSTM Autoencoder ({well_type}) - Test MSE: {test_loss:.4f}")
    return {"model": model, "scaler": scaler, "seq_len": SEQ_LEN}, metrics


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
    Main training pipeline: train models per well from S3 sensor parquet.
    Requires S3_BUCKET + ORGANIZATION_ID or ANOMALY_SENSOR_PARQUET_S3.
    """
    logger.info("=" * 60)
    logger.info(
        f"STARTING MODEL TRAINING PIPELINE (per well, last {PER_WELL_TRAINING_LIMIT} rows per well)"
    )
    logger.info("=" * 60)

    try:
        full_df = resolve_training_sensor_data()
    except (ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        sys.exit(1)

    train_all_models_from_dataframe(
        full_df,
        output_dir=MODELS_DIR,
        per_well_limit=PER_WELL_TRAINING_LIMIT,
    )

    # Upload trained models to S3 for inference
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


def train_all_models_from_dataframe(full_df, output_dir, per_well_limit=None):
    """
    Train per-well anomaly models from a sensor DataFrame; write to output_dir.
    output_dir will contain anomaly_detection/wells/<safe_well_id>/ and training_summary.json.
    """
    if per_well_limit is None:
        per_well_limit = PER_WELL_TRAINING_LIMIT
    logger.info("=" * 60)
    logger.info(
        "TRAINING FROM SENSOR DATA -> %s (last %s rows/well, %s total rows)",
        output_dir,
        per_well_limit,
        len(full_df),
    )
    logger.info("=" * 60)

    create_models_dir(base_dir=output_dir)

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
            torch.save(lstm_model["model"].state_dict(), os.path.join(well_dir, LSTM_PT_FILENAME))
            joblib.dump(lstm_model["scaler"], os.path.join(well_dir, LSTM_SCALER_FILENAME))
            meta = {
                "input_dim": lstm_model["model"].input_dim,
                "features": features,
                "seq_len": SEQ_LEN,
            }
            joblib.dump(meta, os.path.join(well_dir, LSTM_META_FILENAME))

            all_metrics[label] = {
                "isolation_forest": iso_metrics,
                "statistical": stat_metrics,
                "lstm": lstm_metrics,
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
    logger.info("TRAINING COMPLETE - artifacts in %s", output_dir)


def train_all_models_from_parquet(parquet_path, output_dir, per_well_limit=None):
    """
    Train per-well anomaly models from a single parquet file (local or s3://).
    Used when a specific file path is provided; prefer train_all_models_from_dataframe
    or resolve_training_sensor_data() for org partition layout.
    """
    logger.info("Loading training parquet: %s", parquet_path)
    full_df = load_well_sensor_readings_from_parquet(parquet_path)
    logger.info("Loaded %s rows from parquet", len(full_df))
    train_all_models_from_dataframe(full_df, output_dir, per_well_limit=per_well_limit)


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
