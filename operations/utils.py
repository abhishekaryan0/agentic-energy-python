"""
Shared utilities for the operations module.

This module contains common functions used across the operations package,
including path resolution, database operations, and type conversion.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# PATH RESOLUTION
# ============================================================================

# Project root for consistent path resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / 'models'
MODELS_DIR = _DEFAULT_MODELS_DIR  # backward compat; use get_models_dir() for env override

# One-time sync of operations models from S3 (bucket + ORGANIZATION_ID from env)
_OPERATIONS_MODELS_S3_SYNCED = False


def _ensure_operations_models_from_s3() -> None:
    """
    If S3_BUCKET and ORGANIZATION_ID are set, download the latest operations-train-* model
    from s3://BUCKET/{ORGANIZATION_ID}/model/operations-train-*/output/model.tar.gz,
    extract to a local cache, and set OPERATIONS_MODELS_DIR. Uses ORGANIZATION_ID from env.
    Only runs once per process.
    """
    global _OPERATIONS_MODELS_S3_SYNCED
    if _OPERATIONS_MODELS_S3_SYNCED:
        return
    try:
        from operations.config import get_operations_model_s3_base
        base = get_operations_model_s3_base()
    except Exception:
        base = None
    if not base:
        return
    try:
        import tarfile
        import tempfile
        from urllib.parse import urlparse
        import boto3
        s3 = boto3.client("s3")
        parsed = urlparse(base)
        bucket = parsed.netloc
        prefix = (parsed.path or "").lstrip("/")
        if not prefix:
            prefix = ""
        elif not prefix.endswith("/"):
            prefix = prefix + "/"
        if base.rstrip("/").endswith("model.tar.gz"):
            key = parsed.path.lstrip("/")
        else:
            paginator = s3.get_paginator("list_objects_v2")
            candidates = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    k = obj["Key"]
                    if "operations-train-" in k and k.endswith("output/model.tar.gz"):
                        candidates.append(k)
            if not candidates:
                logger.warning("No operations-train-*/output/model.tar.gz under s3://%s/%s", bucket, prefix)
                return
            candidates.sort(reverse=True)
            key = candidates[0]
        logger.info("Downloading operations model from s3://%s/%s", bucket, key)
        cache_dir = os.environ.get("OPERATIONS_MODEL_CACHE_DIR") or tempfile.mkdtemp(prefix="operations_models_")
        os.makedirs(cache_dir, exist_ok=True)
        local_tar = os.path.join(cache_dir, "model.tar.gz")
        s3.download_file(bucket, key, local_tar)
        with tarfile.open(local_tar, "r:gz") as tf:
            tf.extractall(cache_dir)
        try:
            os.remove(local_tar)
        except OSError:
            pass
        os.environ["OPERATIONS_MODELS_DIR"] = cache_dir
        _OPERATIONS_MODELS_S3_SYNCED = True
        logger.info("Operations models loaded from S3 -> %s (ORGANIZATION_ID from env)", cache_dir)
    except Exception as e:
        logger.warning("Failed to sync operations models from S3: %s", e)


def get_project_root() -> Path:
    """Get the project root directory."""
    return _PROJECT_ROOT


def get_models_dir() -> Path:
    """Get the models directory. Syncs from S3 if S3_BUCKET and ORGANIZATION_ID set."""
    _ensure_operations_models_from_s3()
    return Path(os.environ.get("OPERATIONS_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))


def get_model_path(relative_path: str) -> str:
    """
    Convert a relative model path to an absolute path. Uses get_models_dir() so OPERATIONS_MODELS_DIR is respected.
    
    Args:
        relative_path: Path to the model file (can be relative or absolute)
        
    Returns:
        Absolute path to the model file
    """
    p = Path(relative_path)
    if p.is_absolute():
        return str(p)
    models_dir = get_models_dir()
    if relative_path.startswith('models/'):
        return str(models_dir / relative_path.replace('models/', '', 1))
    return str(models_dir / p.name)


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

T = TypeVar('T')


def retry_db_operation(
    func: Callable[[], T],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """
    Retry a database operation with exponential backoff.
    
    Args:
        func: Callable that performs the database operation
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception if all retries fail
        
    Example:
        >>> def save_record():
        ...     cursor.execute("INSERT INTO ...")
        >>> retry_db_operation(save_record, max_retries=3)
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (backoff_factor ** attempt)
                time.sleep(wait_time)
            else:
                pass
    
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected error in retry_db_operation")


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_native_type(value: Any) -> Any:
    """
    Convert numpy/pandas types and complex objects to plain Python types.
    
    Useful for JSON serialization and database operations.
    
    Args:
        value: Any value to convert
        
    Returns:
        Python native type equivalent
        
    Example:
        >>> to_native_type(np.float64(3.14))
        3.14
        >>> to_native_type({'a': np.int64(1)})
        {'a': 1}
    """
    # numpy scalars
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.str_,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    # datetime handling
    from datetime import datetime
    if isinstance(value, datetime):
        return value
    
    # None and basic types
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    
    # Containers - recursively convert
    if isinstance(value, dict):
        return {k: to_native_type(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native_type(v) for v in value]
    
    # Try JSON serialization for other objects
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def truncate_string(s: Any, max_length: int) -> Optional[str]:
    """
    Truncate a string to a maximum length.
    
    Args:
        s: String or value to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string or None if input is None
    """
    if s is None:
        return None
    try:
        s_str = str(s)
        return s_str if len(s_str) <= max_length else s_str[:max_length]
    except Exception:
        return None


# ============================================================================
# FEATURE UTILITIES
# ============================================================================

def features_from_metrics(
    metrics: Dict[str, Any],
    feature_list: Optional[List[str]] = None,
    default_value: float = 0.0
) -> tuple:
    """
    Convert a metrics dict to a flat numeric feature vector.
    
    Args:
        metrics: Dict of computed metrics
        feature_list: Explicit list of features to extract
        default_value: Default value for missing features
        
    Returns:
        Tuple of (feature_array, feature_names)
    """
    if feature_list is None:
        # Use all available metrics that look like features
        feature_list = [
            k for k in metrics.keys()
            if k.endswith('_slope') or k.endswith('_roll_mean') or k == 'missing_rate'
        ]
        # Add correlation features if present
        feature_list.extend([k for k in metrics.keys() if k.endswith('_corr')])
        # Remove duplicates while preserving order
        seen = set()
        feature_list = [k for k in feature_list if not (k in seen or seen.add(k))]
    
    features = []
    for k in feature_list:
        v = metrics.get(k)
        features.append(safe_float(v, default_value))
    
    return np.array(features), feature_list
