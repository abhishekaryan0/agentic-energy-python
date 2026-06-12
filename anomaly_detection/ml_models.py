"""
Shared ML helpers: PyOD ensemble, MAD statistics, score normalization.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ISO_N_ESTIMATORS = int(os.getenv("ANOMALY_ISO_N_ESTIMATORS", "200") or 200)
ISO_CONTAMINATION = float(os.getenv("ANOMALY_ISO_CONTAMINATION", "0.05") or 0.05)
PYOD_CONTAMINATION = float(os.getenv("ANOMALY_PYOD_CONTAMINATION", "0.05") or 0.05)
MAD_SCALE = 1.4826

try:
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD

    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    ECOD = None
    COPOD = None


def pyod_available() -> bool:
    return PYOD_AVAILABLE


def fit_pyod_ensemble(X_train: np.ndarray) -> Optional[Dict[str, Any]]:
    if not PYOD_AVAILABLE:
        logger.warning("PyOD not installed; skipping ECOD/COPOD training")
        return None
    ecod = ECOD(contamination=PYOD_CONTAMINATION)
    copod = COPOD(contamination=PYOD_CONTAMINATION)
    ecod.fit(X_train)
    copod.fit(X_train)
    return {"ecod": ecod, "copod": copod}


def pyod_anomaly_scores(X: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    """Per-row combined ECOD+COPOD score in [0, 1] (higher = more anomalous)."""
    ecod = bundle["ecod"]
    copod = bundle["copod"]
    ecod_raw = ecod.decision_function(X)
    copod_raw = copod.decision_function(X)
    ecod_norm = _normalize_scores(ecod_raw)
    copod_norm = _normalize_scores(copod_raw)
    return (ecod_norm + copod_norm) / 2.0


def _normalize_scores(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = np.min(arr), np.max(arr)
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def build_robust_stats(X_train: np.ndarray, features: List[str]) -> Dict[str, Any]:
    median = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - median), axis=0)
    return {
        "median": median,
        "mad": mad,
        "mean": np.mean(X_train, axis=0),
        "std": np.std(X_train, axis=0),
        "min": np.min(X_train, axis=0),
        "max": np.max(X_train, axis=0),
        "robust_method": "mad",
        "features": features,
    }


def mad_z_scores(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    if stats.get("robust_method") == "mad" and "median" in stats and "mad" in stats:
        median = np.asarray(stats["median"], dtype=float)
        mad = np.asarray(stats["mad"], dtype=float)
        denom = MAD_SCALE * mad + 1e-8
        return np.abs((values - median) / denom)
    mean = np.asarray(stats["mean"], dtype=float)
    std = np.asarray(stats["std"], dtype=float)
    return np.abs((values - mean) / (std + 1e-8))


def mad_anomaly_score(
    values: np.ndarray,
    stats: Dict[str, Any],
    *,
    z_threshold: float,
) -> float:
    z = mad_z_scores(values, stats)
    return float(min(1.0, np.max(z) / max(z_threshold, 1e-8)))
