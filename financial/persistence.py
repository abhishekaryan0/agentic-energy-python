"""
Load and run the financial model from a directory (e.g. /opt/ml/model).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import joblib
except Exception:
    joblib = None


def load_financial_model(model_dir: str, filename: str = "financial_implementation_cost.joblib") -> Dict[str, Any]:
    if joblib is None:
        raise RuntimeError("joblib not available")
    p = Path(model_dir) / filename
    if not p.exists():
        raise FileNotFoundError(str(p))
    obj = joblib.load(str(p))
    if not isinstance(obj, dict) or "model" not in obj or "features" not in obj:
        raise ValueError("Invalid financial model artifact")
    return obj


def predict_implementation_cost_usd_per_day(
    model_obj: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> float:
    """
    Predict implementation cost ($/day) from a feature dict.
    Missing features -> 0.0 (safe default).
    """
    clf = model_obj["model"]
    cols: List[str] = list(model_obj.get("features") or [])
    x = []
    for c in cols:
        v = feature_values.get(c)
        try:
            x.append(float(v) if v is not None else 0.0)
        except Exception:
            x.append(0.0)
    pred = clf.predict([np.array(x, dtype=float)])[0]
    try:
        return float(pred)
    except Exception:
        return 0.0

