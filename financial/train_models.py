#!/usr/bin/env python3
"""
Train financial models from S3/local parquet (invoice + production schemas aligned with scripts/generate_*).

Inputs:
- sensor parquet: well_sensor_readings.parquet (optional; not required for first version)
- production parquet: daily_production.parquet (required)
- invoices parquet(s): invoices_v1.parquet and/or invoices_v2.parquet (required)

Model output (joblib dict):
- model: scikit-learn regressor
- features: list[str]
- target: "implementation_cost_usd_per_day"
- metadata: training stats

This is designed to run inside SageMaker training:
- reads from SM_CHANNEL_SENSOR_DATA and SM_CHANNEL_FINANCIAL_DATA
- writes to SM_MODEL_DIR (/opt/ml/model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    SKLEARN_OK = True
except Exception:
    joblib = None
    RandomForestRegressor = None
    mean_absolute_error = None
    r2_score = None
    SKLEARN_OK = False


@dataclass
class FinancialTrainingConfig:
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_leaf: int = 2
    n_jobs: int = -1

    # How to derive the supervised label from invoices:
    # label = sum(workover_like_invoices over previous window_days)/window_days
    # Using a longer window reduces "mostly zero" labels while keeping the meaning
    # of implementation cost as a workover/repair-driven one-time intervention.
    window_days: int = 365

    # What invoice totals should drive the supervised label for implementation cost.
    # - "workover_like": only invoices whose `category` looks like workover/repair.
    # - "all_invoices": all invoices (including recurring LOE) drive the label.
    # Using "all_invoices" avoids the model predicting 0 when a well only has LOE activity.
    implementation_invoice_mode: str = "workover_like"


DEFAULT_FINANCIAL_TRAINING_CONFIG = FinancialTrainingConfig()


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _normalize_daily_production(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Support both twin schema and canonical schema
    if "well_id" not in out.columns and "asset_id" in out.columns:
        out["well_id"] = out["asset_id"].astype(str)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date
    # Coerce production numeric fields if present
    for c in ("true_oil", "true_gas", "true_water", "oil_volume", "gas_volume", "water_volume", "boe"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "status" in out.columns:
        out["status"] = out["status"].astype(str)
    if "lift_type" in out.columns:
        out["lift_type"] = out["lift_type"].astype(str)
    return out


def _normalize_invoices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "asset_id" not in out.columns and "well_id" in out.columns:
        out["asset_id"] = out["well_id"].astype(str)
    if "invoice_date" in out.columns:
        out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.date
    if "total_usd" in out.columns:
        out["total_usd"] = pd.to_numeric(out["total_usd"], errors="coerce")
    if "category" in out.columns:
        out["category"] = out["category"].astype(str)
    if "invoice_version" in out.columns:
        out["invoice_version"] = out["invoice_version"].astype(str)
    return out


def _is_workover_like(category: Optional[str]) -> bool:
    if not category:
        return False
    s = str(category).lower()
    return any(k in s for k in ("workover", "work over", "repair", "work_over"))


def build_financial_training_frame(
    daily_production_df: pd.DataFrame,
    invoices_frames: List[pd.DataFrame],
    cfg: FinancialTrainingConfig = DEFAULT_FINANCIAL_TRAINING_CONFIG,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a supervised training dataset at (well_id, date) granularity.

    Features:
    - current production signals (oil/gas/water) if present
    - recent invoice totals (sum over past window_days) overall and workover-like
    - invoice counts over past window_days
    - optional status flags (shut-in)

    Label (target):
    - implementation_cost_usd_per_day: (workover_like_invoice_total over past window_days) / window_days
      This is a proxy label that makes the model learn expected maintenance/workover cost pressure.
    """
    prod = _normalize_daily_production(daily_production_df)
    inv_all = pd.concat([_normalize_invoices(x) for x in invoices_frames if x is not None], ignore_index=True)
    if inv_all.empty:
        raise ValueError("No invoice rows provided to build_financial_training_frame")

    inv_all = inv_all.rename(columns={"asset_id": "well_id"})
    inv_all["well_id"] = inv_all["well_id"].astype(str)

    # Aggregate invoices per (well_id, invoice_date)
    inv_all["is_workover_like"] = inv_all["category"].map(_is_workover_like)
    inv_day = (
        inv_all.groupby(["well_id", "invoice_date"], as_index=False)
        .agg(
            invoice_total_usd=("total_usd", "sum"),
            invoice_count=("invoice_id", "count" if "invoice_id" in inv_all.columns else "size"),
            workover_total_usd=("total_usd", lambda s: float(np.nansum(s[inv_all.loc[s.index, "is_workover_like"]]))),
            workover_count=("is_workover_like", "sum"),
        )
    )
    inv_day = inv_day.rename(columns={"invoice_date": "date"})
    inv_day["date"] = pd.to_datetime(inv_day["date"]).dt.date

    # Join invoices onto daily production days (left join; missing -> NaN -> filled to 0 for features)
    base = prod.copy()
    base = base[["well_id", "date"] + [c for c in base.columns if c in ("true_oil", "true_gas", "true_water", "oil_volume", "gas_volume", "water_volume", "boe", "status", "lift_type")]]
    base = base.dropna(subset=["well_id", "date"])
    base["well_id"] = base["well_id"].astype(str)

    merged = base.merge(inv_day, on=["well_id", "date"], how="left")

    # Rolling features per well
    window = int(cfg.window_days)
    merged = merged.sort_values(["well_id", "date"])
    for col in ("invoice_total_usd", "invoice_count", "workover_total_usd", "workover_count"):
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        merged[f"{col}_roll{window}d"] = (
            merged.groupby("well_id")[col]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Status feature
    merged["is_shut_in"] = merged.get("status", "").astype(str).str.upper().eq("SHUT_IN").astype(float)

    # Build label from rolling invoice totals.
    # This is the only "business-rule" that determines what implementation cost represents.
    mode = (cfg.implementation_invoice_mode or "").strip().lower()
    if mode in ("workover_like", "workover", "repair"):
        merged["implementation_cost_usd_per_day"] = merged[f"workover_total_usd_roll{window}d"] / float(window)
    elif mode in ("all_invoices", "all", "total"):
        merged["implementation_cost_usd_per_day"] = merged[f"invoice_total_usd_roll{window}d"] / float(window)
    else:
        raise ValueError(f"Unknown implementation_invoice_mode: {cfg.implementation_invoice_mode!r}")

    # Feature set
    feature_cols: List[str] = []
    for c in ("true_oil", "true_gas", "true_water", "oil_volume", "gas_volume", "water_volume", "boe"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
            feature_cols.append(c)

    feature_cols.extend(
        [
            "is_shut_in",
            f"invoice_total_usd_roll{window}d",
            f"invoice_count_roll{window}d",
            f"workover_total_usd_roll{window}d",
            f"workover_count_roll{window}d",
        ]
    )

    # Ensure all features exist
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0

    return merged, feature_cols


def train_financial_from_parquet(
    production_parquet_path: str,
    invoices_parquet_paths: List[str],
    output_dir: str,
    cfg: FinancialTrainingConfig = DEFAULT_FINANCIAL_TRAINING_CONFIG,
    model_filename: str = "financial_implementation_cost.joblib",
) -> Dict[str, object]:
    if not SKLEARN_OK or joblib is None:
        raise RuntimeError("scikit-learn/joblib not available")

    prod_df = _read_parquet(production_parquet_path)
    inv_frames = [_read_parquet(p) for p in invoices_parquet_paths]

    frame, feature_cols = build_financial_training_frame(prod_df, inv_frames, cfg=cfg)
    y = frame["implementation_cost_usd_per_day"].astype(float).values
    X = frame[feature_cols].astype(float).values

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    model.fit(X, y)

    pred = model.predict(X)
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))

    out_obj = {
        "model": model,
        "features": feature_cols,
        "target": "implementation_cost_usd_per_day",
        "window_days": cfg.window_days,
        "implementation_invoice_mode": cfg.implementation_invoice_mode,
        "metrics": {
            "mae": mae,
            "r2": r2,
            "n_samples": int(len(y)),
        },
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = str(Path(output_dir) / model_filename)
    joblib.dump(out_obj, model_path)
    return {"model_path": model_path, **out_obj["metrics"]}

