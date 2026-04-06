"""
Financial module: train and run models using invoice (financial ledger) data.

Goal (initial version):
- Train a model that predicts an estimated implementation cost in USD/day for a well,
  using operational signals (production + sensor metrics) and recent invoice patterns.
"""

from financial.train_models import train_financial_from_parquet
from financial.persistence import (
    load_financial_model,
    predict_implementation_cost_usd_per_day,
)

__all__ = [
    "train_financial_from_parquet",
    "load_financial_model",
    "predict_implementation_cost_usd_per_day",
]

