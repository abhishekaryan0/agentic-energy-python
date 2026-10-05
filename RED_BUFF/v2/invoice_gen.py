"""Invoices and revenue for RED_BUFF V2 (like v1 invoice_gen)."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import List, Optional

_V2_DIR = Path(__file__).resolve().parent


def _load_twin():
    path = _V2_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("red_buff_v2_twin_inv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_well_invoices(
    asset_id: str,
    lift: str,
    history: List[dict],
    failure_events: Optional[List[dict]] = None,
    *,
    working_interest: float = 1.0,
) -> List[dict]:
    twin = _load_twin()
    invoices = twin.make_well_invoices(
        asset_id, history, working_interest, lift, failure_events or []
    )
    for inv in invoices:
        inv["asset_id"] = asset_id
        inv["organization_id"] = twin.ORGANIZATION_ID
    return invoices


def generate_supporting_invoices(
    asset_id: str,
    asset_type: str,
    history: List[dict],
    *,
    working_interest: float = 1.0,
) -> List[dict]:
    twin = _load_twin()
    invoices = twin.make_supp_invoices(asset_id, asset_type, history, working_interest)
    for inv in invoices:
        inv["asset_id"] = asset_id
        inv["organization_id"] = twin.ORGANIZATION_ID
    return invoices


def generate_well_revenue(
    asset_id: str,
    history: List[dict],
    *,
    net_revenue_interest: float,
) -> List[dict]:
    twin = _load_twin()
    ngl = random.uniform(twin.NGL_YIELD_MIN, twin.NGL_YIELD_MAX)
    rows = twin.make_revenue(asset_id, history, net_revenue_interest, ngl)
    for r in rows:
        r["asset_id"] = asset_id
        r["organization_id"] = twin.ORGANIZATION_ID
    return rows
