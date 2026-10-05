"""Supporting assets, personnel, and clean-well catalog helpers for STEP V2 (like v1 master_assets)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

_V2_DIR = Path(__file__).resolve().parent

SUPPORTING_ASSET_DEFS = [
    ("Facility", "FAC-01", "make_facility"),
    ("Compressor", "COMP-01", "make_compressor"),
    ("Tank Battery", "TANK-01", "make_tank"),
    ("SWD Well", "SWD-01", "make_swd"),
]


def _load_twin():
    path = _V2_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("step_v2_twin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_clean_well_ids(wells: List[dict]) -> List[str]:
    return [w["asset_id"] for w in wells if w.get("is_clean")]


def compute_lease_starts(wells: List[dict]) -> Dict[str, str]:
    starts: Dict[str, datetime] = {}
    for w in wells:
        lease = w["lease"]
        spud = w["spud_date"]
        if isinstance(spud, str):
            spud = datetime.strptime(spud[:10], "%Y-%m-%d")
        if lease not in starts or spud < starts[lease]:
            starts[lease] = spud
    return {k: v.strftime("%Y-%m-%d") for k, v in starts.items()}


def build_supporting_asset_catalog(lease_starts: Dict[str, str], twin=None) -> List[dict]:
    twin = twin or _load_twin()
    rows = []
    for lease, info in twin.LEASES.items():
        for asset_type, suffix, _ in SUPPORTING_ASSET_DEFS:
            rows.append(
                {
                    "asset_id": f"{lease}-{suffix}",
                    "asset_type": asset_type,
                    "lift_type": None,
                    "lease": lease,
                    "area": info["area"],
                    "lease_type": info["lt"],
                    "working_interest": info["wi"],
                    "net_revenue_interest": info["nri"] if asset_type == "SWD Well" else None,
                    "history_start": lease_starts.get(lease),
                }
            )
    return rows


def get_lease_area(lease: str, twin=None) -> str:
    twin = twin or _load_twin()
    return twin.LEASES[lease]["area"]


def get_lease_location(lease: str, twin=None) -> dict:
    twin = twin or _load_twin()
    return twin.make_loc(twin.LEASES[lease])


def get_personnel(lease: str, twin=None) -> dict:
    twin = twin or _load_twin()
    return twin.make_pers(lease)


def dates_in_window(
    start: datetime,
    end: datetime,
    range_start: datetime,
    range_end: datetime,
) -> List[str]:
    lo = max(start, range_start)
    hi = min(end, range_end)
    if lo > hi:
        return []
    out = []
    d = lo
    while d <= hi:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def generate_supporting_history(
    asset_type: str,
    wells_only: dict,
    lease: str,
    sd: datetime,
    ed: datetime,
    twin=None,
) -> List[dict]:
    """Build supporting-asset daily history using V2 production-coupled generators."""
    twin = twin or _load_twin()
    info = twin.LEASES[lease]
    if asset_type == "Facility":
        return twin.make_facility(wells_only, lease, info, sd, ed)
    if asset_type == "Compressor":
        return twin.make_compressor(wells_only, lease, sd, ed)
    if asset_type == "Tank Battery":
        return twin.make_tank(wells_only, lease, sd, ed)
    if asset_type == "SWD Well":
        return twin.make_swd(wells_only, lease, info, sd, ed)
    raise ValueError(f"Unknown supporting asset type: {asset_type}")
