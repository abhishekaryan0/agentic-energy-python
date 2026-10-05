"""Supporting assets helpers for RED_BUFF V2 (like v1 master_assets)."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_V2_DIR = Path(__file__).resolve().parent

SUPPORTING_ASSET_DEFS = [
    ("Facility", "FAC-01"),
    ("Compressor", "COMP-01"),
    ("Tank Battery", "TANK-01"),
    ("SWD Well", "SWD-01"),
]


def _load_twin():
    path = _V2_DIR / "generate_digital_twin.py"
    spec = importlib.util.spec_from_file_location("red_buff_v2_twin", path)
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
        for asset_type, suffix in SUPPORTING_ASSET_DEFS:
            rows.append(
                {
                    "asset_id": f"{lease}-{suffix}",
                    "asset_type": asset_type,
                    "lift_type": None,
                    "lease": lease,
                    "area": info["sub_basin"],
                    "lease_type": info["lease_type"],
                    "working_interest": info["wi"],
                    "net_revenue_interest": info["nri"] if asset_type == "SWD Well" else None,
                    "history_start": lease_starts.get(lease),
                    "swd_pump": info.get("swd_pump"),
                }
            )
    return rows


def get_personnel(lease: str, twin=None) -> dict:
    twin = twin or _load_twin()
    return twin.make_pers(lease)


def generate_supporting_history(
    asset_type: str,
    wells_assets: dict,
    lease: str,
    sd: datetime,
    ed: datetime,
    twin=None,
    tank_hist=None,
) -> List[dict]:
    twin = twin or _load_twin()
    info = twin.LEASES[lease]
    if asset_type == "Facility":
        return twin.make_facility(wells_assets, lease, info, sd, ed)
    if asset_type == "Compressor":
        return twin.make_compressor(wells_assets, lease, sd, ed)
    if asset_type == "Tank Battery":
        return twin.make_tank(wells_assets, lease, sd, ed)
    if asset_type == "SWD Well":
        return twin.make_swd(wells_assets, lease, info, sd, ed, tank_hist or [])
    raise ValueError(f"Unknown supporting asset type: {asset_type}")
