"""Supporting assets, personnel, and clean-well handling for the STEP pipeline."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

_STEP_DIR = Path(__file__).resolve().parent

LEASES = ["Bravo", "Maverick", "Carrizo", "Cotulla", "Pearsall"]

# Fixed clean wells — 3 per lease (from dataset spec).
CLEAN_WELL_IDS = [
    "Bravo-1H-1-H",
    "Bravo-1H-2-H",
    "Bravo-1H-3-H",
    "Maverick-1H-21-H",
    "Maverick-1H-22-V",
    "Maverick-1H-23-H",
    "Carrizo-1H-41-H",
    "Carrizo-1H-42-H",
    "Carrizo-1H-43-H",
    "Cotulla-1H-61-H",
    "Cotulla-1H-62-H",
    "Cotulla-1H-63-H",
    "Pearsall-1H-81-H",
    "Pearsall-1H-82-H",
    "Pearsall-1H-83-H",
]

SUPPORTING_ASSET_DEFS = [
    ("Facility", "FAC-01", "build_facility"),
    ("Compressor", "COMP-01", "build_compressor"),
    ("SWD Well", "SWD-01", "build_swd"),
    ("Tank Battery", "TANK-01", "build_tank"),
]


def _load_master_builder():
    path = _STEP_DIR / "build_master.py"
    spec = importlib.util.spec_from_file_location("step_master_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_clean_well_ids(wells: List[dict]) -> List[str]:
    """Return fixed clean well IDs that exist in the well catalog."""
    known = {w["asset_id"] for w in wells}
    return [wid for wid in CLEAN_WELL_IDS if wid in known]


def compute_lease_starts(wells: List[dict]) -> Dict[str, str]:
    """Earliest spud date per lease (ISO date strings)."""
    starts: Dict[str, datetime] = {}
    for w in wells:
        lease = w["lease"]
        spud = w["spud_date"]
        if isinstance(spud, str):
            spud = datetime.strptime(spud[:10], "%Y-%m-%d")
        if lease not in starts or spud < starts[lease]:
            starts[lease] = spud
    return {k: v.strftime("%Y-%m-%d") for k, v in starts.items()}


def build_supporting_asset_catalog(lease_starts: Dict[str, str]) -> List[dict]:
    rows = []
    for lease in LEASES:
        for asset_type, suffix, _ in SUPPORTING_ASSET_DEFS:
            rows.append(
                {
                    "asset_id": f"{lease}-{suffix}",
                    "asset_type": asset_type,
                    "lift_type": None,
                    "lease": lease,
                    "history_start": lease_starts.get(lease),
                }
            )
    return rows


def dates_in_window(
    start: datetime,
    end: datetime,
    range_start: datetime,
    range_end: datetime,
) -> List[str]:
    """Daily YYYY-MM-DD strings from start..end that fall inside the pipeline window."""
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
    builder_name: str,
    date_strings: List[str],
) -> List[dict]:
    if not date_strings:
        return []
    master = _load_master_builder()
    builder = getattr(master, builder_name)
    return builder(date_strings)


def apply_clean_well_history(history: List[dict], lift: str) -> List[dict]:
    master = _load_master_builder()
    return master.rebuild_clean_well(history, lift)


def get_personnel(lease: str) -> dict:
    return _load_master_builder().get_personnel(lease)


def get_lease_location(lease: str) -> dict:
    return _load_master_builder().LEASE_LOCATION[lease]


def get_lease_area(lease: str) -> str:
    return _load_master_builder().AREA_MAP.get(lease, "")
