"""Supporting assets, personnel, and clean-well handling for the Red Bluff pipeline."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

_RED_BUFF_DIR = Path(__file__).resolve().parent

LEASES = ["Chaparral", "Mesquite", "Caliche", "Llano"]
_CLEAN_WELL_SLOTS = [9, 10, 15]

SUPPORTING_ASSET_DEFS = [
    ("Facility", "FAC-01", "build_facility"),
    ("Compressor", "COMP-01", "build_compressor"),
    ("SWD Well", "SWD-01", "build_swd"),
    ("Tank Battery", "TANK-01", "build_tank"),
]


def _load_master_builder():
    path = _RED_BUFF_DIR / "build_master.py"
    spec = importlib.util.spec_from_file_location("red_buff_master_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_clean_well_ids(wells: List[dict]) -> List[str]:
    """Pick 3 wells per lease (slots 9, 10, 15) — matches packaged clean-well design."""
    clean: List[str] = []
    for lease_idx, lease in enumerate(LEASES):
        base = lease_idx * 15
        for slot in _CLEAN_WELL_SLOTS:
            well_num = base + slot
            prefix = f"{lease}-12-{well_num}-"
            for w in wells:
                if w["asset_id"].startswith(prefix):
                    clean.append(w["asset_id"])
                    break
    return clean


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
                    "history_start": lease_starts[lease],
                }
            )
    return rows


def dates_in_window(start: datetime, end: datetime, range_start: datetime, range_end: datetime) -> List[str]:
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
