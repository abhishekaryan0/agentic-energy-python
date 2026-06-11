"""Generate well and supporting-asset invoices for a date-bounded history chunk."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import List, Optional

_RED_BUFF_DIR = Path(__file__).resolve().parent


def _load_well_invoice_lib():
    path = _RED_BUFF_DIR / "generate_invoices.py"
    spec = importlib.util.spec_from_file_location("red_buff_well_invoices", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_master_invoice_lib():
    path = _RED_BUFF_DIR / "build_master.py"
    spec = importlib.util.spec_from_file_location("red_buff_master_invoices", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_well_invoices(
    asset_id: str,
    lift: str,
    history: List[dict],
    *,
    skip_workover: bool = False,
) -> List[dict]:
    """Build invoice dicts for one well over a history chunk (inclusive dates)."""
    lib = _load_well_invoice_lib()
    invoices: List[dict] = []
    fail_start: Optional[str] = None

    for day in history:
        date_str = day["date"]
        status = day.get("run_status", "")

        if date_str.endswith("-28"):
            for exp in lib.EXPENSE_LIBRARY[lift]["Monthly"]:
                invoices.append(
                    {
                        "invoice_id": f"RBNR-{date_str[:4]}-{random.randint(10000, 99999)}",
                        "invoice_date": date_str,
                        "category": "Monthly LOE",
                        "gl_code": exp["gl"],
                        "service_description": exp["desc"],
                        "vendor": exp["vendor"],
                        "total_usd": round(random.uniform(*exp["range"]), 2),
                        "asset_id": asset_id,
                    }
                )
            for exp in lib.EXPENSE_LIBRARY[lift]["Ad_Hoc"]:
                if random.random() < exp["prob"]:
                    invoices.append(
                        {
                            "invoice_id": f"RBNR-{date_str[:4]}-{random.randint(10000, 99999)}",
                            "invoice_date": date_str,
                            "category": "Ad-Hoc LOE",
                            "gl_code": exp["gl"],
                            "service_description": exp["desc"],
                            "vendor": exp["vendor"],
                            "total_usd": round(random.uniform(*exp["range"]), 2),
                            "asset_id": asset_id,
                        }
                    )

        if status == "SHUT-IN" and fail_start is None:
            fail_start = date_str
        elif status == "PRODUCING" and fail_start is not None:
            if not skip_workover:
                inv = lib.build_workover_invoice(asset_id, date_str, lift)
                inv["asset_id"] = asset_id
                invoices.append(inv)
            fail_start = None

    return invoices


def generate_supporting_invoices(
    asset_id: str,
    asset_type: str,
    history: List[dict],
) -> List[dict]:
    """Monthly / ad-hoc invoices for facility, compressor, SWD, tank battery."""
    master = _load_master_invoice_lib()
    lib = master.EXPENSE_MAP[asset_type]
    raw = master.gen_invoices(history, lib)
    return [{**inv, "asset_id": asset_id} for inv in raw]
