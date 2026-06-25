"""Well ID helpers for joining production/sensor rows to invoice parquets."""

from __future__ import annotations

from typing import AbstractSet, Iterable, Set


def swap_hv_suffix(well_id: str) -> str:
    """Swap trailing -H / -V (Apex digital-twin naming drift)."""
    w = str(well_id).strip()
    if w.endswith("-H"):
        return f"{w[:-2]}-V"
    if w.endswith("-V"):
        return f"{w[:-2]}-H"
    return w


def match_well_for_invoice_lookup(
    production_well_id: str,
    invoice_well_ids: AbstractSet[str],
) -> str:
    """
    Resolve a production/sensor well_id to the key used in invoice parquet.

    Tries exact match, then a single H/V suffix swap (common in Apex Permian data).
    """
    w = str(production_well_id).strip()
    if w in invoice_well_ids:
        return w
    alt = swap_hv_suffix(w)
    if alt in invoice_well_ids:
        return alt
    return w


def remap_invoice_rows_to_production_wells(
    invoice_well_ids: Iterable[str],
    production_well_ids: AbstractSet[str],
) -> dict[str, str]:
    """Map each invoice well_id to the production well_id it should roll up under."""
    prod: Set[str] = {str(x).strip() for x in production_well_ids}
    inv: Set[str] = {str(x).strip() for x in invoice_well_ids}
    out: dict[str, str] = {}
    for inv_id in inv:
        if inv_id in prod:
            out[inv_id] = inv_id
            continue
        alt = swap_hv_suffix(inv_id)
        if alt in prod:
            out[inv_id] = alt
        else:
            out[inv_id] = inv_id
    return out
