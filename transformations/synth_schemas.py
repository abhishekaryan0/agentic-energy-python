"""Snowflake SYNTH_* table specs for master + financial S3 exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# Parquet outputs (scripts/generate_master.py)
MASTER_ASSETS_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "asset_type",
    "lift_type",
    "assigned_personnel_json",
    "financial_ledger_json",
    "raw_asset_json",
)

MASTER_HISTORY_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "asset_type",
    "lift_type",
    "date",
    "status",
    "notes",
    "oil_prod",
    "allocated_oil",
    "test_oil",
    "avg_tubing_pressure",
    "avg_casing_pressure",
    "avg_motor_amps",
    "gross_stroke_len",
    "net_stroke_len",
    "spm",
    "pump_fillage_pct",
    "freq_hz",
    "pump_intake_pressure",
    "motor_temp_f",
    "injection_rate_mcf",
    "raw_history_json",
)

INVOICES_V2_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "invoice_version",
    "invoice_id",
    "invoice_date",
    "status",
    "category",
    "service_description",
    "service_period_start",
    "service_period_end",
    "asset_id",
    "vendor_name",
    "vendor_type",
    "total_usd",
    "subtotal_usd",
    "gl_code",
    "cost_type",
    "capex_opex_flag",
    "line_items_count",
    "line_total_usd_sum",
    "raw_invoice_json",
)

SYNTH_ASSETS_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "asset_type",
    "lift_type",
    "lease",
    "basin",
    "sub_basin",
    "state",
    "county",
    "region",
)

SYNTH_PERSONNEL_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "role_key",
    "name",
    "role",
    "email",
)

SYNTH_HISTORY_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "asset_id",
    "date",
    "asset_type",
    "lift_type",
    "run_status",
    "history_json",
)

SYNTH_INVOICES_COLUMNS: Tuple[str, ...] = (
    "organization_id",
    "invoice_id",
    "invoice_date",
    "asset_id",
    "category",
    "gl_code",
    "service_description",
    "vendor",
    "total_usd",
    "invoice_json",
)


@dataclass(frozen=True)
class SynthSourceSpec:
    """Snowflake source for one leg of the synth export."""

    source_table: str
    columns: Tuple[str, ...]
    order_by: Tuple[str, ...] = ()
    watermark_column: str = ""  # date | invoice_date | empty = always full
    organization_column: str = "organization_id"


SYNTH_ASSETS_SPEC = SynthSourceSpec(
    source_table="SYNTH_ASSETS",
    columns=SYNTH_ASSETS_COLUMNS,
    order_by=("asset_id",),
)

SYNTH_PERSONNEL_SPEC = SynthSourceSpec(
    source_table="SYNTH_PERSONNEL",
    columns=SYNTH_PERSONNEL_COLUMNS,
    order_by=("asset_id", "role_key"),
)

SYNTH_HISTORY_SPEC = SynthSourceSpec(
    source_table="SYNTH_ASSET_DAILY_HISTORY",
    columns=SYNTH_HISTORY_COLUMNS,
    order_by=("asset_id", "date"),
    watermark_column="date",
)

SYNTH_INVOICES_SPEC = SynthSourceSpec(
    source_table="SYNTH_INVOICES",
    columns=SYNTH_INVOICES_COLUMNS,
    order_by=("asset_id", "invoice_date"),
    watermark_column="invoice_date",
)

SYNTH_SOURCE_ALIASES: Dict[str, SynthSourceSpec] = {
    "synth_assets": SYNTH_ASSETS_SPEC,
    "assets": SYNTH_ASSETS_SPEC,
    "synth_personnel": SYNTH_PERSONNEL_SPEC,
    "personnel": SYNTH_PERSONNEL_SPEC,
    "synth_history": SYNTH_HISTORY_SPEC,
    "history": SYNTH_HISTORY_SPEC,
    "synth_asset_daily_history": SYNTH_HISTORY_SPEC,
    "synth_invoices": SYNTH_INVOICES_SPEC,
    "invoices": SYNTH_INVOICES_SPEC,
    "financial": SYNTH_INVOICES_SPEC,
}

DEFAULT_SYNTH_SOURCES: Tuple[str, ...] = (
    "synth_assets",
    "synth_personnel",
    "synth_history",
    "synth_invoices",
)


def get_synth_source_specs(names: Iterable[str]) -> List[SynthSourceSpec]:
    specs: List[SynthSourceSpec] = []
    seen: set[str] = set()
    for name in names:
        key = (name or "").strip().lower()
        if not key or key in seen:
            continue
        spec = SYNTH_SOURCE_ALIASES.get(key)
        if spec is None:
            valid = ", ".join(sorted(SYNTH_SOURCE_ALIASES))
            raise ValueError(f"Unknown synth export source {name!r}. Valid: {valid}")
        if spec.source_table not in seen:
            specs.append(spec)
            seen.add(spec.source_table)
    return specs
