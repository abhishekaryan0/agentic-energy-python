"""Canonical table specs for Snowflake exports consumed by ML pipelines."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from transformations.columns import WELL_DAILY_PRODUCTION_COLUMNS, WELL_SENSOR_READINGS_COLUMNS


def _sensor_parquet_columns() -> Tuple[str, ...]:
    """
    Canonical sensor columns plus script-compatible alias.

    scripts/generate_digital_twin2.py writes `freq_hz`; Snowflake stores the same
    signal as DRIVE_FREQUENCY (see RED_BUFF/column_maps). We select both so
    parquet matches twin script field names without breaking consumers of
    `drive_frequency`.
    """
    base = tuple(column for column, _dtype in WELL_SENSOR_READINGS_COLUMNS)
    if "freq_hz" in base:
        return base
    out: List[str] = []
    inserted = False
    for column in base:
        out.append(column)
        if column == "drive_frequency" and not inserted:
            out.append("freq_hz")
            inserted = True
    return tuple(out)


@dataclass(frozen=True)
class TableExportSpec:
    """Snowflake source table and destination parquet shape."""

    source_table: str
    output_filename: str
    columns: Tuple[str, ...]
    order_by: Tuple[str, ...] = ()
    organization_column: str = "organization_id"
    partition_subdir: str = ""
    partition_column: str = ""  # date | timestamp_date
    watermark_column: str = ""  # date | timestamp
    dedupe_keys: Tuple[str, ...] = ()


WELL_SENSOR_READINGS_SPEC = TableExportSpec(
    source_table="WELL_SENSOR_READINGS",
    output_filename="well_sensor_readings.parquet",
    columns=_sensor_parquet_columns(),
    order_by=("well_id", "timestamp"),
    partition_subdir="well_sensor_readings",
    partition_column="timestamp_date",
    watermark_column="timestamp",
    dedupe_keys=("well_id", "timestamp"),
)

WELL_DAILY_PRODUCTION_SPEC = TableExportSpec(
    source_table="WELL_DAILY_PRODUCTION",
    output_filename="daily_production.parquet",
    columns=tuple(column for column, _dtype in WELL_DAILY_PRODUCTION_COLUMNS),
    order_by=("well_id", "date"),
    partition_subdir="daily_production",
    partition_column="date",
    watermark_column="date",
    dedupe_keys=("well_id", "date"),
)

EXPORT_SPECS: Dict[str, TableExportSpec] = {
    "well_sensor_readings": WELL_SENSOR_READINGS_SPEC,
    "sensor_readings": WELL_SENSOR_READINGS_SPEC,
    "sensors": WELL_SENSOR_READINGS_SPEC,
    "well_daily_production": WELL_DAILY_PRODUCTION_SPEC,
    "daily_production": WELL_DAILY_PRODUCTION_SPEC,
    "production": WELL_DAILY_PRODUCTION_SPEC,
}


def get_export_specs(table_names: Iterable[str]) -> List[TableExportSpec]:
    """Resolve requested table names or aliases to de-duplicated export specs."""
    specs: List[TableExportSpec] = []
    seen = set()
    unknown = []

    for raw_name in table_names:
        name = (raw_name or "").strip().lower()
        if not name:
            continue
        spec = EXPORT_SPECS.get(name)
        if spec is None:
            unknown.append(raw_name)
            continue
        if spec.source_table not in seen:
            specs.append(spec)
            seen.add(spec.source_table)

    if unknown:
        known = ", ".join(sorted(EXPORT_SPECS))
        raise ValueError(f"Unknown transform table(s): {unknown}. Known values: {known}")
    if not specs:
        raise ValueError("No transform tables selected")
    return specs
