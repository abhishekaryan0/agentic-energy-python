"""Normalize and validate Snowflake batches before writing parquet.

Dtypes and core field names align with `APEX_PERMIAN/generate_digital_twin.py` for
sensor + daily production parquet (date as calendar date, float32 numerics,
naive timestamps, object strings, and `freq_hz` alongside `drive_frequency`).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional, Sequence, Tuple

import pandas as pd

from transformations.columns import WELL_SENSOR_READINGS_COLUMNS
from transformations.field_mappings import apply_export_field_mapping, resolve_parquet_columns
from transformations.schemas import TableExportSpec

logger = logging.getLogger(__name__)

_FLOAT32_COLUMNS = frozenset(
    col
    for col, _ in WELL_SENSOR_READINGS_COLUMNS
    if col not in ("organization_id", "well_id", "timestamp", "lift_type", "well_status")
) | {"freq_hz"}

_PROD_FLOAT32 = frozenset(
    {
        "oil_volume",
        "gas_volume",
        "water_volume",
        "boe",
        "test_oil",
        "test_water",
        "test_gas",
    }
)


def _to_object_str_series(s: pd.Series) -> pd.Series:
    """Object dtype with Python str or None (like script-built DataFrames)."""

    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if pd.isna(v):
            return None
        return str(v)

    return s.map(cell).astype(object)


def _coerce_production(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Match APEX_PERMIAN.generate_digital_twin._uniform_production_df for overlapping fields."""
    out = df.reindex(columns=list(columns)).copy()
    if "date" in out.columns:
        # Script: pd.to_datetime(df["date"]).dt.date
        parsed = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = parsed.dt.date
    for col in _PROD_FLOAT32:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for col in ("organization_id", "well_id", "lift_type", "status", "notes", "facility_id"):
        if col in out.columns:
            out[col] = _to_object_str_series(out[col])
    return out


def _strip_timestamp_tz_to_naive(s: pd.Series) -> pd.Series:
    """Scripts use naive datetimes; Snowflake TIMESTAMP_NTZ is naive."""
    ts = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(ts):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts


def _coerce_sensor(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Match APEX_PERMIAN.generate_digital_twin._build_well_sensor_readings_df patterns."""
    out = df.reindex(columns=list(columns)).copy()
    if "timestamp" in out.columns:
        out["timestamp"] = _strip_timestamp_tz_to_naive(out["timestamp"])
    if "freq_hz" in out.columns and "drive_frequency" in out.columns:
        out["freq_hz"] = out["freq_hz"].fillna(out["drive_frequency"])
    if "drive_frequency" in out.columns and "freq_hz" in out.columns:
        out["drive_frequency"] = out["drive_frequency"].fillna(out["freq_hz"])
    for col in ("organization_id", "well_id", "lift_type", "well_status"):
        if col in out.columns:
            out[col] = _to_object_str_series(out[col])
    for col in out.columns:
        if col in ("organization_id", "well_id", "timestamp", "lift_type", "well_status"):
            continue
        if col in _FLOAT32_COLUMNS or pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def normalize_batch(
    batch: pd.DataFrame,
    spec: TableExportSpec,
    organization_id: Optional[str] = None,
) -> pd.DataFrame:
    """Apply canonical column order, dtypes, and per-org export field mapping."""
    if batch is None or batch.empty:
        cols = resolve_parquet_columns(organization_id, spec.source_table, spec.columns)
        return pd.DataFrame(columns=list(cols))
    if spec.source_table in ("WELL_DAILY_PRODUCTION", "WELL_DAILY_PRODUCTION_V2"):
        if spec.source_table.endswith("_V2"):
            out = _coerce_v2_daily(batch, spec.columns)
        else:
            out = _coerce_production(batch, spec.columns)
    elif spec.source_table.endswith("_V2"):
        out = _coerce_v2_sensor(batch, spec.columns)
    else:
        out = _coerce_sensor(batch, spec.columns)
    if spec.source_table.endswith("_V2"):
        return out
    return apply_export_field_mapping(out, organization_id, spec.source_table)


def _coerce_v2_daily(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.reindex(columns=list(columns)).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    for col in ("organization_id", "asset_id", "lift_type", "run_status", "timestamp"):
        if col in out.columns:
            out[col] = _to_object_str_series(out[col])
    for col in out.columns:
        if col in ("organization_id", "asset_id", "date", "timestamp", "lift_type", "run_status"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def _coerce_v2_sensor(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.reindex(columns=list(columns)).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    for col in ("organization_id", "asset_id", "lift_type", "run_status", "timestamp"):
        if col in out.columns:
            out[col] = _to_object_str_series(out[col])
    for col in out.columns:
        if col in ("organization_id", "asset_id", "date", "timestamp", "lift_type", "run_status"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def validate_batch(
    batch: pd.DataFrame,
    spec: TableExportSpec,
    *,
    expected_organization_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Drop invalid rows and return (clean_df, dropped_count).

    Production requires well_id and date (v1) or asset_id and date (v2).
    Sensors require well_id and timestamp (v1) or asset_id and date (v2).

    When expected_organization_id is set and the batch includes organization_id,
    every remaining row must match (defense in depth on top of SQL WHERE).
    """
    if batch is None or batch.empty:
        return batch, 0

    df = batch
    before = len(df)
    is_v2 = spec.source_table.endswith("_V2")

    if is_v2:
        df = df.dropna(subset=["asset_id", "date"])
        df = df[df["asset_id"].astype(str).str.len() > 0]
    elif spec.source_table == "WELL_DAILY_PRODUCTION":
        df = df.dropna(subset=["well_id", "date"])
        df = df[df["well_id"].astype(str).str.len() > 0]
    else:
        df = df.dropna(subset=["well_id", "timestamp"])
        df = df[df["well_id"].astype(str).str.len() > 0]

    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped %d invalid rows exporting %s",
            dropped,
            spec.source_table,
        )

    if expected_organization_id and not df.empty and "organization_id" in df.columns:
        exp = str(expected_organization_id).strip()
        mask = df["organization_id"].astype(str).str.strip() == exp
        if not mask.all():
            bad = int((~mask).sum())
            raise ValueError(
                f"Organization mismatch in {spec.source_table} batch: {bad} row(s) "
                f"have organization_id != {exp!r} (check Snowflake data or export SQL)."
            )

    return df, dropped


def partition_date_value(row_date) -> str:
    """Format a partition folder value date=YYYY-MM-DD."""
    if isinstance(row_date, datetime):
        return row_date.date().isoformat()
    if isinstance(row_date, date):
        return row_date.isoformat()
    return pd.to_datetime(row_date).date().isoformat()


def add_partition_column(df: pd.DataFrame, spec: TableExportSpec) -> pd.DataFrame:
    """Add _partition_date used for hive-style S3 paths."""
    out = df.copy()
    if spec.partition_column == "date":
        out["_partition_date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif spec.partition_column == "timestamp_date":
        out["_partition_date"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    return out


def batch_high_watermark(df: pd.DataFrame, spec: TableExportSpec) -> Optional[str]:
    """Return ISO watermark string from a batch (max date or timestamp)."""
    if df is None or df.empty:
        return None
    if spec.watermark_column == "date":
        val = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(val):
            return None
        return val.date().isoformat()
    val = pd.to_datetime(df["timestamp"], errors="coerce").max()
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp) and val.tzinfo is not None:
        val = val.tz_convert("UTC").tz_localize(None)
    return val.isoformat()
