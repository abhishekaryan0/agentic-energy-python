"""Export Snowflake SYNTH_* tables to S3 master + financial parquet (scripts layout)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from transformations.config import TransformationConfig, _env_bool
from transformations.s3 import write_simple_parquet_to_s3
from transformations.snowflake import ExportWindow
from transformations.synth_checkpoint import (
    SynthExportCheckpoint,
    date_window,
    load_synth_checkpoint,
    save_synth_checkpoint,
)
from transformations.synth_mappers import (
    build_invoices_v2_df,
    build_master_assets_df,
    build_master_history_df,
)
from transformations.synth_schemas import (
    DEFAULT_SYNTH_SOURCES,
    SYNTH_ASSETS_SPEC,
    SYNTH_HISTORY_SPEC,
    SYNTH_INVOICES_SPEC,
    SYNTH_PERSONNEL_SPEC,
    get_synth_source_specs,
)
from transformations.synth_snowflake import (
    iter_synth_batches,
    preflight_synth_counts,
)

logger = logging.getLogger(__name__)

SYNTH_MODES = ("full", "incremental")


def _run_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _collect_table(
    conn,
    config: TransformationConfig,
    spec,
    window: Optional[ExportWindow],
) -> pd.DataFrame:
    frames = [b for b in iter_synth_batches(conn, config, spec, window=window) if not b.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _max_date(df: pd.DataFrame, col: str) -> Optional[str]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return str(s.max().date())


def run_synth_export(
    config: TransformationConfig,
    *,
    mode: Optional[str] = None,
    source_names: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Snowflake SYNTH_* → S3:
      {org}/digital_twin/master_data/run_ts=.../master_*.parquet
      {org}/financial_data/run_ts=.../invoices_v2.parquet
    """
    org = (config.organization_id or "").strip()
    if not org:
        raise ValueError("ORGANIZATION_ID is required for synth export")

    import os

    mode = (
        mode
        or os.getenv("SYNTH_EXPORT_MODE")
        or config.mode
        or "full"
    ).strip().lower()
    if mode not in SYNTH_MODES:
        raise ValueError(f"synth mode must be one of {SYNTH_MODES}")

    sources = get_synth_source_specs(source_names or DEFAULT_SYNTH_SOURCES)
    checkpoint = load_synth_checkpoint(config.s3_bucket, org, region_name=config.aws_region)
    if checkpoint is None:
        checkpoint = SynthExportCheckpoint(organization_id=org)

    hist_start, hist_end = date_window(
        checkpoint if mode == "incremental" else None,
        "history_high_watermark",
    )
    inv_start, inv_end = date_window(
        checkpoint if mode == "incremental" else None,
        "invoice_high_watermark",
    )
    hist_window = (
        ExportWindow(exclusive_start=hist_start, inclusive_end=hist_end)
        if mode == "incremental"
        else None
    )
    inv_window = (
        ExportWindow(exclusive_start=inv_start, inclusive_end=inv_end)
        if mode == "incremental"
        else None
    )

    ts = _run_ts()
    master_prefix = f"{org}/digital_twin/master_data/run_ts={ts}"
    financial_prefix = f"{org}/financial_data/run_ts={ts}"

    plan = {
        "mode": mode,
        "organization_id": org,
        "run_ts": ts,
        "master_prefix": f"s3://{config.s3_bucket}/{master_prefix}",
        "financial_prefix": f"s3://{config.s3_bucket}/{financial_prefix}",
        "history_window": {"start": str(hist_start), "end": str(hist_end)},
        "invoice_window": {"start": str(inv_start), "end": str(inv_end)},
    }
    if dry_run:
        return {"status": "success", "dry_run": True, **plan}

    from transformations.synth_snowflake import connect_snowflake

    conn = connect_snowflake(config.snowflake)
    try:
        preflight = preflight_synth_counts(conn, config, sources)
        logger.info("Synth preflight counts for %s: %s", org, preflight)
        if _env_bool("SYNTH_EXPORT_FAIL_ON_EMPTY", False) and sum(preflight.values()) == 0:
            raise ValueError(f"No SYNTH rows for organization_id={org!r}")

        assets_df = _collect_table(conn, config, SYNTH_ASSETS_SPEC, None)
        personnel_df = _collect_table(conn, config, SYNTH_PERSONNEL_SPEC, None)
        history_df = _collect_table(conn, config, SYNTH_HISTORY_SPEC, hist_window)
        invoices_df = _collect_table(conn, config, SYNTH_INVOICES_SPEC, inv_window)

        master_assets = build_master_assets_df(assets_df, personnel_df, invoices_df, org)
        master_history = build_master_history_df(history_df, org)
        invoices_v2 = build_invoices_v2_df(invoices_df, org)

        outputs: Dict[str, Any] = {}
        if not master_assets.empty or config.write_empty_files:
            uri, n = write_simple_parquet_to_s3(
                master_assets,
                config.s3_bucket,
                f"{master_prefix}/master_assets.parquet",
                region_name=config.aws_region,
                write_empty=config.write_empty_files,
            )
            outputs["master_assets"] = {"uri": uri, "rows": n}

        if not master_history.empty or config.write_empty_files:
            uri, n = write_simple_parquet_to_s3(
                master_history,
                config.s3_bucket,
                f"{master_prefix}/master_history.parquet",
                region_name=config.aws_region,
                write_empty=config.write_empty_files,
            )
            outputs["master_history"] = {"uri": uri, "rows": n}

        if not invoices_v2.empty or config.write_empty_files:
            uri, n = write_simple_parquet_to_s3(
                invoices_v2,
                config.s3_bucket,
                f"{financial_prefix}/invoices_v2.parquet",
                region_name=config.aws_region,
                write_empty=config.write_empty_files,
            )
            outputs["invoices_v2"] = {"uri": uri, "rows": n}

        if _max_date(history_df, "date"):
            checkpoint.history_high_watermark = _max_date(history_df, "date")
        if _max_date(invoices_df, "invoice_date"):
            checkpoint.invoice_high_watermark = _max_date(invoices_df, "invoice_date")
        checkpoint.mode = mode
        checkpoint.last_run_status = "success"
        checkpoint.last_run_ts = ts
        ck_uri = save_synth_checkpoint(config.s3_bucket, checkpoint, region_name=config.aws_region)

        return {
            "status": "success",
            "dry_run": False,
            "mode": mode,
            "organization_id": org,
            "run_ts": ts,
            "preflight_snowflake_row_counts": preflight,
            "snowflake_rows_read": {
                "SYNTH_ASSETS": len(assets_df),
                "SYNTH_PERSONNEL": len(personnel_df),
                "SYNTH_ASSET_DAILY_HISTORY": len(history_df),
                "SYNTH_INVOICES": len(invoices_df),
            },
            "outputs": outputs,
            "checkpoint_uri": ck_uri,
            "checkpoint": checkpoint.to_dict(),
        }
    except Exception:
        checkpoint.last_run_status = "failed"
        try:
            save_synth_checkpoint(config.s3_bucket, checkpoint, region_name=config.aws_region)
        except Exception:
            logger.exception("Failed to save synth checkpoint after error")
        raise
    finally:
        conn.close()
