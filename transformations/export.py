"""Orchestrate Snowflake exports to S3 snapshots and date partitions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from transformations.checkpoint import (
    ExportCheckpoint,
    advance_production_watermark,
    advance_sensor_watermark,
    load_checkpoint,
    production_window,
    save_checkpoint,
    sensor_window,
)
from transformations.config import TransformationConfig
from transformations.mappers import (
    add_partition_column,
    batch_high_watermark,
    normalize_batch,
    validate_batch,
)
from transformations.schemas import TableExportSpec
from transformations.s3 import (
    build_s3_key,
    merge_snapshot,
    partition_s3_key,
    write_dataframe_to_s3,
    write_parquet_batches_to_s3,
)
from transformations.snowflake import (
    ExportWindow,
    count_incremental_new_rows,
    iter_table_batches,
    preflight_org_row_counts,
)

logger = logging.getLogger(__name__)


def _window_for_spec(
    spec: TableExportSpec,
    checkpoint: Optional[ExportCheckpoint],
    mode: str,
) -> Optional[ExportWindow]:
    if mode != "incremental":
        return None
    if spec.watermark_column == "date":
        start, end = production_window(checkpoint)
        return ExportWindow(exclusive_start=start, inclusive_end=end)
    start, end = sensor_window(checkpoint)
    return ExportWindow(exclusive_start=start, inclusive_end=end)


def export_table(
    conn,
    config: TransformationConfig,
    spec: TableExportSpec,
    checkpoint: Optional[ExportCheckpoint],
) -> Dict[str, Any]:
    """Export one table: optional partitions, snapshot, watermark updates."""
    prefix = config.sensor_data_prefix()
    mode = config.mode
    window = _window_for_spec(spec, checkpoint, mode)

    partition_frames: Dict[str, List[pd.DataFrame]] = {}
    snapshot_frames: List[pd.DataFrame] = []
    row_count = 0
    dropped_total = 0
    wm_out: Optional[str] = None
    keep_snapshot_frames = config.write_snapshots or (
        config.merge_snapshots and mode == "incremental"
    )

    for raw in iter_table_batches(conn, config, spec, window=window):
        batch = normalize_batch(raw, spec)
        batch, dropped = validate_batch(
            batch,
            spec,
            expected_organization_id=config.organization_id,
        )
        dropped_total += dropped
        if batch.empty:
            continue

        row_count += len(batch)

        batch_wm = batch_high_watermark(batch, spec)
        if batch_wm:
            if spec.watermark_column == "date":
                wm_out = advance_production_watermark(wm_out, batch_wm)
            else:
                wm_out = advance_sensor_watermark(wm_out, batch_wm)

        if config.write_partitions:
            partitioned = add_partition_column(batch, spec)
            for part_date, group in partitioned.groupby("_partition_date", sort=True):
                clean = group.drop(columns=["_partition_date"], errors="ignore")
                partition_frames.setdefault(str(part_date), []).append(clean)

        if keep_snapshot_frames:
            snapshot_frames.append(batch)

    partition_uris: List[str] = []
    if config.write_partitions and partition_frames:
        for part_date, frames in sorted(partition_frames.items()):
            combined = pd.concat(frames, ignore_index=True)
            key = partition_s3_key(prefix, spec.partition_subdir, part_date)
            uri, _ = write_dataframe_to_s3(
                combined,
                config.s3_bucket,
                key,
                spec,
                region_name=config.aws_region,
                write_empty=config.write_empty_files,
            )
            if uri:
                partition_uris.append(uri)

    snapshot_uri: Optional[str] = None
    snapshot_rows = 0
    if config.write_snapshots:
        new_df = (
            pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
        )
        if mode == "full":
            if snapshot_frames or config.write_empty_files:
                key = build_s3_key(prefix, spec.output_filename)
                snapshot_uri, snapshot_rows = write_parquet_batches_to_s3(
                    snapshot_frames,
                    config.s3_bucket,
                    key,
                    spec,
                    region_name=config.aws_region,
                    write_empty=config.write_empty_files,
                )
        elif config.merge_snapshots and not new_df.empty:
            snapshot_uri, snapshot_rows = merge_snapshot(
                new_df,
                config.s3_bucket,
                prefix,
                spec,
                region_name=config.aws_region,
                write_empty=config.write_empty_files,
            )
        elif not new_df.empty:
            key = build_s3_key(prefix, spec.output_filename)
            snapshot_uri, snapshot_rows = write_parquet_batches_to_s3(
                [new_df],
                config.s3_bucket,
                key,
                spec,
                region_name=config.aws_region,
                write_empty=config.write_empty_files,
            )

    return {
        "table": spec.source_table,
        "mode": mode,
        "rows_read": row_count,
        "rows_dropped": dropped_total,
        "snapshot_uri": snapshot_uri,
        "snapshot_rows": snapshot_rows,
        "partition_uris": partition_uris,
        "watermark": wm_out,
        "window": {
            "exclusive_start": str(window.exclusive_start) if window else None,
            "inclusive_end": str(window.inclusive_end) if window else None,
        },
    }


def run_export(
    config: TransformationConfig,
    specs: List[TableExportSpec],
) -> Dict[str, Any]:
    """Run exports for all specs and persist checkpoint on success."""
    from transformations.snowflake import connect_snowflake

    prefix = config.sensor_data_prefix()
    org = (config.organization_id or "").strip()
    checkpoint: Optional[ExportCheckpoint] = None
    if org:
        checkpoint = load_checkpoint(
            config.s3_bucket,
            prefix,
            org,
            region_name=config.aws_region,
        )
        if checkpoint is None:
            checkpoint = ExportCheckpoint(organization_id=org)

    conn = connect_snowflake(config.snowflake)
    exports: List[Dict[str, Any]] = []
    preflight_counts: Dict[str, int] = {}
    incremental_new_counts: Dict[str, int] = {}
    try:
        from transformations.config import _env_bool

        if org:
            preflight_counts = preflight_org_row_counts(conn, config, specs)
            logger.info(
                "Snowflake preflight for organization_id=%s (tenant-scoped row counts): %s",
                org,
                preflight_counts,
            )
            total_preflight = sum(preflight_counts.values())
            if total_preflight == 0:
                logger.warning(
                    "No rows found in Snowflake for organization_id=%r on tables %s — "
                    "parquet exports may be empty (check ORGANIZATION_ID matches loaded data).",
                    org,
                    list(preflight_counts.keys()),
                )

            if _env_bool("TRANSFORM_FAIL_ON_EMPTY_SNOWFLAKE", False) and total_preflight == 0:
                raise ValueError(
                    f"No Snowflake rows for organization_id={org!r}; "
                    "set TRANSFORM_FAIL_ON_EMPTY_SNOWFLAKE=0 to export empty files anyway."
                )

        if (
            config.mode == "incremental"
            and org
            and checkpoint
            and _env_bool("TRANSFORM_SKIP_WHEN_NO_NEW_DATA", True)
        ):
            incremental_new_counts = count_incremental_new_rows(
                conn, config, specs, checkpoint
            )
            if incremental_new_counts and sum(incremental_new_counts.values()) == 0:
                logger.info(
                    "No new Snowflake rows since checkpoint for org %s — skipping export (%s)",
                    org,
                    incremental_new_counts,
                )
                if checkpoint:
                    checkpoint.last_run_status = "skipped"
                    checkpoint.mode = config.mode
                ck_uri = None
                if org and checkpoint:
                    ck_uri = save_checkpoint(
                        config.s3_bucket,
                        prefix,
                        checkpoint,
                        region_name=config.aws_region,
                    )
                return {
                    "status": "skipped",
                    "reason": "no_new_rows_since_checkpoint",
                    "mode": config.mode,
                    "organization_id": org,
                    "s3_prefix": prefix,
                    "preflight_snowflake_row_counts": preflight_counts,
                    "incremental_new_row_counts": incremental_new_counts,
                    "checkpoint_uri": ck_uri,
                    "checkpoint": checkpoint.to_dict() if checkpoint else {},
                    "exports": [],
                }

        if incremental_new_counts:
            logger.info(
                "Incremental new rows since checkpoint: %s",
                incremental_new_counts,
            )

        for spec in specs:
            result = export_table(conn, config, spec, checkpoint)
            exports.append(result)

            if checkpoint:
                wm = result.get("watermark")
                if wm and result.get("rows_read", 0) > 0:
                    if spec.watermark_column == "date":
                        checkpoint.production_high_watermark = advance_production_watermark(
                            checkpoint.production_high_watermark,
                            wm,
                        )
                    else:
                        checkpoint.sensor_high_watermark = advance_sensor_watermark(
                            checkpoint.sensor_high_watermark,
                            wm,
                        )

        if checkpoint:
            checkpoint.last_run_status = "success"
            checkpoint.mode = config.mode
        ck_uri = None
        if org and checkpoint:
            ck_uri = save_checkpoint(
                config.s3_bucket,
                prefix,
                checkpoint,
                region_name=config.aws_region,
            )

        return {
            "status": "success",
            "mode": config.mode,
            "organization_id": org or None,
            "s3_prefix": prefix,
            "preflight_snowflake_row_counts": preflight_counts,
            "incremental_new_row_counts": incremental_new_counts,
            "checkpoint_uri": ck_uri,
            "checkpoint": checkpoint.to_dict() if checkpoint else {},
            "exports": exports,
        }
    except Exception:
        if checkpoint:
            checkpoint.last_run_status = "failed"
            if org:
                try:
                    save_checkpoint(
                        config.s3_bucket,
                        prefix,
                        checkpoint,
                        region_name=config.aws_region,
                    )
                except Exception:
                    logger.exception("Failed to persist failed-run checkpoint")
        raise
    finally:
        conn.close()
