"""Snowflake reads for SYNTH_* exports."""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import pandas as pd

from transformations.config import TransformationConfig
from transformations.snowflake import ExportWindow, build_select_sql, connect_snowflake, quote_identifier
from transformations.synth_schemas import SynthSourceSpec

logger = logging.getLogger(__name__)


def _table_export_spec(spec: SynthSourceSpec):
    """Adapt SynthSourceSpec to transformations.schemas.TableExportSpec."""
    from transformations.schemas import TableExportSpec

    return TableExportSpec(
        source_table=spec.source_table,
        output_filename="",
        columns=spec.columns,
        order_by=spec.order_by,
        organization_column=spec.organization_column,
        watermark_column=spec.watermark_column,
    )


def iter_synth_batches(
    conn,
    config: TransformationConfig,
    spec: SynthSourceSpec,
    window: Optional[ExportWindow] = None,
) -> Iterator[pd.DataFrame]:
    export_spec = _table_export_spec(spec)
    sql, params = build_select_sql(config, export_spec, window=window)
    logger.info("Reading Snowflake %s", spec.source_table)
    cur = conn.cursor()
    try:
        cur.arraysize = config.batch_size
        cur.execute(sql, params)
        for batch in cur.fetch_pandas_batches():
            if batch is None or batch.empty:
                continue
            batch.columns = [str(c).lower() for c in batch.columns]
            yield batch.reindex(columns=list(spec.columns))
    finally:
        cur.close()


def count_synth_rows(
    conn,
    config: TransformationConfig,
    spec: SynthSourceSpec,
    window: Optional[ExportWindow] = None,
) -> int:
    from transformations.snowflake import count_rows_for_export_window

    return count_rows_for_export_window(conn, config, _table_export_spec(spec), window=window)


def preflight_synth_counts(
    conn,
    config: TransformationConfig,
    specs: List[SynthSourceSpec],
) -> dict:
    from transformations.snowflake import count_rows_for_organization

    org = (config.organization_id or "").strip()
    if not org:
        return {}
    db = config.snowflake.database or ""
    schema = config.snowflake.schema or ""
    out = {}
    for spec in specs:
        out[spec.source_table] = count_rows_for_organization(
            conn, db, schema, spec.source_table, org
        )
    return out
