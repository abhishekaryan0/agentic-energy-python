"""Sync operation_recommendations parquet files from S3 into PostgreSQL."""

from __future__ import annotations

from typing import Any, Dict, Optional

from s3_to_postgres.db import (
    dataframe_to_operation_recommendation_rows,
    ensure_operation_recommendations_schema,
    upsert_operation_recommendations,
)
from s3_to_postgres.s3_io import OPERATIONS_RECOMMENDATIONS_KEY_FRAGMENT, list_operation_recommendation_keys
from s3_to_postgres.sync_common import run_table_sync


def sync_operation_recommendations(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_table_sync(
        event,
        default_sync_table="operation_recommendations",
        list_keys_fn=list_operation_recommendation_keys,
        key_fragment=OPERATIONS_RECOMMENDATIONS_KEY_FRAGMENT,
        dataframe_to_rows_fn=dataframe_to_operation_recommendation_rows,
        upsert_fn=upsert_operation_recommendations,
        on_connect=ensure_operation_recommendations_schema,
    )
