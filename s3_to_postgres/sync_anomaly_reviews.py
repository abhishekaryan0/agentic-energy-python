"""Sync anomaly_reviews parquet files from S3 into PostgreSQL."""

from __future__ import annotations

from typing import Any, Dict, Optional

from s3_to_postgres.db import dataframe_to_anomaly_review_rows, upsert_anomaly_reviews
from s3_to_postgres.s3_io import ANOMALY_REVIEWS_KEY_FRAGMENT, list_anomaly_review_keys
from s3_to_postgres.sync_common import run_table_sync


def sync_anomaly_reviews(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_table_sync(
        event,
        default_sync_table="anomaly_reviews",
        list_keys_fn=list_anomaly_review_keys,
        key_fragment=ANOMALY_REVIEWS_KEY_FRAGMENT,
        dataframe_to_rows_fn=dataframe_to_anomaly_review_rows,
        upsert_fn=upsert_anomaly_reviews,
    )
