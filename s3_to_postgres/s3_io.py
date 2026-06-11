"""S3 helpers for listing and reading detection parquet files."""

from __future__ import annotations

import io
import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ANOMALY_SUGGESTIONS_KEY_FRAGMENT = "anomaly_suggestions_run_ts="
ANOMALY_VIOLATIONS_KEY_FRAGMENT = "anomaly_violations_run_ts="
ANOMALY_REVIEWS_KEY_FRAGMENT = "anomaly_reviews.parquet"
OPERATIONS_RECOMMENDATIONS_KEY_FRAGMENT = "operations_recommendations.parquet"
HISTORY_PREFIX = "history/"
RUN_TS_PREFIX = "run_ts="


def list_anomaly_suggestion_keys(
    s3_client,
    bucket: str,
    detection_prefix: str,
) -> List[str]:
    prefix = f"{detection_prefix.rstrip('/')}/"
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = item["Key"]
            if ANOMALY_SUGGESTIONS_KEY_FRAGMENT in key and key.endswith(".parquet"):
                keys.append(key)
    keys.sort()
    return keys


def list_anomaly_violation_keys(
    s3_client,
    bucket: str,
    detection_prefix: str,
) -> List[str]:
    prefix = f"{detection_prefix.rstrip('/')}/{HISTORY_PREFIX}"
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = item["Key"]
            if ANOMALY_VIOLATIONS_KEY_FRAGMENT in key and key.endswith(".parquet"):
                keys.append(key)
    keys.sort()
    return keys


def list_run_ts_parquet_keys(
    s3_client,
    bucket: str,
    detection_prefix: str,
    *,
    key_fragment: str,
) -> List[str]:
    prefix = f"{detection_prefix.rstrip('/')}/"
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = item["Key"]
            if RUN_TS_PREFIX in key and key_fragment in key and key.endswith(".parquet"):
                keys.append(key)
    keys.sort()
    return keys


def list_anomaly_review_keys(s3_client, bucket: str, detection_prefix: str) -> List[str]:
    return list_run_ts_parquet_keys(
        s3_client,
        bucket,
        detection_prefix,
        key_fragment=ANOMALY_REVIEWS_KEY_FRAGMENT,
    )


def list_operation_recommendation_keys(s3_client, bucket: str, detection_prefix: str) -> List[str]:
    return list_run_ts_parquet_keys(
        s3_client,
        bucket,
        detection_prefix,
        key_fragment=OPERATIONS_RECOMMENDATIONS_KEY_FRAGMENT,
    )


def read_parquet_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def event_detail_s3_key(event: dict) -> Optional[str]:
    """Extract an object key from EventBridge S3 or direct Lambda test events."""
    if not event:
        return None
    if event.get("s3_key"):
        return str(event["s3_key"])

    detail = event.get("detail") or {}
    obj = detail.get("object") or {}
    if obj.get("key"):
        return str(obj["key"])

    records = event.get("Records") or []
    if records:
        record = records[0]
        return record.get("s3", {}).get("object", {}).get("key")

    return None
