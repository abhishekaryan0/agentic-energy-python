"""
S3 helpers for operations / financial inputs (invoices, production).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)


def get_latest_invoices_parquet_uris(
    *,
    bucket: Optional[str] = None,
    org: Optional[str] = None,
) -> List[str]:
    """
    Return s3:// URIs for the newest invoices_v1 and invoices_v2 under
    ``{org}/financial_data/run_ts=*/``.
    """
    bucket = (bucket or os.getenv("S3_BUCKET") or "").strip()
    org = (org or os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        return []

    import boto3

    s3 = boto3.client("s3")
    prefix = f"{org}/financial_data/run_ts="

    v1_keys: List[str] = []
    v2_keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj.get("Key") or ""
            if not k.endswith(".parquet"):
                continue
            if k.endswith("/invoices_v1.parquet"):
                v1_keys.append(k)
            elif k.endswith("/invoices_v2.parquet"):
                v2_keys.append(k)

    out: List[str] = []
    if v1_keys:
        v1_keys.sort(reverse=True)
        out.append(f"s3://{bucket}/{v1_keys[0]}")
    if v2_keys:
        v2_keys.sort(reverse=True)
        out.append(f"s3://{bucket}/{v2_keys[0]}")
    return out


def download_s3_uri_to_temp(s3_uri: str) -> str:
    """Download s3://bucket/key to a temp parquet file; return local path."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    import boto3

    fd, path = tempfile.mkstemp(prefix="ops_s3_", suffix=".parquet")
    os.close(fd)
    boto3.client("s3").download_file(bucket, key, path)
    return path


def _list_partition_keys(bucket: str, org: str, subdir: str) -> List[str]:
    import boto3

    prefix = f"{org}/sensor_data/{subdir}/date="
    keys: List[str] = []
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj.get("Key") or ""
            if k.endswith("/part-0000.parquet"):
                keys.append(k)
    keys.sort()
    return keys


def load_well_sensor_readings_from_s3(
    *,
    bucket: Optional[str] = None,
    org: Optional[str] = None,
    max_partition_days: Optional[int] = None,
) -> pd.DataFrame:
    """Load well sensor readings from date partitions; fall back to legacy single parquet."""
    bucket = (bucket or os.getenv("S3_BUCKET") or "").strip()
    org = (org or os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        raise ValueError("S3_BUCKET and ORGANIZATION_ID required")

    if max_partition_days is None:
        raw = (os.getenv("OPS_TRAINING_PARTITION_DAYS") or os.getenv("OPS_DETECTION_DAYS") or "250").strip()
        max_partition_days = int(raw) if raw.isdigit() else 250

    import io

    import boto3

    s3 = boto3.client("s3")
    keys = _list_partition_keys(bucket, org, "well_sensor_readings")
    if keys:
        if max_partition_days > 0:
            keys = keys[-max_partition_days:]
        frames: List[pd.DataFrame] = []
        for key in keys:
            buf = io.BytesIO()
            s3.download_fileobj(bucket, key, buf)
            buf.seek(0)
            frames.append(pd.read_parquet(buf))
        if frames:
            out = pd.concat(frames, ignore_index=True)
            logger.info(
                "Loaded well_sensor_readings from %s partition(s), rows=%s",
                len(frames),
                len(out),
            )
            return out

    legacy = f"{org}/sensor_data/well_sensor_readings.parquet"
    try:
        s3.head_object(Bucket=bucket, Key=legacy)
    except Exception as e:
        raise FileNotFoundError(
            f"No well_sensor_readings partitions or legacy s3://{bucket}/{legacy}"
        ) from e

    buf = io.BytesIO()
    s3.download_fileobj(bucket, legacy, buf)
    buf.seek(0)
    df = pd.read_parquet(buf)
    logger.info("Loaded legacy well_sensor_readings.parquet rows=%s", len(df))
    return df


def materialize_training_parquets_from_s3(
    *,
    bucket: Optional[str] = None,
    org: Optional[str] = None,
    max_partition_days: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Load partitioned sensor + production from S3 and write temp parquet files for training.
    Returns (sensor_parquet_path, production_parquet_path).
    """
    sensor_df = load_well_sensor_readings_from_s3(
        bucket=bucket, org=org, max_partition_days=max_partition_days
    )
    prod_df = load_daily_production_from_s3(
        bucket=bucket, org=org, max_partition_days=max_partition_days
    )
    fd_s, sensor_path = tempfile.mkstemp(prefix="ops_train_sensor_", suffix=".parquet")
    fd_p, prod_path = tempfile.mkstemp(prefix="ops_train_prod_", suffix=".parquet")
    os.close(fd_s)
    os.close(fd_p)
    sensor_df.to_parquet(sensor_path, index=False)
    prod_df.to_parquet(prod_path, index=False)
    return sensor_path, prod_path


def load_daily_production_from_s3(
    *,
    bucket: Optional[str] = None,
    org: Optional[str] = None,
    max_partition_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load daily production from date partitions; fall back to legacy single parquet.
    """
    bucket = (bucket or os.getenv("S3_BUCKET") or "").strip()
    org = (org or os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        raise ValueError("S3_BUCKET and ORGANIZATION_ID required")

    if max_partition_days is None:
        raw = (os.getenv("FINANCIAL_TRAIN_PARTITION_DAYS") or "365").strip()
        max_partition_days = int(raw) if raw.isdigit() else 365

    import io

    import boto3

    s3 = boto3.client("s3")
    keys = _list_partition_keys(bucket, org, "daily_production")
    if keys:
        if max_partition_days > 0:
            keys = keys[-max_partition_days:]
        frames: List[pd.DataFrame] = []
        for key in keys:
            buf = io.BytesIO()
            s3.download_fileobj(bucket, key, buf)
            buf.seek(0)
            frames.append(pd.read_parquet(buf))
        if frames:
            out = pd.concat(frames, ignore_index=True)
            logger.info(
                "Loaded daily_production from %s partition(s), rows=%s",
                len(frames),
                len(out),
            )
            return out

    legacy = f"{org}/sensor_data/daily_production.parquet"
    try:
        s3.head_object(Bucket=bucket, Key=legacy)
    except Exception as e:
        raise FileNotFoundError(
            f"No daily_production partitions or legacy s3://{bucket}/{legacy}"
        ) from e

    buf = io.BytesIO()
    s3.download_fileobj(bucket, legacy, buf)
    buf.seek(0)
    df = pd.read_parquet(buf)
    logger.info("Loaded legacy daily_production.parquet rows=%s", len(df))
    return df
