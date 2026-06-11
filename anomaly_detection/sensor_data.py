"""
Load well_sensor_readings for anomaly detection from org-scoped S3 layout.

Preferred (transformations export):
  s3://{bucket}/{org_id}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet

Legacy fallback:
  s3://{bucket}/{org_id}/sensor_data/well_sensor_readings.parquet
"""

from __future__ import annotations

import io
import logging
import os
import re
from datetime import date, timedelta
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

SENSOR_SUBDIR = "well_sensor_readings"
LEGACY_SENSOR_FILENAME = "well_sensor_readings.parquet"
PARTITION_DATE_RE = re.compile(r"date=(\d{4}-\d{2}-\d{2})")


def sensor_data_prefix(organization_id: str) -> str:
    """S3 key prefix without bucket: {org}/sensor_data."""
    org = (organization_id or "").strip().rstrip("/")
    if not org:
        raise ValueError("organization_id is required")
    return f"{org}/sensor_data"


def legacy_sensor_parquet_key(organization_id: str) -> str:
    return f"{sensor_data_prefix(organization_id)}/{LEGACY_SENSOR_FILENAME}"


def legacy_sensor_parquet_uri(bucket: str, organization_id: str) -> str:
    return f"s3://{bucket}/{legacy_sensor_parquet_key(organization_id)}"


def partition_prefix(organization_id: str, subdir: str = SENSOR_SUBDIR) -> str:
    return f"{sensor_data_prefix(organization_id)}/{subdir}/date="


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _get_s3_client():
    import boto3

    return boto3.client("s3")


def _read_parquet_s3_object(bucket: str, key: str) -> pd.DataFrame:
    s3 = _get_s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def _read_parquet_local(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def list_sensor_partition_keys(
    bucket: str,
    organization_id: str,
    *,
    subdir: str = SENSOR_SUBDIR,
) -> List[Tuple[str, str]]:
    """
    List partition parquet keys newest-first.
    Returns [(date_str, key), ...].
    """
    prefix = partition_prefix(organization_id, subdir=subdir)
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    entries: List[Tuple[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("part-0000.parquet"):
                continue
            m = PARTITION_DATE_RE.search(key)
            if not m:
                continue
            entries.append((m.group(1), key))
    entries.sort(key=lambda x: x[0], reverse=True)
    return entries


def filter_partition_keys_in_date_range(
    keyed: List[Tuple[str, str]],
    *,
    start: date,
    end: date,
) -> List[Tuple[str, str]]:
    """Return partition (date_str, key) pairs with start <= date <= end, ascending by date."""
    out: List[Tuple[str, str]] = []
    for date_str, key in keyed:
        try:
            d = pd.to_datetime(date_str).date()
        except Exception:
            continue
        if start <= d <= end:
            out.append((date_str, key))
    out.sort(key=lambda x: x[0])
    return out


def load_well_sensor_readings_from_s3_date_range(
    bucket: str,
    organization_id: str,
    *,
    start_date: date,
    end_date: date,
    prefer_partitions: bool = True,
    subdir: str = SENSOR_SUBDIR,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load sensor rows from date partitions in [start_date, end_date] (inclusive).

    Returns (dataframe, partition_keys_loaded).
    Falls back to legacy single parquet when no partitions match.
    """
    if start_date > end_date:
        return pd.DataFrame(), []

    partition_keys = list_sensor_partition_keys(bucket, organization_id, subdir=subdir)
    if prefer_partitions and partition_keys:
        selected = filter_partition_keys_in_date_range(
            partition_keys, start=start_date, end=end_date
        )
        if selected:
            df = _load_frames_from_keys(bucket, selected, max_partition_days=None)
            keys_loaded = [k for _, k in selected]
            return df, keys_loaded
        logger.info(
            "No partitions in [%s, %s] for org %s (listed %s partition dates)",
            start_date,
            end_date,
            organization_id,
            len(partition_keys),
        )

    legacy_key = legacy_sensor_parquet_key(organization_id)
    s3 = _get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=legacy_key)
    except Exception:
        return pd.DataFrame(), []

    logger.info("Loading legacy sensor parquet s3://%s/%s", bucket, legacy_key)
    return _read_parquet_s3_object(bucket, legacy_key), [legacy_key]


def _load_frames_from_keys(
    bucket: str,
    keyed: List[Tuple[str, str]],
    *,
    max_partition_days: Optional[int],
) -> pd.DataFrame:
    if max_partition_days is not None and max_partition_days > 0:
        seen_dates: List[str] = []
        selected: List[Tuple[str, str]] = []
        for date_str, key in keyed:
            if date_str not in seen_dates:
                seen_dates.append(date_str)
            if len(seen_dates) > max_partition_days:
                break
            selected.append((date_str, key))
        keyed = selected

    if not keyed:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for date_str, key in keyed:
        try:
            frames.append(_read_parquet_s3_object(bucket, key))
        except Exception as e:
            logger.warning("Skipping partition s3://%s/%s: %s", bucket, key, e)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %s rows from %s partition file(s) (dates %s .. %s)",
        len(out),
        len(frames),
        keyed[-1][0],
        keyed[0][0],
    )
    return out


def load_well_sensor_readings_from_s3(
    bucket: str,
    organization_id: str,
    *,
    max_partition_days: Optional[int] = None,
    prefer_partitions: bool = True,
) -> pd.DataFrame:
    """
    Load sensor readings for an org from S3.
    Uses date partitions when present; falls back to legacy root parquet.
    """
    if max_partition_days is None:
        raw = os.getenv("ANOMALY_SENSOR_PARTITION_DAYS", "250").strip()
        max_partition_days = int(raw) if raw.isdigit() else 250

    partition_keys = list_sensor_partition_keys(bucket, organization_id)
    legacy_key = legacy_sensor_parquet_key(organization_id)

    if prefer_partitions and partition_keys:
        df = _load_frames_from_keys(
            bucket, partition_keys, max_partition_days=max_partition_days
        )
        if not df.empty:
            return df
        logger.warning("Partition listing found keys but no rows loaded; trying legacy file")

    s3 = _get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=legacy_key)
    except Exception:
        if partition_keys:
            return _load_frames_from_keys(
                bucket, partition_keys, max_partition_days=max_partition_days
            )
        raise FileNotFoundError(
            f"No sensor data for org {organization_id}: "
            f"no partitions under {partition_prefix(organization_id)} "
            f"and no s3://{bucket}/{legacy_key}"
        )

    logger.info("Loading legacy sensor parquet s3://%s/%s", bucket, legacy_key)
    return _read_parquet_s3_object(bucket, legacy_key)


def load_well_sensor_readings_from_parquet(path: str) -> pd.DataFrame:
    """Load a single parquet file (local path or s3://bucket/key)."""
    if path.startswith("s3://"):
        bucket, key = _parse_s3_uri(path)
        return _read_parquet_s3_object(bucket, key)
    return _read_parquet_local(path)


def _collect_local_partition_paths(
    base_dir: str,
    subdir: str = SENSOR_SUBDIR,
) -> List[Tuple[str, str]]:
    """
    Return [(date_str, path), ...] sorted by date descending.

    Supports channel layouts:
    - .../sensor_data/well_sensor_readings/date=.../  (full sensor_data prefix)
    - .../well_sensor_readings/date=.../              (sensor-only S3 prefix)
    """
    entries: List[Tuple[str, str]] = []
    roots = [os.path.join(base_dir, subdir), base_dir]
    seen_paths: set = set()
    for part_root in roots:
        if not os.path.isdir(part_root):
            continue
        for name in os.listdir(part_root):
            if not name.startswith("date="):
                continue
            date_str = name.split("=", 1)[1]
            part_file = os.path.join(part_root, name, "part-0000.parquet")
            if os.path.isfile(part_file) and part_file not in seen_paths:
                seen_paths.add(part_file)
                entries.append((date_str, part_file))
        if entries:
            break
    entries.sort(key=lambda x: x[0], reverse=True)
    return entries


def load_well_sensor_readings_from_local_channel(
    channel_dir: str,
    *,
    max_partition_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load sensor data from a SageMaker (or local) input channel directory.
    Supports partition folders or a single well_sensor_readings.parquet.
    """
    if max_partition_days is None:
        raw = os.getenv("ANOMALY_SENSOR_PARTITION_DAYS", "250").strip()
        max_partition_days = int(raw) if raw.isdigit() else 250

    legacy_candidates = [
        os.path.join(channel_dir, LEGACY_SENSOR_FILENAME),
        os.path.join(channel_dir, SENSOR_SUBDIR, LEGACY_SENSOR_FILENAME),
    ]
    partition_paths = _collect_local_partition_paths(channel_dir)

    if partition_paths:
        if max_partition_days is not None and max_partition_days > 0:
            seen: List[str] = []
            trimmed: List[Tuple[str, str]] = []
            for date_str, path in partition_paths:
                if date_str not in seen:
                    seen.append(date_str)
                if len(seen) > max_partition_days:
                    break
                trimmed.append((date_str, path))
            partition_paths = trimmed
        frames = [_read_parquet_local(p) for _, p in partition_paths]
        out = pd.concat(frames, ignore_index=True)
        logger.info(
            "Loaded %s rows from %s local partition file(s) under %s",
            len(out),
            len(frames),
            channel_dir,
        )
        return out

    for legacy_path in legacy_candidates:
        if os.path.isfile(legacy_path):
            logger.info("Loading legacy channel parquet %s", legacy_path)
            return _read_parquet_local(legacy_path)

    candidates = [
        os.path.join(channel_dir, f)
        for f in os.listdir(channel_dir)
        if f.endswith(".parquet")
    ]
    if len(candidates) == 1:
        logger.info("Using single parquet in channel: %s", candidates[0])
        return _read_parquet_local(candidates[0])

    raise FileNotFoundError(
        f"No sensor data in channel {channel_dir}: "
        f"expected {LEGACY_SENSOR_FILENAME} or {SENSOR_SUBDIR}/date=*/part-0000.parquet"
    )


def resolve_training_sensor_data(
    *,
    bucket: Optional[str] = None,
    organization_id: Optional[str] = None,
    parquet_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Resolve training input: explicit parquet URI/path, or org partition layout on S3.
    """
    override = (parquet_override or os.getenv("ANOMALY_SENSOR_PARQUET_S3", "")).strip()
    if override:
        logger.info("Using ANOMALY_SENSOR_PARQUET_S3 override: %s", override)
        return load_well_sensor_readings_from_parquet(override)

    bucket = (bucket or os.getenv("S3_BUCKET", "")).strip()
    org = (organization_id or os.getenv("ORGANIZATION_ID", "")).strip()
    if not bucket or not org:
        raise ValueError(
            "Set S3_BUCKET and ORGANIZATION_ID, or ANOMALY_SENSOR_PARQUET_S3."
        )

    prefer = os.getenv("ANOMALY_SENSOR_PREFER_PARTITIONS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    uri = legacy_sensor_parquet_uri(bucket, org)
    logger.info(
        "Loading sensor data for org %s (partitions under %s; legacy %s)",
        org,
        partition_prefix(org),
        uri,
    )
    return load_well_sensor_readings_from_s3(
        bucket, org, prefer_partitions=prefer
    )
