"""S3 parquet writers for transformation outputs."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from transformations.mappers import normalize_batch
from transformations.schemas import TableExportSpec

logger = logging.getLogger(__name__)


def build_s3_key(prefix: str, filename: str) -> str:
    """Join an S3 prefix and filename without introducing duplicate slashes."""
    return f"{prefix.strip().strip('/')}/{filename}"


def partition_s3_key(prefix: str, subdir: str, partition_date: str) -> str:
    return build_s3_key(prefix, f"{subdir}/date={partition_date}/part-0000.parquet")


def get_s3_client(region_name: Optional[str] = None):
    import boto3

    kwargs = {}
    if region_name:
        kwargs["region_name"] = region_name
    return boto3.client("s3", **kwargs)


def read_s3_json(
    bucket: str,
    key: str,
    *,
    region_name: Optional[str] = None,
) -> Optional[dict]:
    s3 = get_s3_client(region_name)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        err = str(exc).lower()
        if "404" in err or "not found" in err or "nosuchkey" in err:
            return None
        raise


def write_s3_json(
    bucket: str,
    key: str,
    payload: dict,
    *,
    region_name: Optional[str] = None,
) -> str:
    s3 = get_s3_client(region_name)
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return f"s3://{bucket}/{key}"


def read_parquet_from_s3(
    bucket: str,
    key: str,
    *,
    region_name: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    s3 = get_s3_client(region_name)
    try:
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return pd.read_parquet(buf)
    except Exception as exc:
        err = str(exc).lower()
        if "404" in err or "not found" in err or "nosuchkey" in err:
            return None
        raise


def _write_local_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression=compression)


def upload_parquet_file(
    local_path: str,
    bucket: str,
    key: str,
    *,
    region_name: Optional[str] = None,
) -> str:
    s3 = get_s3_client(region_name)
    with open(local_path, "rb") as file_obj:
        s3.upload_fileobj(file_obj, bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded %s", uri)
    return uri


def write_simple_parquet_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key: str,
    *,
    region_name: Optional[str] = None,
    compression: str = "snappy",
    write_empty: bool = True,
) -> Tuple[Optional[str], int]:
    """Write a dataframe to S3 parquet without TableExportSpec normalization."""
    if df is None or df.empty:
        if not write_empty:
            return None, 0
        df = pd.DataFrame()
    out = df.where(pd.notnull(df), None)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = tmp.name
    tmp.close()
    try:
        _write_local_parquet(out, tmp_path, compression=compression)
        uri = upload_parquet_file(tmp_path, bucket, key, region_name=region_name)
        return uri, len(out)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def write_dataframe_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key: str,
    spec: TableExportSpec,
    *,
    region_name: Optional[str] = None,
    compression: str = "snappy",
    write_empty: bool = True,
) -> Tuple[Optional[str], int]:
    """Write a single dataframe to S3 as parquet."""
    if df is None or df.empty:
        if not write_empty:
            return None, 0
        df = pd.DataFrame(columns=list(spec.columns))

    df = normalize_batch(df, spec)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = tmp.name
    tmp.close()
    try:
        _write_local_parquet(df, tmp_path, compression=compression)
        uri = upload_parquet_file(tmp_path, bucket, key, region_name=region_name)
        return uri, len(df)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def write_parquet_batches_to_s3(
    batches: Iterable[pd.DataFrame],
    bucket: str,
    key: str,
    spec: TableExportSpec,
    *,
    region_name: Optional[str] = None,
    compression: str = "snappy",
    write_empty: bool = True,
) -> Tuple[Optional[str], int]:
    """
    Write DataFrame batches as one parquet object in S3.

    Returns (s3_uri, row_count). If there are no rows and write_empty is False,
    no object is uploaded and s3_uri is None.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = tmp.name
    tmp.close()

    writer = None
    row_count = 0
    try:
        for batch in batches:
            if batch is None or batch.empty:
                continue
            batch = normalize_batch(batch, spec)
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression=compression)
            writer.write_table(table)
            row_count += len(batch)

        if writer is not None:
            writer.close()
            writer = None
        elif write_empty:
            empty = pd.DataFrame(columns=list(spec.columns))
            _write_local_parquet(empty, tmp_path, compression=compression)
        else:
            return None, 0

        uri = upload_parquet_file(tmp_path, bucket, key, region_name=region_name)
        return uri, row_count
    finally:
        if writer is not None:
            writer.close()
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def merge_snapshot(
    new_df: pd.DataFrame,
    bucket: str,
    prefix: str,
    spec: TableExportSpec,
    *,
    region_name: Optional[str] = None,
    write_empty: bool = True,
) -> Tuple[Optional[str], int]:
    """Merge new rows into the canonical snapshot parquet (dedupe by spec.dedupe_keys)."""
    key = build_s3_key(prefix, spec.output_filename)
    existing = read_parquet_from_s3(bucket, key, region_name=region_name)
    if existing is not None and not existing.empty:
        existing.columns = [str(c).lower() for c in existing.columns]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    if combined.empty:
        return write_dataframe_to_s3(
            combined,
            bucket,
            key,
            spec,
            region_name=region_name,
            write_empty=write_empty,
        )

    keys = list(spec.dedupe_keys)
    combined = normalize_batch(combined, spec)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    if spec.order_by:
        combined = combined.sort_values(list(spec.order_by))
    return write_dataframe_to_s3(
        combined,
        bucket,
        key,
        spec,
        region_name=region_name,
        write_empty=write_empty,
    )
