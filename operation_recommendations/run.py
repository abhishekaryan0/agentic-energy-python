#!/usr/bin/env python3
"""
Weekly operations recommendations detection (S3 only).

Goal:
  - Read ONLY the most recent window of partitioned S3 data (default last 7 days)
  - Generate recommendations using the trained operations model from S3
  - Write recommendations to S3
  - Maintain a checkpoint in S3 so weekly runs are incremental

Inputs (required):
  - S3_BUCKET
  - ORGANIZATION_ID
  - OPERATIONS_MODEL_TARBALL_S3_URI (optional; auto-detected from s3://BUCKET/ORG/model/operations-train-...)

S3 input layout (partitioned):
  - s3://BUCKET/{ORG}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet
  - s3://BUCKET/{ORG}/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet

Checkpoint:
  - s3://BUCKET/{ORG}/detections/operations/checkpoints/last_processed.json
    {
      "updated_at": "...",
      "last_checked_ts": "...",
      "last_processed_date": "YYYY-MM-DD"
    }
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
from dotenv import load_dotenv

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from operation_recommendations.recommendations.persistence import (
    recommend_from_metrics,
    write_operations_recommendations_to_s3,
)
from operation_recommendations.s3_inputs import download_s3_uri_to_temp, get_latest_invoices_parquet_uris
from operation_recommendations.train_models import build_metrics_from_parquet
from operation_recommendations.s3_models import resolve_operations_model_tarball_s3_uri
from operation_recommendations.eia_prices import get_brent_spot_usd_per_bbl

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Env:
    bucket: str
    org: str
    region: str


def _env() -> _Env:
    bucket = (os.getenv("S3_BUCKET") or "").strip()
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    region = (os.getenv("AWS_REGION") or "us-east-1").strip()
    if not bucket or not org:
        raise RuntimeError("S3_BUCKET and ORGANIZATION_ID must be set")
    return _Env(bucket=bucket, org=org, region=region)


def _checkpoint_key(org: str) -> str:
    override = (os.getenv("OPS_DETECTION_CHECKPOINT_S3_KEY") or "").strip()
    if override:
        return override.lstrip("/")
    return f"{org}/detections/operations/checkpoints/last_processed.json"


def _load_checkpoint(s3, bucket: str, key: str) -> Dict[str, Any]:
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _save_checkpoint(
    s3,
    *,
    bucket: str,
    key: str,
    last_checked_ts: datetime,
    last_processed_date: Optional[date],
) -> None:
    payload: Dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_checked_ts": last_checked_ts.isoformat(),
        "last_processed_date": last_processed_date.isoformat() if last_processed_date else None,
    }
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _partition_key(org: str, subdir: str, d: date) -> str:
    return f"{org}/sensor_data/{subdir}/date={d.isoformat()}/part-0000.parquet"


def _list_partition_keys_in_range(
    s3,
    *,
    bucket: str,
    org: str,
    subdir: str,
    start: date,
    end: date,
) -> List[str]:
    """
    List partition parquet keys under:
      {org}/sensor_data/{subdir}/date=YYYY-MM-DD/part-0000.parquet

    Only returns keys with date between [start, end].
    """
    prefix = f"{org}/sensor_data/{subdir}/date="
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key") or ""
            if not key.endswith("/part-0000.parquet"):
                continue
            try:
                part = key.split("date=", 1)[1].split("/", 1)[0]
                d = pd.to_datetime(part).date()
            except Exception:
                continue
            if start <= d <= end:
                keys.append(key)
    keys.sort()
    return keys


def _read_partition_df(s3, *, bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        return pd.read_parquet(buf)
    except Exception:
        return None


def _load_window_partitions(
    *,
    s3,
    bucket: str,
    org: str,
    subdir: str,
    start: date,
    end: date,
) -> Tuple[pd.DataFrame, List[str]]:
    keys = _list_partition_keys_in_range(
        s3,
        bucket=bucket,
        org=org,
        subdir=subdir,
        start=start,
        end=end,
    )
    frames: List[pd.DataFrame] = []
    for key in keys:
        df = _read_partition_df(s3, bucket=bucket, key=key)
        if df is None or df.empty:
            continue
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, keys


def _tmp_parquet(df: pd.DataFrame, name: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f"ops_{name}_", suffix=".parquet")
    os.close(fd)
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    cfg = _env()
    s3 = boto3.client("s3", region_name=cfg.region)

    window_days_raw = (os.getenv("OPS_DETECTION_DAYS") or "7").strip()
    window_days = int(window_days_raw) if window_days_raw.isdigit() else 7
    window_days = max(1, min(window_days, 31))
    force_window = (os.getenv("OPS_DETECTION_FORCE_WINDOW") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    ck_key = _checkpoint_key(cfg.org)
    reset_ck = (os.getenv("OPS_RESET_CHECKPOINT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    ck = {} if reset_ck else _load_checkpoint(s3, cfg.bucket, ck_key)
    last_processed_date = _parse_date(ck.get("last_processed_date"))
    if reset_ck:
        logger.info("OPS_RESET_CHECKPOINT=1 — treating as first run (no prior watermark)")

    today = datetime.now(timezone.utc).date()
    if force_window or last_processed_date is None:
        start_date = today - timedelta(days=window_days)
    else:
        start_date = last_processed_date + timedelta(days=1)
    end_date = today

    logger.info(
        "Operations detection window: %s → %s (checkpoint=%s)",
        start_date.isoformat(),
        end_date.isoformat(),
        last_processed_date.isoformat() if last_processed_date else "none",
    )

    sensor_df, sensor_keys = _load_window_partitions(
        s3=s3,
        bucket=cfg.bucket,
        org=cfg.org,
        subdir="well_sensor_readings",
        start=start_date,
        end=end_date,
    )
    prod_df, prod_keys = _load_window_partitions(
        s3=s3,
        bucket=cfg.bucket,
        org=cfg.org,
        subdir="daily_production",
        start=start_date,
        end=end_date,
    )

    logger.info(
        "Loaded partitions: sensor_files=%s prod_files=%s (sensor_rows=%s prod_rows=%s)",
        len(sensor_keys),
        len(prod_keys),
        len(sensor_df),
        len(prod_df),
    )

    if sensor_df.empty or prod_df.empty:
        logger.info(
            "No new partition data (sensor_rows=%s, prod_rows=%s). Updating checkpoint last_checked_ts only.",
            len(sensor_df),
            len(prod_df),
        )
        _save_checkpoint(
            s3,
            bucket=cfg.bucket,
            key=ck_key,
            last_checked_ts=datetime.now(timezone.utc),
            last_processed_date=last_processed_date,
        )
        return

    model_uri = resolve_operations_model_tarball_s3_uri()
    if not model_uri:
        raise RuntimeError(
            "No operations model on S3. Set OPERATIONS_MODEL_TARBALL_S3_URI or S3_BUCKET+ORGANIZATION_ID."
        )
    logger.info("Operations model tarball: %s", model_uri)

    invoice_uris = get_latest_invoices_parquet_uris(bucket=cfg.bucket, org=cfg.org)
    invoice_paths: List[str] = []
    if invoice_uris:
        logger.info("Financial invoices: %s", invoice_uris)
        for uri in invoice_uris:
            try:
                invoice_paths.append(download_s3_uri_to_temp(uri))
            except Exception as exc:
                logger.warning("Failed to download invoices %s: %s", uri, exc)
    else:
        logger.warning(
            "No invoices_v2/v1 under s3://%s/%s/financial_data/ — "
            "implementation_cost_usd will be 0 until financial model is trained",
            cfg.bucket,
            cfg.org,
        )

    try:
        eia_price, eia_src, eia_period = get_brent_spot_usd_per_bbl()
        logger.info(
            "EIA Brent prefetch: $%.2f/bbl source=%s period=%s",
            eia_price,
            eia_src,
            eia_period,
        )
    except Exception as exc:
        logger.warning("EIA Brent prefetch failed: %s", exc)

    sensor_path = _tmp_parquet(sensor_df, "sensor")
    prod_path = _tmp_parquet(prod_df, "production")
    try:
        all_metrics = build_metrics_from_parquet(
            sensor_path,
            prod_path,
            invoices_parquet_paths=invoice_paths or None,
        )
        logger.info("Built metrics for %s wells (invoices=%s)", len(all_metrics), len(invoice_paths))

        conf_raw = (os.getenv("OPERATIONS_CONFIDENCE_THRESHOLD") or "0.6").strip()
        confidence_threshold = float(conf_raw) if conf_raw else 0.6

        all_recs: List[Dict[str, Any]] = []
        for well_id, lift_type, metrics in all_metrics:
            try:
                recs = recommend_from_metrics(
                    str(well_id),
                    str(lift_type or "Unknown"),
                    metrics,
                    confidence_threshold=confidence_threshold,
                )
                for r in recs:
                    r["well_id"] = well_id
                    all_recs.append(r)
            except Exception as exc:
                logger.warning("Recommendation failed for well %s: %s", well_id, exc)

        logger.info("Generated %s recommendations", len(all_recs))
        out_key = write_operations_recommendations_to_s3(all_recs)
        logger.info("Wrote recommendations to %s", out_key or "(no output)")

        # Advance checkpoint to the latest partition date we actually had data for.
        processed_dates: List[date] = []
        for k in sensor_keys:
            try:
                part = k.split("date=", 1)[1].split("/", 1)[0]
                processed_dates.append(pd.to_datetime(part).date())
            except Exception:
                pass
        new_last = max(processed_dates) if processed_dates else end_date
        _save_checkpoint(
            s3,
            bucket=cfg.bucket,
            key=ck_key,
            last_checked_ts=datetime.now(timezone.utc),
            last_processed_date=new_last,
        )
        logger.info("Checkpoint updated: last_processed_date=%s", new_last.isoformat())
    finally:
        for p in (sensor_path, prod_path, *invoice_paths):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()

