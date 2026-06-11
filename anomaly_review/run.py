#!/usr/bin/env python3
"""
Weekly anomaly review detection (S3 only).

First run (no checkpoint or ANOMALY_REVIEW_FORCE_WINDOW=1):
  - Process wells with sensor data in the last ANOMALY_REVIEW_INITIAL_DAYS (default 180).

Weekly incremental runs:
  - Process wells with new partitions since last_processed_date + 1
  - Each well still receives ANOMALY_REVIEW_FETCH_DAYS of history for baselines/ML

Inputs:
  - S3_BUCKET, ORGANIZATION_ID
  - Trained review model on S3 (auto-detected review-train-*/output/model.tar.gz)

Checkpoint:
  s3://BUCKET/{ORG}/detections/anomaly_review/checkpoints/last_processed.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from dotenv import load_dotenv

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from anomaly_review.partition_data import (
    build_well_dataframe,
    load_partitions,
    max_partition_date,
    wells_with_data_in_range,
)
from anomaly_review.service import detect_anomalies, save_reviews

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
    override = (os.getenv("ANOMALY_REVIEW_CHECKPOINT_S3_KEY") or "").strip()
    if override:
        return override.lstrip("/")
    return f"{org}/detections/anomaly_review/checkpoints/last_processed.json"


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
    payload = {
        "updated_at": last_checked_ts.isoformat(),
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


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "y")


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or str(default)).strip()
    return int(raw) if raw.isdigit() else default


def _ensure_review_model_on_s3() -> None:
    """Trigger model sync from S3 before parallel workers start."""
    from anomaly_review.models import _ensure_review_models_from_s3

    _ensure_review_models_from_s3()


def _process_well(
    well_id: str,
    sensor_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    fetch_days: int,
) -> List[Dict]:
    well_df = build_well_dataframe(
        sensor_df, prod_df, well_id, fetch_days=fetch_days
    )
    if well_df.empty or len(well_df) < 5:
        return []
    return detect_anomalies(well_id, df=well_df, save_to_db=False)


def main() -> None:
    cfg = _env()
    s3 = boto3.client("s3", region_name=cfg.region)

    fetch_days = _int_env("ANOMALY_REVIEW_FETCH_DAYS", 180)
    initial_days = _int_env("ANOMALY_REVIEW_INITIAL_DAYS", 180)
    _int_env("ANOMALY_REVIEW_DETECTION_DAYS", 7)  # reserved for future window tuning
    max_workers = max(1, _int_env("ANOMALY_REVIEW_MAX_WORKERS", 5))
    force_window = _truthy("ANOMALY_REVIEW_FORCE_WINDOW")

    ck_key = _checkpoint_key(cfg.org)
    ck = _load_checkpoint(s3, cfg.bucket, ck_key)
    last_processed_date = _parse_date(ck.get("last_processed_date"))

    today = datetime.now(timezone.utc).date()
    if last_processed_date is None or force_window:
        incr_start = today - timedelta(days=initial_days)
        run_mode = "initial"
    else:
        incr_start = last_processed_date + timedelta(days=1)
        run_mode = "incremental"

    end_date = today
    data_start = end_date - timedelta(days=fetch_days)

    logger.info(
        "Anomaly review detection (%s): incr=%s → %s, fetch_history=%s days, checkpoint=%s",
        run_mode,
        incr_start.isoformat(),
        end_date.isoformat(),
        fetch_days,
        last_processed_date.isoformat() if last_processed_date else "none",
    )

    sensor_df, sensor_keys = load_partitions(
        s3,
        bucket=cfg.bucket,
        org=cfg.org,
        subdir="well_sensor_readings",
        start=data_start,
        end=end_date,
    )
    prod_df, prod_keys = load_partitions(
        s3,
        bucket=cfg.bucket,
        org=cfg.org,
        subdir="daily_production",
        start=data_start,
        end=end_date,
    )

    logger.info(
        "Loaded partitions: sensor_files=%s prod_files=%s (sensor_rows=%s prod_rows=%s)",
        len(sensor_keys),
        len(prod_keys),
        len(sensor_df),
        len(prod_df),
    )

    if sensor_df.empty:
        logger.info("No sensor partition data in fetch window; updating checkpoint only.")
        _save_checkpoint(
            s3,
            bucket=cfg.bucket,
            key=ck_key,
            last_checked_ts=datetime.now(timezone.utc),
            last_processed_date=last_processed_date,
        )
        return

    wells = wells_with_data_in_range(
        sensor_df, start=incr_start, end=end_date
    )
    if not wells:
        logger.info("No wells with sensor data in incremental window %s → %s", incr_start, end_date)
        _save_checkpoint(
            s3,
            bucket=cfg.bucket,
            key=ck_key,
            last_checked_ts=datetime.now(timezone.utc),
            last_processed_date=last_processed_date,
        )
        return

    logger.info("Wells to process: %s (mode=%s)", len(wells), run_mode)

    _ensure_review_model_on_s3()

    all_reviews: List[Dict] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_well, w, sensor_df, prod_df, fetch_days): w
            for w in wells
        }
        for i, future in enumerate(as_completed(futures), start=1):
            well_id = futures[future]
            try:
                results = future.result()
                all_reviews.extend(results)
                if results:
                    logger.info("[%s/%s] %s: %s findings", i, len(wells), well_id, len(results))
            except Exception as exc:
                failed += 1
                logger.error("[%s/%s] %s failed: %s", i, len(wells), well_id, exc)

    logger.info(
        "Detection complete: wells=%s failed=%s findings=%s",
        len(wells),
        failed,
        len(all_reviews),
    )

    if all_reviews:
        save_reviews(all_reviews)

    new_last = max_partition_date(sensor_keys) or end_date
    if run_mode == "incremental" and incr_start > end_date:
        new_last = last_processed_date

    _save_checkpoint(
        s3,
        bucket=cfg.bucket,
        key=ck_key,
        last_checked_ts=datetime.now(timezone.utc),
        last_processed_date=new_last,
    )
    logger.info("Checkpoint updated: last_processed_date=%s", new_last.isoformat())


if __name__ == "__main__":
    main()
