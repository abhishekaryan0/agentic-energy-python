#!/usr/bin/env python3
"""
Train the financial implementation-cost model from org data on S3 and upload artifact.

Reads:
  - {ORG}/sensor_data/daily_production/... (partitions or legacy parquet)
  - {ORG}/financial_data/run_ts=*/invoices_v2.parquet (latest)

Writes locally (default):
  - models/financial_local_{timestamp}/financial_implementation_cost.joblib

Optional upload:
  - s3://{BUCKET}/{ORG}/model/{TRAINING_JOB_NAME}/output/model.tar.gz

Usage:
  source .env
  python -m financial.run
  python -m financial.run --upload
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _upload_tarball(local_model_dir: str, *, bucket: str, org: str, job_name: str) -> str:
    import boto3

    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(
                os.path.join(local_model_dir, "financial_implementation_cost.joblib"),
                arcname="financial_implementation_cost.joblib",
            )
        key = f"{org}/model/{job_name}/output/model.tar.gz"
        boto3.client("s3").upload_file(tar_path, bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info("Uploaded financial model tarball: %s", uri)
        return uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Train financial model from S3 parquet")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload model.tar.gz under {ORG}/model/{job}/output/",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Local output directory (default: models/financial_local_<ts>)",
    )
    parser.add_argument(
        "--job-name",
        default="",
        help="SageMaker-style job folder name (default: financial-train-YYYYMMDD-HHMM)",
    )
    args = parser.parse_args()

    bucket = (os.getenv("S3_BUCKET") or "").strip()
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        raise SystemExit("Set S3_BUCKET and ORGANIZATION_ID in .env")

    from financial.s3_inputs import (
        download_s3_uri_to_temp,
        get_latest_invoices_parquet_uris,
        load_daily_production_from_s3,
    )
    from financial.train_models import train_financial_from_parquet

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or str(_PROJECT_ROOT / "models" / f"financial_local_{ts}")
    job_name = args.job_name or f"financial-train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    logger.info("Loading daily production from S3 org=%s", org)
    prod_df = load_daily_production_from_s3(bucket=bucket, org=org)
    prod_path = os.path.join(tempfile.gettempdir(), f"fin_prod_{ts}.parquet")
    prod_df.to_parquet(prod_path, index=False)

    invoice_uris = get_latest_invoices_parquet_uris(bucket=bucket, org=org)
    if not invoice_uris:
        raise SystemExit(
            f"No invoices found under s3://{bucket}/{org}/financial_data/ — "
            "run synth export or generate_invoices first."
        )

    invoice_paths: list[str] = []
    try:
        for uri in invoice_uris:
            invoice_paths.append(download_s3_uri_to_temp(uri))

        logger.info(
            "Training financial model production_rows=%s invoice_files=%s -> %s",
            len(prod_df),
            len(invoice_paths),
            out_dir,
        )
        metrics = train_financial_from_parquet(prod_path, invoice_paths, out_dir)
        logger.info("Training complete: %s", metrics)

        if args.upload:
            _upload_tarball(out_dir, bucket=bucket, org=org, job_name=job_name)
    finally:
        for p in (prod_path, *invoice_paths):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
