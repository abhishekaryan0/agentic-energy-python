#!/usr/bin/env python3
"""Pack models/anomaly_detection/ and upload to S3."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def pack_and_upload(
    *,
    bucket: str,
    prefix: str = "models/anomaly",
    version: str = "v1",
    models_dir: str | None = None,
    job_name: str | None = None,
    org_id: str | None = None,
) -> str:
    import boto3

    root = Path(models_dir or _PROJECT_ROOT / "models")
    src = root / "anomaly_detection"
    if not src.is_dir():
        raise FileNotFoundError(f"Missing trained models directory: {src}")

    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(str(src), arcname="anomaly_detection")

        s3 = boto3.client("s3")
        if job_name and org_id:
            key = f"{org_id}/model/{job_name}/output/model.tar.gz"
        else:
            key = f"{prefix.rstrip('/')}/anomaly_detection_{version}.tar.gz"

        s3.upload_file(tar_path, bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info("Uploaded anomaly models: %s", uri)
        return uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack and upload anomaly detection models")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="models/anomaly")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--org-id", default="")
    args = parser.parse_args()

    org = args.org_id or (os.getenv("ORGANIZATION_ID") or "").strip()
    job = args.job_name or (os.getenv("TRAINING_JOB_NAME") or "").strip() or None
    pack_and_upload(
        bucket=args.bucket,
        prefix=args.prefix,
        version=args.version,
        models_dir=args.models_dir or None,
        job_name=job,
        org_id=org or None,
    )


if __name__ == "__main__":
    main()
