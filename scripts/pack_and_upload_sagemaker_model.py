#!/usr/bin/env python3
"""
Pack anomaly_detection models into a tarball and upload to S3 for SageMaker.

Usage:
  python scripts/pack_and_upload_sagemaker_model.py --bucket MY_BUCKET --prefix models/anomaly
  python scripts/pack_and_upload_sagemaker_model.py --bucket MY_BUCKET --prefix models/anomaly --version v2

Creates: s3://BUCKET/PREFIX/anomaly_detection_VERSION.tar.gz
SageMaker model data URL: s3://BUCKET/PREFIX/anomaly_detection_VERSION.tar.gz

Expects: models/anomaly_detection/wells/ and models/anomaly_detection/well_types/ to exist
(local layout after train_all_models).
"""

import argparse
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Pack anomaly models and upload to S3 for SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="models", help="S3 key prefix (default: models)")
    parser.add_argument("--version", default="v1", help="Version tag for tarball (default: v1)")
    parser.add_argument("--models-dir", default="models", help="Local models directory (default: models)")
    parser.add_argument("--dry-run", action="store_true", help="Only pack, do not upload")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, args.models_dir)
    anomaly_dir = os.path.join(models_dir, "anomaly_detection")
    if not os.path.isdir(anomaly_dir):
        print(f"ERROR: {anomaly_dir} not found. Run training first: python -m anomaly_detection.train_models")
        return 1

    tarball_name = f"anomaly_detection_{args.version}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tarball_path = os.path.join(tmp, tarball_name)
        # Create tarball with anomaly_detection/ at top level (so model_dir/anomaly_detection/wells/...)
        cmd = [
            "tar", "-czvf", tarball_path,
            "-C", models_dir,
            "anomaly_detection",
        ]
        print(f"Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=project_root)
        if r.returncode != 0:
            print("ERROR: tar failed")
            return 1
        if not os.path.isfile(tarball_path):
            print("ERROR: tarball not created")
            return 1
        print(f"Created {tarball_path} ({os.path.getsize(tarball_path) / (1024*1024):.2f} MB)")

        if args.dry_run:
            print("Dry run: not uploading")
            return 0

        prefix = args.prefix.rstrip("/")
        s3_uri = f"s3://{args.bucket}/{prefix}/{tarball_name}"
        cmd_upload = ["aws", "s3", "cp", tarball_path, s3_uri]
        print(f"Uploading: {' '.join(cmd_upload)}")
        r = subprocess.run(cmd_upload)
        if r.returncode != 0:
            print("ERROR: upload failed")
            return 1
        print(f"Uploaded to {s3_uri}")
        print(f"SageMaker model data URL: {s3_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
