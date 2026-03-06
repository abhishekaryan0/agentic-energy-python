#!/usr/bin/env python3
"""
Pack anomaly_review per-well models into a tarball and upload to S3 for SageMaker.

Usage:
  python scripts/pack_and_upload_review_models.py --bucket MY_BUCKET --prefix models/review
  python scripts/pack_and_upload_review_models.py --bucket MY_BUCKET --prefix models/review --version v1

Creates: s3://BUCKET/PREFIX/review_models_VERSION.tar.gz
SageMaker extracts to model_dir so that model_dir/Well_001/, model_dir/Well_002/ exist.

Expects: models/ to contain per-well subdirs (e.g. models/Well_001/, models/Well_002/).
Excludes models/anomaly_detection/ (used by the anomaly_detection endpoint).
"""

import argparse
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Pack anomaly review per-well models and upload to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="models/review", help="S3 key prefix")
    parser.add_argument("--version", default="v1", help="Version tag for tarball")
    parser.add_argument("--models-dir", default="models", help="Local models directory")
    parser.add_argument("--dry-run", action="store_true", help="Only pack, do not upload")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, args.models_dir)
    if not os.path.isdir(models_dir):
        print(f"ERROR: {models_dir} not found.")
        return 1

    # Per-well dirs only (exclude anomaly_detection)
    well_dirs = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d)) and d != "anomaly_detection"
    ]
    tarball_name = f"review_models_{args.version}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tarball_path = os.path.join(tmp, tarball_name)
        if not well_dirs:
            print("WARN: No per-well subdirs under models/. Creating minimal tarball for testing.")
            placeholder = os.path.join(tmp, "_placeholder")
            os.makedirs(placeholder, exist_ok=True)
            cmd = ["tar", "-czvf", tarball_path, "-C", tmp, "_placeholder"]
        else:
            cmd = ["tar", "-czvf", tarball_path, "-C", models_dir] + well_dirs
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
