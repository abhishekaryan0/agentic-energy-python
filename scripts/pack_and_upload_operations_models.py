#!/usr/bin/env python3
"""
Pack operational recommendation models into a tarball and upload to S3 for SageMaker.

Usage:
  python scripts/pack_and_upload_operations_models.py --bucket MY_BUCKET --prefix models/operations
  python scripts/pack_and_upload_operations_models.py --bucket MY_BUCKET --prefix models/operations --version v1

Creates: s3://BUCKET/PREFIX/operations_models_VERSION.tar.gz
SageMaker extracts to model_dir so that model_dir/category_classifier.joblib, model_dir/decision_tree_*.joblib exist.

Expects under models/: category_classifier.joblib, decision_tree_alldata.joblib (and optional decision_tree_rod_pump.joblib, etc.).
Run training first: python -m operations.train_models train_category && python -m operations.train_models train_action
"""

import argparse
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Pack operations models and upload to S3 for SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="models/operations", help="S3 key prefix")
    parser.add_argument("--version", default="v1", help="Version tag for tarball")
    parser.add_argument("--models-dir", default="models", help="Local models directory")
    parser.add_argument("--dry-run", action="store_true", help="Only pack, do not upload")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, args.models_dir)
    if not os.path.isdir(models_dir):
        print(f"ERROR: {models_dir} not found.")
        return 1

    # Operations-related joblib files only
    wanted = ["category_classifier.joblib"]
    all_joblib = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    decision_tree = [f for f in all_joblib if f.startswith("decision_tree")]
    wanted.extend(decision_tree)
    existing = [f for f in wanted if os.path.isfile(os.path.join(models_dir, f))]
    if not existing:
        print("WARN: No operations model files found (category_classifier.joblib or decision_tree*.joblib).")
        print("Run: python -m operations.train_models train_category && python -m operations.train_models train_action")

    tarball_name = f"operations_models_{args.version}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tarball_path = os.path.join(tmp, tarball_name)
        if not existing:
            # Minimal placeholder so tar has one member
            placeholder = os.path.join(tmp, ".keep_operations")
            open(placeholder, "w").close()
            cmd = ["tar", "-czvf", tarball_path, "-C", tmp, ".keep_operations"]
        else:
            cmd = ["tar", "-czvf", tarball_path, "-C", models_dir] + existing
        print(f"Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=project_root)
        if r.returncode != 0:
            print("ERROR: tar failed")
            return 1
        if not os.path.isfile(tarball_path):
            print("ERROR: tarball not created")
            return 1
        print(f"Created {tarball_path} ({os.path.getsize(tarball_path) / 1024:.2f} KB)")

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
