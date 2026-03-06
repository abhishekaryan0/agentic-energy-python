#!/usr/bin/env python3
"""
Sync local trained models to S3 so each model type lives in its own S3 directory.
Run after training so S3 is the store for trained artifacts; detection (SageMaker) uses
the tarball you upload via pack_and_upload_* (which can be run after this sync).

Usage:
  export S3_BUCKET=my-models-bucket
  python scripts/sync_models_to_s3.py
  python scripts/sync_models_to_s3.py --bucket my-bucket --dry-run

S3 layout (prefixes under bucket):
  models/anomaly_detection/  <- sync of local models/anomaly_detection/
  models/review/             <- sync of local models/ (per-well dirs only, exclude anomaly_detection)
  models/operations/         <- sync of local models/*.joblib (category_classifier, decision_tree_*)
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Sync local models to S3 (per-module directories)")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"), help="S3 bucket (or set S3_BUCKET)")
    parser.add_argument("--prefix", default="models", help="S3 key prefix under bucket")
    parser.add_argument("--dry-run", action="store_true", help="Only print aws s3 sync commands")
    args = parser.parse_args()
    if not args.bucket:
        print("ERROR: Set --bucket or S3_BUCKET")
        return 1

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_models = os.path.join(project_root, "models")
    if not os.path.isdir(local_models):
        print(f"ERROR: {local_models} not found.")
        return 1

    prefix = args.prefix.rstrip("/")
    ok = True

    # 1) Anomaly detection: models/anomaly_detection/ -> s3://bucket/models/anomaly_detection/
    anomaly_dir = os.path.join(local_models, "anomaly_detection")
    if os.path.isdir(anomaly_dir):
        s3_uri = f"s3://{args.bucket}/{prefix}/anomaly_detection/"
        cmd = ["aws", "s3", "sync", anomaly_dir, s3_uri, "--delete"]
        if args.dry_run:
            print(f"DRY-RUN: {' '.join(cmd)}")
        else:
            r = subprocess.run(cmd, cwd=project_root)
            if r.returncode != 0:
                ok = False
            else:
                print(f"Synced anomaly_detection -> {s3_uri}")
    else:
        print(f"Skip (not found): {anomaly_dir}")

    # 2) Review: per-well dirs under models/ (exclude anomaly_detection) -> s3://bucket/models/review/
    well_dirs = [
        d for d in os.listdir(local_models)
        if os.path.isdir(os.path.join(local_models, d)) and d != "anomaly_detection"
    ]
    if well_dirs:
        # Sync each well dir to models/review/Well_001/, etc.
        review_prefix = f"{prefix}/review"
        for d in well_dirs:
            local_path = os.path.join(local_models, d)
            s3_uri = f"s3://{args.bucket}/{review_prefix}/{d}/"
            cmd = ["aws", "s3", "sync", local_path, s3_uri, "--delete"]
            if args.dry_run:
                print(f"DRY-RUN: {' '.join(cmd)}")
            else:
                r = subprocess.run(cmd, cwd=project_root)
                if r.returncode != 0:
                    ok = False
        if not args.dry_run and well_dirs:
            print(f"Synced review ({len(well_dirs)} well dirs) -> s3://{args.bucket}/{review_prefix}/")
    else:
        print("Skip review: no per-well dirs under models/")

    # 3) Operations: models/*.joblib -> s3://bucket/models/operations/
    ops_prefix = f"{prefix}/operations"
    joblib_files = [f for f in os.listdir(local_models) if f.endswith(".joblib")]
    ops_wanted = [f for f in joblib_files if f == "category_classifier.joblib" or f.startswith("decision_tree")]
    if ops_wanted:
        for f in ops_wanted:
            local_path = os.path.join(local_models, f)
            s3_uri = f"s3://{args.bucket}/{ops_prefix}/{f}"
            cmd = ["aws", "s3", "cp", local_path, s3_uri]
            if args.dry_run:
                print(f"DRY-RUN: {' '.join(cmd)}")
            else:
                r = subprocess.run(cmd, cwd=project_root)
                if r.returncode != 0:
                    ok = False
        if not args.dry_run:
            print(f"Synced operations ({len(ops_wanted)} files) -> s3://{args.bucket}/{ops_prefix}/")
    else:
        print("Skip operations: no category_classifier.joblib or decision_tree*.joblib under models/")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
