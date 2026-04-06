#!/usr/bin/env python3
"""
Unified add-on training script
==============================

Runs training for anomaly_detection, anomaly_review, and operations **only when
new data** has been inserted in the last N hours (default 24). Checks both
well_sensor_readings and well_daily_production so training runs when either
sensors or production are updated (all three modules use one or both tables).

Usage:
    # Run all three modules when new data exists (default last 24h)
    python automate/run_training.py

    # Only check last 12 hours for new data
    python automate/run_training.py --hours 12

    # Run only specific module(s)
    python automate/run_training.py --anomaly-detection
    python automate/run_training.py --anomaly-review --operations

    # Dry run: only report if new data exists, do not train
    python automate/run_training.py --dry-run

    # Limit anomaly_review to first N wells (e.g. for testing)
    python automate/run_training.py --limit 5

Schedule (e.g. cron every 24h):
    0 2 * * * cd /path/to/repo && python automate/run_training.py
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RunTraining")

# Tables used by the three modules: trigger training when either has new data
SENSOR_TABLE = "well_sensor_readings"
PRODUCTION_TABLE = "well_daily_production"


def get_snowflake_connection():
    """Create Snowflake connection (shared env: SNOWFLAKE_*)."""
    try:
        from snowflake.connector import connect as snowflake_connect
    except ImportError:
        raise RuntimeError("Snowflake connector not installed")
    cfg = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
    }
    missing = [k for k in cfg if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake env: {missing}")
    return snowflake_connect(**cfg)


def has_new_training_data(conn, hours: int = 24):
    """
    Check if any new rows exist in sensor or production tables in the last `hours`.
    All three modules train on well_sensor_readings; anomaly_review and operations
    also use well_daily_production. Trigger when either table has new data.
    Returns (has_new_data: bool, list of well_id with new data for anomaly_review).
    """
    cur = conn.cursor()
    try:
        # Wells with new sensor readings (timestamp in last N hours)
        cur.execute(
            f"""
            SELECT DISTINCT well_id
            FROM {SENSOR_TABLE}
            WHERE timestamp >= DATEADD(hour, -%s, CURRENT_TIMESTAMP())
            ORDER BY well_id
            """,
            (hours,),
        )
        sensor_wells = [row[0] for row in cur.fetchall() if row[0]]

        # Wells with new production (date in last N hours; production is daily so use days)
        days_back = max(1, (hours + 23) // 24)
        cur.execute(
            f"""
            SELECT DISTINCT well_id
            FROM {PRODUCTION_TABLE}
            WHERE date >= DATEADD(day, -%s, CURRENT_DATE())
            ORDER BY well_id
            """,
            (days_back,),
        )
        production_wells = [row[0] for row in cur.fetchall() if row[0]]

        # Union: any well with new sensors or new production
        well_ids = sorted(set(sensor_wells) | set(production_wells))
        has_new = len(well_ids) > 0
        if has_new:
            logger.info(
                "New data: %s well(s) with sensor data, %s well(s) with production data, %s total.",
                len(sensor_wells), len(production_wells), len(well_ids),
            )
        return (has_new, well_ids)
    finally:
        cur.close()


def run_anomaly_detection_training():
    """Run anomaly_detection full retrain (per well-type models). Uses full data including new."""
    logger.info("[anomaly_detection] Starting add-on training (full retrain with new data)...")
    t0 = time.time()
    try:
        from anomaly_detection.train_models import train_all_models
        train_all_models()
        logger.info("[anomaly_detection] Training finished in %.1fs", time.time() - t0)
        return True
    except Exception as e:
        logger.error("[anomaly_detection] Training failed: %s", e, exc_info=True)
        return False


def run_anomaly_review_training(well_ids, force_retrain: bool = True, workers: int = 2, limit=None):
    """
    Train anomaly_review per-well models only for wells that have new data.
    Uses full history (e.g. 180 days) so new data is included; force_retrain=True
    so we always retrain for those wells (add-on).
    """
    if limit is not None:
        well_ids = well_ids[:limit]
    if not well_ids:
        return True
    logger.info("[anomaly_review] Starting add-on training for %s wells (workers=%s)...", len(well_ids), workers)
    t0 = time.time()
    from anomaly_review.data import fetch_well_data
    from anomaly_review.models import get_or_train_model

    trained, skipped, failed = 0, 0, 0

    def train_one(well_id):
        try:
            df = fetch_well_data(well_id, days=180)
            if df.empty or len(df) < 60:
                return ("skip", well_id)
            get_or_train_model(well_id, df, force_retrain=force_retrain)
            return ("ok", well_id)
        except Exception as e:
            logger.warning("[anomaly_review] %s: %s", well_id, e)
            return ("fail", well_id)

    if workers <= 1:
        for wid in well_ids:
            status, _ = train_one(wid)
            if status == "ok":
                trained += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(train_one, wid): wid for wid in well_ids}
            for future in as_completed(futures):
                status, wid = future.result()
                if status == "ok":
                    trained += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1

    logger.info(
        "[anomaly_review] Done in %.1fs: trained=%s skipped=%s failed=%s",
        time.time() - t0, trained, skipped, failed,
    )
    return failed == 0


def run_operations_training(conn=None):
    """Run operations training from S3/local parquet paths (S3-only; no DB)."""
    logger.info("[operations] Starting training from parquet...")
    t0 = time.time()
    try:
        import os
        from operations.train_models import train_operations_from_parquet
        sp = os.getenv("OPS_TRAIN_SENSOR_PARQUET", "").strip()
        pp = os.getenv("OPS_TRAIN_PRODUCTION_PARQUET", "").strip()
        out = os.getenv("OPERATIONS_MODELS_DIR", "models").strip() or "models"
        if not sp or not pp:
            logger.error(
                "[operations] Set OPS_TRAIN_SENSOR_PARQUET and OPS_TRAIN_PRODUCTION_PARQUET "
                "(e.g. s3://bucket/org/sensor_data/well_sensor_readings.parquet)"
            )
            return False
        train_operations_from_parquet(sp, pp, out)
        logger.info("[operations] Training finished in %.1fs", time.time() - t0)
        return True
    except Exception as e:
        logger.error("[operations] Training failed: %s", e, exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add-on training for anomaly_detection, anomaly_review, and operations when new data exists.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Consider data newer than this many hours as 'new' (default: 24)",
    )
    parser.add_argument(
        "--anomaly-detection",
        action="store_true",
        help="Run only anomaly_detection training",
    )
    parser.add_argument(
        "--anomaly-review",
        action="store_true",
        help="Run only anomaly_review training",
    )
    parser.add_argument(
        "--operations",
        action="store_true",
        help="Run only operations training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check for new data; do not run any training",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for anomaly_review per-well training (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of wells to train for anomaly_review (for testing)",
    )
    args = parser.parse_args()

    run_all = not (args.anomaly_detection or args.anomaly_review or args.operations)

    logger.info(
        "Add-on training check: last %s hours (tables: %s, %s)",
        args.hours, SENSOR_TABLE, PRODUCTION_TABLE,
    )
    conn = get_snowflake_connection()
    try:
        has_new, well_ids = has_new_training_data(conn, args.hours)

        if not has_new:
            logger.info(
                "No new data in %s or %s in the last %s hours. Skipping all training.",
                SENSOR_TABLE, PRODUCTION_TABLE, args.hours,
            )
            return 0

        logger.info(
            "New data found for %s well(s) (sensor and/or production). Running add-on training.",
            len(well_ids),
        )

        if args.dry_run:
            logger.info(
                "Dry run: would train anomaly_detection=%s anomaly_review=%s operations=%s",
                run_all or args.anomaly_detection,
                run_all or args.anomaly_review,
                run_all or args.operations,
            )
            return 0

        ok = True
        if run_all or args.anomaly_detection:
            ok = run_anomaly_detection_training() and ok
        if run_all or args.anomaly_review:
            ok = run_anomaly_review_training(
                well_ids,
                force_retrain=True,
                workers=args.workers,
                limit=args.limit,
            ) and ok
        if run_all or args.operations:
            ok = run_operations_training(conn) and ok

        return 0 if ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

