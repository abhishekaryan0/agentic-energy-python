"""
Batch runner for anomaly review.

Processes multiple wells in parallel using ThreadPoolExecutor.

Usage:
    python -m anomaly_review.run
    
Configuration is done via BatchConfig in config.py.
"""

import os
import sys
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from dotenv import load_dotenv

from anomaly_review.config import BatchConfig, DEFAULT_BATCH_CONFIG
from anomaly_review.data import get_snowflake_conn
from anomaly_review.service import detect_anomalies

load_dotenv()

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_execution.log"),
    ],
)

logger = logging.getLogger("BatchRunner")


# =============================================================================
# BATCH CONFIGURATION
# =============================================================================

# Use default config or override
BATCH_CONFIG = DEFAULT_BATCH_CONFIG


# =============================================================================
# FETCH WELLS
# =============================================================================

def get_all_active_wells() -> List[str]:
    """Fetch list of wells with data in last 7 days."""
    try:
        conn = get_snowflake_conn()
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT well_id
            FROM well_sensor_readings
            WHERE timestamp >= DATEADD(day, -7, CURRENT_DATE())
            ORDER BY well_id
        """

        cursor.execute(query)
        wells = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return wells

    except Exception as e:
        logger.error(f"Failed to fetch well list: {e}")
        return []


# =============================================================================
# PROCESS SINGLE WELL
# =============================================================================

def process_single_well(well_id: str) -> Dict:
    """Wrapper function for parallel execution."""
    try:
        results = detect_anomalies(well_id)

        return {
            "well_id": well_id,
            "count": len(results),
            "titles": [r.get("title") for r in results],
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error processing {well_id}: {e}")
        return {
            "well_id": well_id,
            "count": 0,
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# RUN BATCH
# =============================================================================

def run_batch(config=None):
    """
    Run batch anomaly detection on all active wells.
    
    Args:
        config: BatchConfig instance (uses default if None)
    """
    if config is None:
        config = BATCH_CONFIG
    
    logger.info("=" * 60)
    logger.info("STARTING ANOMALY REVIEW BATCH (PARALLEL)")
    logger.info("=" * 60)

    start_time = time.time()

    # Fetch wells
    wells = get_all_active_wells()

    if not wells:
        logger.error("No active wells found. Exiting.")
        return

    # Test mode logic
    if config.test_mode:
        logger.warning("⚠️  TEST MODE ENABLED")

        if config.test_well_id:
            logger.warning(f"Running for specific well_id: {config.test_well_id}")
            wells = [config.test_well_id]
        else:
            logger.warning(f"Processing only first {config.test_well_limit} wells")
            wells = wells[:config.test_well_limit]

    logger.info(f"Total wells to process: {len(wells)}")

    logger.info("-" * 30)
    logger.info("Target Wells:")
    for w in wells:
        logger.info(f" -> {w}")
    logger.info("-" * 30)

    stats = {
        "processed": 0,
        "failed": 0,
        "anomalies_found": 0,
    }

    # Parallel processing
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_well = {
            executor.submit(process_single_well, well): well
            for well in wells
        }
 
        for i, future in enumerate(as_completed(future_to_well)):
            result = future.result()
 
            sys.stdout.write(f"\rProgress: {i + 1}/{len(wells)}")
            sys.stdout.flush()
 
            if result["status"] == "success":
                stats["processed"] += 1
                stats["anomalies_found"] += result["count"]
 
                if result["count"] > 0:
                    sys.stdout.write("\n")
                    logger.info(
                        f"[+] {result['well_id']}: "
                        f"{result['count']} anomalies -> {result['titles']}"
                    )
            else:
                stats["failed"] += 1
                sys.stdout.write("\n")
                logger.error(
                    f"[-] {result['well_id']} failed: {result.get('error')}"
                )
 
    total_time = time.time() - start_time
 
    logger.info("\n" + "=" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"Execution Time     : {total_time:.2f}s")
    logger.info(f"Wells Processed   : {stats['processed']}")
    logger.info(f"Wells Failed      : {stats['failed']}")
    logger.info(f"Anomalies Saved   : {stats['anomalies_found']}")
    logger.info("=" * 60)
 
 
# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    run_batch()
 