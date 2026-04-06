#!/usr/bin/env python3
"""
Integrated Operational Recommendations Pipeline (S3 only: model load, input, output).
====================================================================================

1. Optionally trains models from S3 parquet (train_operations_from_parquet).
2. Builds metrics from S3 parquet (s3://BUCKET/{ORGANIZATION_ID}/sensor_data/).
3. Runs recommend_from_metrics (model loaded from S3 via get_models_dir()).
4. Writes recommendations to S3 only (write_operations_recommendations_to_s3).

Env: S3_BUCKET, ORGANIZATION_ID.
"""

import sys
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from operations.utils import get_models_dir, get_model_path
from operations.train_models import build_metrics_from_parquet, train_operations_from_parquet
from operations.recommendations.persistence import (
    recommend_from_metrics,
    write_operations_recommendations_to_s3,
)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_get_model_path = get_model_path


def _get_s3_parquet_paths() -> tuple:
    """Return (sensor_uri, production_uri) from S3_BUCKET and ORGANIZATION_ID."""
    bucket = os.getenv("S3_BUCKET", "").strip()
    org = os.getenv("ORGANIZATION_ID", "").strip()
    if not bucket or not org:
        raise ValueError("S3_BUCKET and ORGANIZATION_ID are required for run_pipeline (S3-only).")
    base = f"{org}/sensor_data"
    return (
        f"s3://{bucket}/{base}/well_sensor_readings.parquet",
        f"s3://{bucket}/{base}/daily_production.parquet",
    )


def _get_latest_invoices_parquet_paths() -> List[str]:
    """
    Return up to [invoices_v1.parquet, invoices_v2.parquet] from the latest
    `s3://{bucket}/{org}/financial_data/run_ts=*/` run.
    """
    bucket = os.getenv("S3_BUCKET", "").strip()
    org = os.getenv("ORGANIZATION_ID", "").strip()
    if not bucket or not org:
        return []

    import boto3

    s3 = boto3.client("s3")
    prefix = f"{org}/financial_data/run_ts="

    v1_keys: List[str] = []
    v2_keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj.get("Key") or ""
            if not k.endswith(".parquet"):
                continue
            if k.endswith("/invoices_v1.parquet"):
                v1_keys.append(k)
            elif k.endswith("/invoices_v2.parquet"):
                v2_keys.append(k)

    out: List[str] = []
    if v1_keys:
        v1_keys.sort(reverse=True)
        out.append(f"s3://{bucket}/{v1_keys[0]}")
    if v2_keys:
        v2_keys.sort(reverse=True)
        out.append(f"s3://{bucket}/{v2_keys[0]}")
    return out


def run_pipeline(skip_training: bool = False, confidence_threshold: float = 0.75) -> Dict[str, Any]:
    """
    Execute the operational recommendations pipeline (S3 only).

    - Model: loaded from S3 via get_models_dir() (OPERATIONS_MODEL_S3_URI or s3://BUCKET/{ORG}/model/).
    - Input: S3 parquet (sensor + production).
    - Output: S3 parquet (write_operations_recommendations_to_s3).
    """
    try:
        logger.info("=" * 80)
        logger.info("OPERATIONAL RECOMMENDATIONS PIPELINE (S3 only)")
        logger.info("=" * 80)
        logger.info("Start Time: %s", datetime.now(timezone.utc).isoformat())
        logger.info("Confidence Threshold: %.0f%%", confidence_threshold * 100)

        sensor_uri, prod_uri = _get_s3_parquet_paths()
        invoice_uris = _get_latest_invoices_parquet_paths()
        logger.info("S3 input: sensor=%s production=%s", sensor_uri, prod_uri)
        if invoice_uris:
            logger.info("S3 input: invoices=%s", invoice_uris)
        else:
            logger.warning("No invoices parquet found; implementation_cost_usd may be 0.")

        # Trigger model load from S3 (get_models_dir() calls _ensure_operations_models_from_s3)
        _ = get_models_dir()

        train_result = None
        if not skip_training:
            logger.info("\n1. Training models from S3 parquet...")
            cache_dir = os.environ.get("OPERATIONS_MODEL_CACHE_DIR") or os.path.join(_PROJECT_ROOT, "models", "operations_run")
            os.makedirs(cache_dir, exist_ok=True)
            train_result = train_operations_from_parquet(
                sensor_uri,
                prod_uri,
                cache_dir,
                model_path=os.path.join(cache_dir, "decision_tree_alldata.joblib"),
            )
            logger.info("Models trained (samples etc. in train_result)")
        else:
            logger.info("\n1. Skipping training (--skip-training); using existing model from S3/env.")

        logger.info("\n2. Building metrics from S3 parquet...")
        all_metrics = build_metrics_from_parquet(
            sensor_uri,
            prod_uri,
            invoices_parquet_paths=invoice_uris or None,
        )
        if not all_metrics:
            logger.warning("No metrics built; check S3 parquet paths and data.")
            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "train_result": train_result,
                "predictions_count": 0,
                "stats": {"total_processed": 0, "saved_s3": 0, "failed": 0, "details": []},
            }

        logger.info("Built metrics for %s wells", len(all_metrics))

        logger.info("\n3. Running recommendations (recommend_from_metrics)...")
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
            except Exception as e:
                logger.warning("Recommendations failed for well %s: %s", well_id, e)

        logger.info("Detected %s recommendations", len(all_recs))

        logger.info("\n3b. Enriching recommendations with LLM (alternative models, time_reduced_hours)...")
        try:
            from operations.suggestions import generate_operation_suggestion_with_openai, _strip_alternative_headers

            def _safe_float(v, default=0.0):
                try:
                    return float(v or default)
                except Exception:
                    return default

            def _t(s, n):
                try:
                    if s is None:
                        return None
                    s2 = str(s)
                    return s2 if len(s2) <= n else s2[:n]
                except Exception:
                    return s

            for rec in all_recs:
                try:
                    well_id = rec.get("well_id", "")
                    action = rec.get("action", "")
                    production_data = (rec.get("metrics") or {}).get("production") or {}
                    sensor_data = (rec.get("metrics") or {}).get("sensors") or {}
                    llm_result = generate_operation_suggestion_with_openai(
                        well_id, action, production_data, sensor_data
                    )
                    alt = llm_result.get("alternative_models", {}) or {}
                    op = llm_result.get("operational_details", {}) or {}
                    fin = llm_result.get("financial_impact", {}) or {}
                    prod = llm_result.get("production_impact", {}) or {}
                    cit = llm_result.get("citations", {}) or {}

                    for key in ("conservative_approach", "aggressive_optimization", "hybrid_model"):
                        if alt.get(key):
                            alt[key] = _strip_alternative_headers(alt[key])

                    rec["conservative_approach"] = _t(alt.get("conservative_approach"), 2000)
                    rec["aggressive_optimization"] = _t(alt.get("aggressive_optimization"), 2000)
                    rec["hybrid_model"] = _t(alt.get("hybrid_model"), 2000)
                    rec["recommended_model"] = _t(alt.get("recommended_model"), 500)

                    # Compute `time_reduced_hours` deterministically from the already
                    # model-driven economics (no constant LLM defaults).
                    # Interpret as time to recover the one-time implementation cost
                    # using daily benefit, capped to the amortization horizon.
                    try:
                        impl = float(rec.get("implementation_cost_usd") or 0.0)
                        daily_benefit = float(rec.get("daily_expense_benefit_usd") or 0.0)
                        amort_days = float(os.getenv("IMPLEMENTATION_AMORTIZATION_DAYS", "180") or 180)
                        if impl > 0.0 and daily_benefit > 0.0:
                            recovery_days = impl / daily_benefit
                            max_hours = max(1.0, amort_days * 24.0)
                            hours = recovery_days * 24.0
                            if hours > max_hours:
                                hours = max_hours
                            rec["time_reduced_hours"] = float(round(hours, 2)) if hours > 0 else None
                        else:
                            rec["time_reduced_hours"] = None
                    except Exception:
                        rec["time_reduced_hours"] = None

                    # Also enrich financial/production/operational fields from LLM if not set
                    if not rec.get("detailed_analysis"):
                        rec["detailed_analysis"] = _t(op.get("detailed_analysis"), 2000)
                    if not rec.get("current_performance"):
                        rec["current_performance"] = _t(op.get("current_performance"), 255)
                    if not rec.get("historical_trend"):
                        rec["historical_trend"] = _t(
                            op.get("historical_trend") or op.get("optimal_performance"), 255
                        )
                    if cit:
                        rec["supporting_analysis"] = _t(cit.get("supporting_analysis"), 2000)
                        rec["data_quality"] = _t(cit.get("data_quality"), 100)

                    logger.info("✓ LLM enriched: %s | %s", well_id, action)
                except Exception as enrich_err:
                    logger.warning("LLM enrichment failed for %s: %s", rec.get("well_id"), enrich_err)

        except Exception as llm_import_err:
            logger.warning("Could not import LLM enrichment module: %s", llm_import_err)

        logger.info("\n4. Writing to S3...")
        s3_key = write_operations_recommendations_to_s3(all_recs)
        stats = {
            "total_processed": len(all_recs),
            "saved_s3": len(all_recs) if s3_key else 0,
            "failed": 0,
            "details": [],
            "s3_key": s3_key or "",
        }
        if all_recs:
            stats["details"] = [{"well_id": r.get("well_id"), "action": r.get("action"), "confidence": r.get("probability") or r.get("confidence")} for r in all_recs]

        logger.info("\n5. Summary")
        logger.info("  Recommendations processed: %s", stats["total_processed"])
        logger.info("  Written to S3: %s", stats["saved_s3"])
        if s3_key:
            logger.info("  S3 key: %s", s3_key)
        logger.info("Pipeline completed successfully.")
        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_result": train_result,
            "predictions_count": len(all_recs),
            "stats": stats,
        }
        
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Operational recommendations pipeline (S3 only)")
    p.add_argument("--skip-training", action="store_true", help="Skip model training")
    p.add_argument("--confidence", type=float, default=0.75, help="Confidence threshold (0-1)")
    args = p.parse_args()
    result = run_pipeline(skip_training=args.skip_training, confidence_threshold=args.confidence)
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
