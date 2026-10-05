#!/usr/bin/env python3
"""
Integrated Operational Recommendations Pipeline (S3 only: model load, input, output).
====================================================================================

1. Optionally trains models from S3 parquet (train_operations_from_parquet).
2. Builds metrics from S3 parquet (s3://BUCKET/{ORGANIZATION_ID}/sensor_data/).
3. Runs recommend_from_metrics (models loaded from S3 model.tar.gz).
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

from operation_recommendations.s3_models import resolve_operations_model_tarball_s3_uri
from operation_recommendations.utils import get_model_path
from operation_recommendations.train_models import build_metrics_from_parquet, train_operations_from_parquet
from operation_recommendations.recommendations.persistence import (
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


def _get_s3_training_parquet_paths() -> tuple:
    """Load partitioned sensor + production from S3 into temp parquet files for training."""
    from operation_recommendations.s3_inputs import materialize_training_parquets_from_s3

    bucket = os.getenv("S3_BUCKET", "").strip()
    org = os.getenv("ORGANIZATION_ID", "").strip()
    if not bucket or not org:
        raise ValueError("S3_BUCKET and ORGANIZATION_ID are required for run_pipeline (S3-only).")
    return materialize_training_parquets_from_s3(bucket=bucket, org=org)


def _get_latest_invoices_parquet_paths() -> List[str]:
    """Latest invoices parquet URIs under financial_data/run_ts=."""
    from operation_recommendations.s3_inputs import get_latest_invoices_parquet_uris

    return get_latest_invoices_parquet_uris()


def run_pipeline(skip_training: bool = False, confidence_threshold: float = 0.75) -> Dict[str, Any]:
    """
    Execute the operational recommendations pipeline (S3 only).

    - Model: joblibs from S3 ``operations-train-.../output/model.tar.gz``.
    - Input: S3 parquet (sensor + production).
    - Output: S3 parquet (write_operations_recommendations_to_s3).
    """
    try:
        logger.info("=" * 80)
        logger.info("OPERATIONAL RECOMMENDATIONS PIPELINE (S3 only)")
        logger.info("=" * 80)
        logger.info("Start Time: %s", datetime.now(timezone.utc).isoformat())
        logger.info("Confidence Threshold: %.0f%%", confidence_threshold * 100)

        sensor_path, prod_path = _get_s3_training_parquet_paths()
        invoice_uris = _get_latest_invoices_parquet_paths()
        logger.info("S3 training data materialized: sensor=%s production=%s", sensor_path, prod_path)
        if invoice_uris:
            logger.info("S3 input: invoices=%s", invoice_uris)
        else:
            logger.warning("No invoices parquet found; implementation_cost_usd may be 0.")

        model_uri = resolve_operations_model_tarball_s3_uri()
        logger.info("Operations model tarball: %s", model_uri or "(training will produce new artifacts)")

        train_result = None
        try:
            if not skip_training:
                logger.info("\n1. Training models from S3 parquet partitions...")
                cache_dir = os.environ.get("OPERATIONS_MODELS_DIR") or os.path.join(
                    _PROJECT_ROOT, "models", "operations_run"
                )
                os.makedirs(cache_dir, exist_ok=True)
                train_result = train_operations_from_parquet(
                    sensor_path,
                    prod_path,
                    cache_dir,
                    model_path=os.path.join(cache_dir, "decision_tree_alldata.joblib"),
                )
                logger.info("Models trained (samples etc. in train_result)")
            else:
                logger.info("\n1. Skipping training (--skip-training); using model from S3.")

            logger.info("\n2. Building metrics from S3 parquet...")
            all_metrics = build_metrics_from_parquet(
                sensor_path,
                prod_path,
                invoices_parquet_paths=invoice_uris or None,
            )
        finally:
            for p in (sensor_path, prod_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
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

        logger.info("\n3b. Enriching recommendations with LLM (alternative models, citations)...")
        try:
            from operation_recommendations.recommendations.persistence import (
                apply_alternatives_and_citations,
                apply_analysis_description,
                apply_metric_descriptions,
                finalize_analysis_fields,
            )

            for rec in all_recs:
                try:
                    well_id = rec.get("well_id", "")
                    action = rec.get("action", "")

                    # Deterministic time_reduced_hours from economics.
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
                            rec["time_reduced_hours"] = float(round(hours, 2)) if hours > 0 else 0.0
                        else:
                            rec["time_reduced_hours"] = 0.0
                    except Exception:
                        rec["time_reduced_hours"] = 0.0

                    apply_metric_descriptions(rec)
                    analysis = str(rec.get("analysis_description") or "")
                    if (not analysis) or analysis.startswith("OpenRouter analysis generation failed"):
                        apply_analysis_description(rec)
                    finalize_analysis_fields(rec)
                    apply_alternatives_and_citations(rec)
                    action = rec.get("action") or action

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
