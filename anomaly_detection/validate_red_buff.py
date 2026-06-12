"""
Red Bluff validation: measure whether anomaly scores rise before SHUT-IN events.

Usage (from repo root):
  export S3_BUCKET=... ORGANIZATION_ID=...
  python -m anomaly_detection.validate_red_buff

Optional:
  ANOMALY_VALIDATE_MAX_WELLS=50
  ANOMALY_VALIDATE_PRE_WINDOW_DAYS=14,120  (min,max days before SHUT-IN)
  ANOMALY_VALIDATE_SCORE_THRESHOLD=0.55
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from anomaly_detection.bulk_detect import _extract_reading_dict, _history_last_n, _normalize_sensor_dataframe
from anomaly_detection.bulk_detect import _load_sensor_dataframe, _get_bucket_and_org
from anomaly_detection.detector import get_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_pre_window() -> Tuple[int, int]:
    raw = (os.getenv("ANOMALY_VALIDATE_PRE_WINDOW_DAYS") or "14,120").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        return 14, 120
    try:
        lo, hi = int(parts[0]), int(parts[1])
        return min(lo, hi), max(lo, hi)
    except ValueError:
        return 14, 120


def _find_shutin_transitions(wdf: pd.DataFrame) -> List[pd.Timestamp]:
    """Return timestamps where status transitions into SHUT-IN."""
    wdf = wdf.sort_values("timestamp").copy()
    if "well_status" not in wdf.columns:
        return []
    status = wdf["well_status"].astype(str).str.upper().str.strip()
    prev = status.shift(1).fillna("PRODUCING")
    mask = (status == "SHUT-IN") & (prev != "SHUT-IN")
    return list(wdf.loc[mask, "timestamp"])


def _score_row(
    detector,
    well_id: str,
    lift_type: str,
    row: Dict[str, Any],
    hist_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    readings = _extract_reading_dict(row)
    historical = [_extract_reading_dict(r) for r in hist_rows]
    return detector.check_anomaly(
        readings,
        historical_data=historical,
        well_id=str(well_id),
        lift_type=lift_type,
    )


def validate_well(
    detector,
    well_id: str,
    wdf: pd.DataFrame,
    *,
    pre_min_days: int,
    pre_max_days: int,
    score_threshold: float,
    last_n: int,
) -> Dict[str, Any]:
    wdf = wdf.sort_values("timestamp").copy()
    lift_type = str(wdf["lift_type"].dropna().iloc[-1]) if not wdf["lift_type"].dropna().empty else ""
    if not lift_type:
        return {"well_id": well_id, "skipped": True, "reason": "missing_lift_type"}

    shutins = _find_shutin_transitions(wdf)
    if not shutins:
        return {"well_id": well_id, "shutin_events": 0, "events_with_elevated_scores": 0}

    events_evaluated = 0
    events_elevated = 0
    event_details: List[Dict[str, Any]] = []

    for shut_ts in shutins:
        window_start = shut_ts - pd.Timedelta(days=pre_max_days)
        window_end = shut_ts - pd.Timedelta(days=pre_min_days)
        if window_end <= window_start:
            continue

        window = wdf[(wdf["timestamp"] >= window_start) & (wdf["timestamp"] <= window_end)]
        if window.empty:
            continue

        max_score = 0.0
        max_detail: Optional[Dict[str, Any]] = None
        for _idx, row in window.iterrows():
            row_dict = row.to_dict()
            current_ts = row_dict["timestamp"]
            hist = wdf[wdf["timestamp"] < current_ts].tail(max(0, last_n - 1))
            detail = _score_row(
                detector,
                well_id,
                lift_type,
                row_dict,
                hist.to_dict(orient="records"),
            )
            score = float(detail.get("anomaly_score", 0) or 0)
            if score > max_score:
                max_score = score
                max_detail = detail

        events_evaluated += 1
        elevated = max_score >= score_threshold
        if elevated:
            events_elevated += 1
        event_details.append(
            {
                "shutin_ts": pd.to_datetime(shut_ts, utc=True).isoformat(),
                "window_start": pd.to_datetime(window_start, utc=True).isoformat(),
                "window_end": pd.to_datetime(window_end, utc=True).isoformat(),
                "max_score": round(max_score, 3),
                "elevated": elevated,
                "max_iso": (max_detail or {}).get("iso_score"),
                "max_pyod": (max_detail or {}).get("pyod_score"),
                "max_mad": (max_detail or {}).get("mad_score"),
            }
        )

    return {
        "well_id": well_id,
        "lift_type": lift_type,
        "shutin_events": len(shutins),
        "events_evaluated": events_evaluated,
        "events_with_elevated_scores": events_elevated,
        "recall": (events_elevated / events_evaluated) if events_evaluated else None,
        "events": event_details,
    }


def run_validation(*, max_wells: Optional[int] = None) -> Dict[str, Any]:
    bucket, org = _get_bucket_and_org()
    pre_min, pre_max = _parse_pre_window()
    score_threshold = float(os.getenv("ANOMALY_VALIDATE_SCORE_THRESHOLD", "0.55"))
    last_n = _history_last_n()

    if max_wells is None:
        raw = (os.getenv("ANOMALY_VALIDATE_MAX_WELLS") or "").strip()
        max_wells = int(raw) if raw.isdigit() else None

    logger.info(
        "Loading sensor data org=%s pre_window=%s-%s days threshold=%.2f last_n=%s",
        org,
        pre_min,
        pre_max,
        score_threshold,
        last_n,
    )
    df = _normalize_sensor_dataframe(_load_sensor_dataframe(bucket, org))
    if df.empty:
        return {"success": False, "message": "No sensor data"}

    detector = get_detector()
    wells = sorted(df["well_id"].astype(str).unique())
    if max_wells is not None:
        wells = wells[: max(0, int(max_wells))]

    results: List[Dict[str, Any]] = []
    total_shutins = 0
    total_evaluated = 0
    total_elevated = 0

    for well_id in wells:
        wdf = df[df["well_id"].astype(str) == well_id]
        summary = validate_well(
            detector,
            well_id,
            wdf,
            pre_min_days=pre_min,
            pre_max_days=pre_max,
            score_threshold=score_threshold,
            last_n=last_n,
        )
        results.append(summary)
        total_shutins += int(summary.get("shutin_events") or 0)
        total_evaluated += int(summary.get("events_evaluated") or 0)
        total_elevated += int(summary.get("events_with_elevated_scores") or 0)

    overall_recall = (total_elevated / total_evaluated) if total_evaluated else None
    report = {
        "success": True,
        "organization_id": org,
        "wells_checked": len(wells),
        "shutin_events": total_shutins,
        "events_evaluated": total_evaluated,
        "events_with_elevated_scores": total_elevated,
        "overall_recall": overall_recall,
        "pre_window_days": [pre_min, pre_max],
        "score_threshold": score_threshold,
        "wells": results,
    }

    logger.info(
        "Validation complete: wells=%s shut-ins=%s evaluated=%s elevated=%s recall=%s",
        len(wells),
        total_shutins,
        total_evaluated,
        total_elevated,
        f"{overall_recall:.1%}" if overall_recall is not None else "n/a",
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate anomaly scores before Red Bluff SHUT-IN events")
    parser.add_argument("--max-wells", type=int, default=None, help="Limit wells (default: env or all)")
    args = parser.parse_args(argv)
    try:
        report = run_validation(max_wells=args.max_wells)
    except Exception as e:
        logger.error("Validation failed: %s", e)
        return 1
    if not report.get("success"):
        logger.error("%s", report.get("message", "failed"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
