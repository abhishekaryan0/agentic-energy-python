"""S3 checkpoint for incremental Snowflake-to-parquet exports."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExportCheckpoint:
    organization_id: str
    production_high_watermark: Optional[str] = None  # YYYY-MM-DD inclusive
    sensor_high_watermark: Optional[str] = None  # ISO timestamp inclusive
    last_run_at: Optional[str] = None
    last_run_status: Optional[str] = None
    mode: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportCheckpoint":
        return cls(
            organization_id=str(data.get("organization_id") or ""),
            production_high_watermark=data.get("production_high_watermark"),
            sensor_high_watermark=data.get("sensor_high_watermark"),
            last_run_at=data.get("last_run_at"),
            last_run_status=data.get("last_run_status"),
            mode=data.get("mode"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def checkpoint_s3_key(prefix: str) -> str:
    return f"{prefix.strip().strip('/')}/_state/export_checkpoint.json"


def load_checkpoint(
    bucket: str,
    prefix: str,
    organization_id: str,
    *,
    region_name: Optional[str] = None,
) -> Optional[ExportCheckpoint]:
    from transformations.s3 import get_s3_client, read_s3_json

    key = checkpoint_s3_key(prefix)
    data = read_s3_json(bucket, key, region_name=region_name)
    if not data:
        return None
    cp = ExportCheckpoint.from_dict(data)
    if cp.organization_id and cp.organization_id != organization_id:
        logger.warning(
            "Checkpoint org %s differs from run org %s; ignoring checkpoint",
            cp.organization_id,
            organization_id,
        )
        return None
    cp.organization_id = organization_id
    return cp


def save_checkpoint(
    bucket: str,
    prefix: str,
    checkpoint: ExportCheckpoint,
    *,
    region_name: Optional[str] = None,
) -> str:
    from transformations.s3 import write_s3_json

    key = checkpoint_s3_key(prefix)
    checkpoint.last_run_at = datetime.now(timezone.utc).isoformat()
    write_s3_json(bucket, key, checkpoint.to_dict(), region_name=region_name)
    return f"s3://{bucket}/{key}"


def production_window(
    checkpoint: Optional[ExportCheckpoint],
    *,
    end_date: Optional[date] = None,
) -> tuple[Optional[date], Optional[date]]:
    """
    Return (exclusive_start, inclusive_end) for production export.

    Full export: (None, end_date).
    Incremental: (last_date, end_date) where query uses DATE > start.
    """
    end = end_date or datetime.now(timezone.utc).date()
    if checkpoint is None or not checkpoint.production_high_watermark:
        return None, end
    last = date.fromisoformat(checkpoint.production_high_watermark[:10])
    return last, end


def sensor_window(
    checkpoint: Optional[ExportCheckpoint],
    *,
    end_ts: Optional[datetime] = None,
) -> tuple[Optional[datetime], datetime]:
    """Return (exclusive_start, inclusive_end) for sensor export."""
    end = end_ts or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if checkpoint is None or not checkpoint.sensor_high_watermark:
        return None, end
    start = datetime.fromisoformat(
        checkpoint.sensor_high_watermark.replace("Z", "+00:00")
    )
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start, end


def advance_production_watermark(
    current: Optional[str],
    candidate: Optional[str],
) -> Optional[str]:
    if not candidate:
        return current
    if not current:
        return candidate[:10]
    return max(current[:10], candidate[:10])


def advance_sensor_watermark(
    current: Optional[str],
    candidate: Optional[str],
) -> Optional[str]:
    if not candidate:
        return current
    if not current:
        return candidate
    cur = datetime.fromisoformat(current.replace("Z", "+00:00"))
    cand = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    return cand.isoformat() if cand > cur else current
