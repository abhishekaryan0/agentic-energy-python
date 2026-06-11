"""Checkpoint for SYNTH Snowflake → S3 exports (master + financial)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SynthExportCheckpoint:
    organization_id: str
    history_high_watermark: Optional[str] = None  # YYYY-MM-DD inclusive
    invoice_high_watermark: Optional[str] = None  # YYYY-MM-DD inclusive
    last_run_at: Optional[str] = None
    last_run_status: Optional[str] = None
    mode: Optional[str] = None
    last_run_ts: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthExportCheckpoint":
        return cls(
            organization_id=str(data.get("organization_id") or ""),
            history_high_watermark=data.get("history_high_watermark"),
            invoice_high_watermark=data.get("invoice_high_watermark"),
            last_run_at=data.get("last_run_at"),
            last_run_status=data.get("last_run_status"),
            mode=data.get("mode"),
            last_run_ts=data.get("last_run_ts"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def checkpoint_s3_key(org_id: str) -> str:
    return f"{org_id.strip()}/synth_export/_state/export_checkpoint.json"


def load_synth_checkpoint(
    bucket: str,
    org_id: str,
    *,
    region_name: Optional[str] = None,
) -> Optional[SynthExportCheckpoint]:
    from transformations.s3 import read_s3_json

    data = read_s3_json(bucket, checkpoint_s3_key(org_id), region_name=region_name)
    if not data:
        return None
    cp = SynthExportCheckpoint.from_dict(data)
    if cp.organization_id and cp.organization_id != org_id:
        return None
    cp.organization_id = org_id
    return cp


def save_synth_checkpoint(
    bucket: str,
    cp: SynthExportCheckpoint,
    *,
    region_name: Optional[str] = None,
) -> str:
    from transformations.s3 import write_s3_json

    cp.last_run_at = datetime.now(timezone.utc).isoformat()
    key = checkpoint_s3_key(cp.organization_id)
    write_s3_json(bucket, key, cp.to_dict(), region_name=region_name)
    return f"s3://{bucket}/{key}"


def date_window(
    checkpoint: Optional[SynthExportCheckpoint],
    attr: str,
    *,
    end_date: Optional[date] = None,
) -> tuple[Optional[date], date]:
    end = end_date or datetime.now(timezone.utc).date()
    wm = getattr(checkpoint, attr, None) if checkpoint else None
    if not wm:
        return None, end
    last = date.fromisoformat(str(wm)[:10])
    return last, end
