"""S3 checkpoint tracking for idempotent S3 -> Postgres sync."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SyncCheckpoint:
    organization_id: str
    table: str
    synced_keys: List[str] = field(default_factory=list)
    last_run_at: Optional[str] = None
    last_run_status: Optional[str] = None
    rows_upserted: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncCheckpoint":
        return cls(
            organization_id=str(data.get("organization_id") or ""),
            table=str(data.get("table") or ""),
            synced_keys=list(data.get("synced_keys") or []),
            last_run_at=data.get("last_run_at"),
            last_run_status=data.get("last_run_status"),
            rows_upserted=int(data.get("rows_upserted") or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != [] or k == "synced_keys"}

    @property
    def synced_key_set(self) -> Set[str]:
        return set(self.synced_keys)


def load_checkpoint(s3_client, bucket: str, key: str, organization_id: str, table: str) -> SyncCheckpoint:
    from botocore.exceptions import ClientError

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        import json

        data = json.loads(obj["Body"].read().decode("utf-8"))
        checkpoint = SyncCheckpoint.from_dict(data)
        if checkpoint.organization_id and checkpoint.organization_id != organization_id:
            logger.warning(
                "Checkpoint org %s does not match %s; starting fresh",
                checkpoint.organization_id,
                organization_id,
            )
            return SyncCheckpoint(organization_id=organization_id, table=table)
        checkpoint.organization_id = organization_id
        checkpoint.table = table
        return checkpoint
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
            return SyncCheckpoint(organization_id=organization_id, table=table)
        raise


def save_checkpoint(
    s3_client,
    bucket: str,
    key: str,
    checkpoint: SyncCheckpoint,
    *,
    status: str,
    rows_upserted: int,
) -> None:
    import json

    checkpoint.last_run_at = datetime.now(timezone.utc).isoformat()
    checkpoint.last_run_status = status
    checkpoint.rows_upserted = rows_upserted
    body = json.dumps(checkpoint.to_dict(), indent=2).encode("utf-8")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
