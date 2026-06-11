"""Runtime configuration for well status updates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from s3_to_postgres.config import _resolve_database_url


@dataclass(frozen=True)
class WellStatusSettings:
    s3_bucket: str
    organization_id: str
    database_url: str
    aws_region: str

    @property
    def postgres_schema(self) -> str:
        return self.organization_id


def load_settings() -> WellStatusSettings:
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    bucket = (os.getenv("S3_BUCKET") or "").strip()
    if not org:
        raise ValueError("ORGANIZATION_ID is required")
    if not bucket:
        raise ValueError("S3_BUCKET is required")
    return WellStatusSettings(
        s3_bucket=bucket,
        organization_id=org,
        database_url=_resolve_database_url(),
        aws_region=(os.getenv("AWS_REGION") or "us-east-1").strip(),
    )
