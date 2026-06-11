"""Runtime configuration from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse, urlunparse


def _strip(value: Optional[str]) -> str:
    return (value or "").strip()


@dataclass(frozen=True)
class Settings:
    s3_bucket: str
    organization_id: str
    database_url: str
    aws_region: str
    sync_table: str
    detection_prefix: str
    checkpoint_key: str

    @property
    def postgres_schema(self) -> str:
        return self.organization_id


def _resolve_database_url() -> str:
    direct = _strip(os.getenv("DATABASE_URL"))
    if direct:
        return direct

    secret_arn = _strip(os.getenv("DATABASE_URL_SECRET_ARN"))
    if not secret_arn:
        raise ValueError("Set DATABASE_URL or DATABASE_URL_SECRET_ARN")

    import boto3

    client = boto3.client("secretsmanager", region_name=_strip(os.getenv("AWS_REGION")) or None)
    response = client.get_secret_value(SecretId=secret_arn)
    secret = response.get("SecretString") or ""
    if not secret:
        raise ValueError(f"Secret {secret_arn} is empty")

    try:
        payload = json.loads(secret)
    except json.JSONDecodeError:
        return secret

    for key in ("DATABASE_URL", "database_url", "url", "connection_string"):
        if payload.get(key):
            return str(payload[key])
    raise ValueError(f"Secret {secret_arn} does not contain DATABASE_URL")


def _s3_subpath_for_table(sync_table: str) -> str:
    mapping = {
        "anomaly_suggestions": "detections/anomaly_detection",
        "anomaly_violations": "detections/anomaly_detection",
        "anomaly_reviews": "detections/anomaly_review",
        "operation_recommendations": "detections/operations",
    }
    subpath = mapping.get(sync_table)
    if not subpath:
        raise ValueError(f"Unsupported sync_table: {sync_table}")
    return subpath


def parse_database_url(database_url: str) -> tuple[str, str]:
    """Return (connection_url_without_query, sslmode)."""
    parsed = urlparse(database_url)
    query = parse_qs(parsed.query)
    sslmode = _strip(query.get("sslmode", ["require"])[0]) or "require"
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return clean_url, sslmode


def load_settings(overrides: Optional[Dict[str, Any]] = None) -> Settings:
    overrides = overrides or {}
    org = _strip(overrides.get("organization_id") or os.getenv("ORGANIZATION_ID"))
    bucket = _strip(overrides.get("s3_bucket") or os.getenv("S3_BUCKET"))
    if not org:
        raise ValueError("ORGANIZATION_ID is required")
    if not bucket:
        raise ValueError("S3_BUCKET is required")

    sync_table = _strip(overrides.get("sync_table") or os.getenv("SYNC_TABLE")) or "anomaly_suggestions"
    s3_subpath = _s3_subpath_for_table(sync_table)
    detection_prefix = f"{org}/{s3_subpath}"
    checkpoint_key = f"{detection_prefix}/_state/postgres_sync_{sync_table}.json"

    return Settings(
        s3_bucket=bucket,
        organization_id=org,
        database_url=_resolve_database_url(),
        aws_region=_strip(overrides.get("aws_region") or os.getenv("AWS_REGION")) or "us-east-1",
        sync_table=sync_table,
        detection_prefix=detection_prefix,
        checkpoint_key=checkpoint_key,
    )
