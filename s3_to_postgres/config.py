"""Runtime configuration from environment variables."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

_ORG_ID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


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
        "daily_production": "sensor_data/daily_production",
        "monthly_production": "sensor_data/daily_production",
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


def _parse_organization_id_list(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [_strip(str(item)) for item in raw if _strip(str(item))]
    return [_strip(part) for part in str(raw).split(",") if _strip(part)]


def load_organization_ids(overrides: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return tenant UUIDs from ORGANIZATION_IDS (comma-separated) or ORGANIZATION_ID."""
    overrides = overrides or {}
    raw_list = overrides.get("organization_ids")
    if raw_list is None:
        raw_list = os.getenv("ORGANIZATION_IDS")
    if raw_list:
        ids = _parse_organization_id_list(raw_list)
        if not ids:
            raise ValueError("ORGANIZATION_IDS is empty")
        return ids

    single = _strip(overrides.get("organization_id") or os.getenv("ORGANIZATION_ID"))
    if single:
        return [single]
    raise ValueError("ORGANIZATION_ID or ORGANIZATION_IDS is required")


def organization_id_from_s3_key(key: str) -> Optional[str]:
    """First path segment when it looks like a tenant UUID."""
    org = _strip(key.split("/", 1)[0])
    return org if _ORG_ID_RE.match(org) else None


def resolve_organization_ids(event: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Resolve which organizations to sync for a handler/CLI/Lambda event.

    Priority: explicit organization_id -> organization_ids list -> infer from s3_key -> env.
    """
    event = event or {}
    overrides = dict(event.get("settings_overrides") or {})

    single = _strip(overrides.get("organization_id") or event.get("organization_id"))
    if single:
        return [single]

    raw_list = overrides.get("organization_ids") or event.get("organization_ids")
    if raw_list:
        ids = _parse_organization_id_list(raw_list)
        if not ids:
            raise ValueError("organization_ids is empty")
        return ids

    s3_key = _strip(event.get("s3_key"))
    if not s3_key:
        from s3_to_postgres.s3_io import event_detail_s3_key

        s3_key = _strip(event_detail_s3_key(event) or "")
    if s3_key:
        org = organization_id_from_s3_key(s3_key)
        if org:
            return [org]

    return load_organization_ids(overrides)


def load_skip_sync_tables(event: Optional[Dict[str, Any]] = None) -> set[str]:
    """Tables to exclude from table=all runs (code kept; manual single-table sync still works)."""
    event = event or {}
    overrides = dict(event.get("settings_overrides") or {})
    raw = (
        overrides.get("skip_tables")
        or event.get("skip_tables")
        or os.getenv("SYNC_SKIP_TABLES")
        or ""
    )
    if isinstance(raw, (list, tuple, set)):
        return {item.strip() for item in raw if str(item).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def load_settings(overrides: Optional[Dict[str, Any]] = None) -> Settings:
    overrides = overrides or {}
    org = _strip(overrides.get("organization_id") or os.getenv("ORGANIZATION_ID"))
    bucket = _strip(overrides.get("s3_bucket") or os.getenv("S3_BUCKET"))
    if not org:
        orgs = load_organization_ids(overrides)
        org = orgs[0]
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
