"""Organization settings from PostgreSQL (per-org schema)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from s3_to_postgres.config import _resolve_database_url
from s3_to_postgres.db import postgres_connection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperationsSettings:
    organization_id: str
    database_url: str

    @property
    def postgres_schema(self) -> str:
        return self.organization_id


def load_operations_settings(organization_id: Optional[str] = None) -> OperationsSettings:
    org = (organization_id or os.getenv("ORGANIZATION_ID") or "").strip()
    if not org:
        raise ValueError("ORGANIZATION_ID is required")
    return OperationsSettings(
        organization_id=org,
        database_url=_resolve_database_url(),
    )


def fetch_org_oil_price_settings(
    settings: Optional[OperationsSettings] = None,
) -> Optional[Dict[str, Any]]:
    """
    Read oil_price_type and manual_value from org_settings in the org schema.

    Each organization uses its UUID as the PostgreSQL search_path schema (same as well_status).
    """
    settings = settings or load_operations_settings()
    query = """
        SELECT oil_price_type, manual_value
        FROM org_settings
        LIMIT 1
    """
    try:
        with postgres_connection(settings.database_url, settings.postgres_schema) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()
                if not row:
                    logger.info(
                        "No org_settings row in schema %s; defaulting to Brent/EIA",
                        settings.postgres_schema,
                    )
                    return None
                return {
                    "oil_price_type": row[0],
                    "manual_value": row[1],
                }
    except Exception as exc:
        logger.warning(
            "Failed to read org_settings for schema %s: %s",
            settings.postgres_schema,
            exc,
        )
        return None
