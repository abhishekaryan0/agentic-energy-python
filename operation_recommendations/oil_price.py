"""
Resolve oil spot price ($/BBL) per organization.

Reads org_settings.oil_price_type from PostgreSQL:
  - Manual  -> org_settings.manual_value
  - Brent   -> latest EIA Brent spot (eia_prices.get_brent_spot_usd_per_bbl)
  - WTI     -> latest EIA WTI spot (eia_prices.get_wti_spot_usd_per_bbl)

Falls back to EIA when org_settings is missing or Postgres is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from operation_recommendations.eia_prices import (
    get_brent_spot_usd_per_bbl,
    get_wti_spot_usd_per_bbl,
)
from operation_recommendations.org_settings import (
    fetch_org_oil_price_settings,
    load_operations_settings,
)

logger = logging.getLogger(__name__)


def resolve_oil_price_usd_per_bbl(
    organization_id: Optional[str] = None,
) -> Tuple[float, str, Optional[str], Optional[str]]:
    """
    Return (price_usd_per_bbl, source, period, oil_price_type).

    source is org_settings_manual, eia_brent, eia_brent_cached, eia_wti, eia_wti_cached, or env_fallback.
    period is the EIA observation date when applicable.
    """
    org = (organization_id or os.getenv("ORGANIZATION_ID") or "").strip()
    configured_type: Optional[str] = None

    if org:
        try:
            settings = load_operations_settings(org)
            org_row = fetch_org_oil_price_settings(settings)
            if org_row:
                configured_type = (org_row.get("oil_price_type") or "").strip()
                normalized = configured_type.lower()

                if normalized == "manual":
                    manual = org_row.get("manual_value")
                    if manual is not None:
                        try:
                            price = float(manual)
                            if price > 0:
                                logger.info(
                                    "Oil price from org_settings (Manual): $%.2f/bbl org=%s",
                                    price,
                                    org,
                                )
                                return price, "org_settings_manual", None, configured_type
                        except (TypeError, ValueError):
                            logger.warning(
                                "Invalid manual_value in org_settings for org %s: %r",
                                org,
                                manual,
                            )
                    logger.warning(
                        "org_settings oil_price_type=Manual but manual_value missing/invalid; "
                        "falling back to EIA Brent org=%s",
                        org,
                    )
                elif normalized == "brent" or not normalized:
                    price, src, period = get_brent_spot_usd_per_bbl()
                    return price, src, period, configured_type or "Brent"
                elif normalized == "wti":
                    price, src, period = get_wti_spot_usd_per_bbl()
                    return price, src, period, configured_type or "WTI"
                else:
                    logger.warning(
                        "Unknown org_settings.oil_price_type=%r for org %s; using EIA Brent",
                        configured_type,
                        org,
                    )
        except ValueError:
            pass
        except Exception as exc:
            logger.warning(
                "org_settings lookup failed for org %s; falling back to EIA Brent: %s",
                org,
                exc,
            )

    price, src, period = get_brent_spot_usd_per_bbl()
    return price, src, period, configured_type or "Brent"
