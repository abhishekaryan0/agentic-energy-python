"""
Latest Brent spot ($/BBL) from the U.S. EIA Open Data API v2.

Uses EIA_API_KEY from the environment. Caches successful responses (TTL configurable)
with a thread-safe lock. Retries transient failures. Validates plausible prices.

Docs: https://www.eia.gov/opendata/
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_CACHE: Dict[str, Any] = {"ts": 0.0, "price": None, "period": None, "raw": None}

_EIA_BRENT_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
_DEFAULT_TTL = 3600.0
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_RETRIES = 3
# Reject absurd API/malformed values; fall back to OPERATIONS_OIL_PRICE_USD
_MIN_PRICE = 5.0
_MAX_PRICE = 500.0


def _cache_ttl_sec() -> float:
    try:
        return float(os.getenv("EIA_CACHE_TTL_SEC", str(_DEFAULT_TTL)) or _DEFAULT_TTL)
    except ValueError:
        return _DEFAULT_TTL


def _http_timeout() -> float:
    try:
        return float(os.getenv("EIA_HTTP_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT)) or _DEFAULT_TIMEOUT)
    except ValueError:
        return _DEFAULT_TIMEOUT


def _retry_count() -> int:
    try:
        return max(1, int(os.getenv("EIA_HTTP_RETRIES", str(_DEFAULT_RETRIES)) or _DEFAULT_RETRIES))
    except ValueError:
        return _DEFAULT_RETRIES


def _fallback_price() -> float:
    return float(os.getenv("OPERATIONS_OIL_PRICE_USD", "70") or 70)


def _validate_price(price: float) -> bool:
    return _MIN_PRICE <= price <= _MAX_PRICE


def _parse_eia_rows(data: dict) -> Tuple[float, Optional[str]]:
    rows = (data.get("response") or {}).get("data") or []
    if not rows:
        raise ValueError("EIA response has no data rows")
    row0 = rows[0]
    val = row0.get("value")
    period = row0.get("period")
    price = float(val)
    if not _validate_price(price):
        raise ValueError(f"EIA price out of bounds: {price}")
    return price, period


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _fetch_brent_spot_data(api_key: str, timeout: float) -> Tuple[float, Optional[str]]:
    """
    Brent spot via petroleum/pri/spt. Prefer series facet RBRTE (Europe Brent);
    fall back to product EPCBRENT.
    """
    base_params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": "1",
    }
    last_err: Optional[BaseException] = None
    for facet_key, facet_val in (
        ("facets[series][]", "RBRTE"),
        ("facets[product][]", "EPCBRENT"),
    ):
        try:
            params = {**base_params, facet_key: facet_val}
            qs = urllib.parse.urlencode(params)
            url = f"{_EIA_BRENT_URL}?{qs}"
            data = _http_get_json(url, timeout)
            return _parse_eia_rows(data)
        except Exception as e:
            last_err = e
    raise ValueError(f"EIA spot facets failed: {last_err}")


def _fetch_brent_seriesid(api_key: str, timeout: float) -> Tuple[float, Optional[str]]:
    """Fallback: v2 seriesid endpoint for daily Europe Brent."""
    series_id = (os.getenv("EIA_BRENT_SERIES_ID") or "PET.RBRTE.D").strip()
    params = {
        "api_key": api_key,
        "length": "1",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
    }
    qs = urllib.parse.urlencode(params)
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?{qs}"
    data = _http_get_json(url, timeout)
    return _parse_eia_rows(data)


def _fetch_brent_once(api_key: str, timeout: float) -> Tuple[float, Optional[str]]:
    try:
        return _fetch_brent_spot_data(api_key, timeout)
    except Exception as spot_err:
        logger.debug("EIA spot facet fetch failed (%s); trying seriesid", spot_err)
        return _fetch_brent_seriesid(api_key, timeout)


def get_brent_spot_usd_per_bbl() -> Tuple[float, str, Optional[str]]:
    """
    Return (price_usd_per_bbl, source, period).

    source is 'eia_brent' on success, 'eia_brent_cached' when served from cache,
    or 'env_fallback' if the request fails or EIA_API_KEY is missing.
    """
    now = time.time()
    ttl = _cache_ttl_sec()
    with _lock:
        if (
            _CACHE["price"] is not None
            and now - float(_CACHE["ts"]) < ttl
        ):
            return float(_CACHE["price"]), "eia_brent_cached", _CACHE.get("period")

    api_key = (os.getenv("EIA_API_KEY") or "").strip()
    if not api_key:
        p = _fallback_price()
        logger.warning("EIA_API_KEY not set; using OPERATIONS_OIL_PRICE_USD=%s", p)
        return p, "env_fallback", None

    timeout = _http_timeout()
    last_err: Optional[BaseException] = None
    for attempt in range(_retry_count()):
        try:
            price, period = _fetch_brent_once(api_key, timeout)
            with _lock:
                _CACHE["ts"] = time.time()
                _CACHE["price"] = price
                _CACHE["period"] = period
                _CACHE["raw"] = None
            logger.info("EIA Brent spot: $%.2f/bbl period=%s", price, period)
            return price, "eia_brent", period
        except (urllib.error.URLError, TimeoutError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < _retry_count() - 1:
                delay = 0.5 * (2**attempt)
                logger.warning("EIA Brent fetch attempt %s failed: %s; retry in %.1fs", attempt + 1, e, delay)
                time.sleep(delay)

    p = _fallback_price()
    logger.warning("EIA Brent fetch failed after retries (%s); using fallback %s $/BBL", last_err, p)
    return p, "env_fallback", None
