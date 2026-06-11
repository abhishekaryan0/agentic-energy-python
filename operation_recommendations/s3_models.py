"""
Load operation-recommendation joblibs directly from SageMaker model.tar.gz on S3.

No persistent local model cache: tarballs are read on first use and joblibs are kept in memory.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_JOBLIB_CACHE: Dict[Tuple[str, str], Any] = {}


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def find_latest_model_tarball_s3_uri(
    *,
    bucket: str,
    prefix: str,
    key_substring: str,
) -> Optional[str]:
    """Return s3:// URI for the newest ``.../output/model.tar.gz`` matching key_substring."""
    import boto3

    if not prefix.endswith("/"):
        prefix = prefix + "/"
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj.get("Key") or ""
            if key_substring in k and k.endswith("output/model.tar.gz"):
                keys.append(k)
    if not keys:
        return None
    keys.sort(reverse=True)
    return f"s3://{bucket}/{keys[0]}"


def resolve_operations_model_tarball_s3_uri() -> Optional[str]:
    override = (
        (os.getenv("OPERATIONS_MODEL_TARBALL_S3_URI") or "").strip()
        or (os.getenv("OPERATIONS_MODEL_S3_URI") or "").strip()
    )
    if override:
        return override if override.startswith("s3://") else f"s3://{override}"

    bucket = (os.getenv("S3_BUCKET") or "").strip()
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        return None
    return find_latest_model_tarball_s3_uri(
        bucket=bucket,
        prefix=f"{org}/model/",
        key_substring="operations-train-",
    )


def resolve_financial_model_tarball_s3_uri() -> Optional[str]:
    override = (os.getenv("FINANCIAL_MODEL_TARBALL_S3_URI") or "").strip()
    if override:
        return override if override.startswith("s3://") else f"s3://{override}"

    bucket = (os.getenv("S3_BUCKET") or "").strip()
    org = (os.getenv("ORGANIZATION_ID") or "").strip()
    if not bucket or not org:
        return None
    return find_latest_model_tarball_s3_uri(
        bucket=bucket,
        prefix=f"{org}/model/",
        key_substring="financial-train-",
    )


def _pick_tar_member(tar: tarfile.TarFile, member_name: str) -> str:
    matches = [m.name for m in tar.getmembers() if m.name.endswith(member_name)]
    if not matches:
        raise FileNotFoundError(f"{member_name!r} not found in tarball")
    exact = [n for n in matches if n.split("/")[-1] == member_name]
    return exact[0] if exact else matches[0]


def load_joblib_from_s3_tarball(s3_uri: str, member_name: str) -> Any:
    """Load one joblib member from a model.tar.gz on S3 (cached in memory)."""
    cache_key = (s3_uri, member_name)
    if cache_key in _JOBLIB_CACHE:
        return _JOBLIB_CACHE[cache_key]

    import boto3
    import joblib

    bucket, key = parse_s3_uri(s3_uri)
    buf = io.BytesIO()
    logger.info("Loading %s from %s", member_name, s3_uri)
    boto3.client("s3").download_fileobj(bucket, key, buf)
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        member_path = _pick_tar_member(tar, member_name)
        extracted = tar.extractfile(member_path)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {member_path!r} from {s3_uri}")
        obj = joblib.load(extracted)

    _JOBLIB_CACHE[cache_key] = obj
    return obj


def load_operations_joblib(member_name: str) -> Optional[Any]:
    uri = resolve_operations_model_tarball_s3_uri()
    if not uri:
        return None
    try:
        return load_joblib_from_s3_tarball(uri, member_name)
    except Exception as exc:
        logger.warning("Failed to load %s from %s: %s", member_name, uri, exc)
        return None


def load_financial_joblib(member_name: str = "financial_implementation_cost.joblib") -> Optional[Any]:
    uri = resolve_financial_model_tarball_s3_uri()
    if not uri:
        return None
    try:
        return load_joblib_from_s3_tarball(uri, member_name)
    except Exception as exc:
        logger.warning("Failed to load %s from %s: %s", member_name, uri, exc)
        return None


def load_operations_model_dict(member_name: str) -> Optional[Dict[str, Any]]:
    """Return {model, features} dict from an operations joblib on S3."""
    from operation_recommendations.config import CANONICAL_FEATURES

    obj = load_operations_joblib(member_name)
    if obj is None:
        return None
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "features": CANONICAL_FEATURES}
