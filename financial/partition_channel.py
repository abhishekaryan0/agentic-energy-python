"""Merge partitioned SageMaker input channels into a single parquet snapshot."""

from __future__ import annotations

import os
from typing import List, Optional


def list_partition_files(channel_dir: str) -> List[str]:
    """Return sorted parquet paths for date=YYYY-MM-DD/part-0000.parquet under channel_dir."""
    out: List[str] = []
    if not os.path.isdir(channel_dir):
        return out
    for name in os.listdir(channel_dir):
        if not name.startswith("date="):
            continue
        part = os.path.join(channel_dir, name, "part-0000.parquet")
        if os.path.isfile(part):
            out.append(part)
    out.sort()
    return out


def resolve_partition_channel_dir(channel_dir: str, nested_subdir: str) -> str:
    """
    Point at partition root when SageMaker downloaded a parent prefix.

    Example: channel has daily_production/date=... when S3Uri was .../sensor_data/.
    """
    nested = os.path.join(channel_dir, nested_subdir)
    if os.path.isdir(nested) and (
        list_partition_files(nested)
        or os.path.isfile(os.path.join(nested, f"{nested_subdir}.parquet"))
    ):
        return nested
    return channel_dir


def materialize_partition_snapshot(
    channel_dir: str,
    *,
    legacy_filename: str,
    snapshot_name: str,
    max_days: Optional[int],
) -> str:
    """
    Return path to legacy parquet or a merged snapshot from date partitions.
    """
    legacy_path = os.path.join(channel_dir, legacy_filename)
    if os.path.isfile(legacy_path):
        return legacy_path

    parts = list_partition_files(channel_dir)
    if not parts:
        raise FileNotFoundError(
            f"No {legacy_filename} or date=*/part-0000.parquet under {channel_dir!r}"
        )

    import pandas as pd

    if max_days is not None and max_days > 0:
        parts = parts[-max_days:]
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = os.path.join(channel_dir, snapshot_name)
    df.to_parquet(out_path, index=False)
    return out_path
