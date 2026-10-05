"""SYNTH_PERSONNEL_V2 — assigned personnel per asset (V2 assigned_personnel)."""

from __future__ import annotations

from typing import List

import pandas as pd

TABLE_NAME = "SYNTH_PERSONNEL_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("asset_id", "VARCHAR(100)"),
    ("role_key", "VARCHAR(50)"),
    ("name", "VARCHAR(200)"),
    ("role", "VARCHAR(100)"),
    ("email", "VARCHAR(200)"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


def ddl(qualified: str) -> str:
    cols = ",\n            ".join(f"{name.upper()} {ctype}" for name, ctype in COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            {cols},
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID, ASSET_ID, ROLE_KEY)
        )
        """


def personnel_to_dataframe(
    asset_id: str,
    personnel: dict,
    organization_id: str,
) -> pd.DataFrame:
    rows = []
    for role_key, person in (personnel or {}).items():
        rows.append(
            {
                "organization_id": organization_id,
                "asset_id": asset_id,
                "role_key": role_key,
                "name": person.get("name"),
                "role": person.get("role"),
                "email": person.get("email"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.DataFrame(rows)[COLUMN_NAMES]


def assets_personnel_to_dataframe(assets: dict, organization_id: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for asset_id, asset in assets.items():
        frames.append(
            personnel_to_dataframe(
                asset_id, asset.get("assigned_personnel") or {}, organization_id
            )
        )
    if not frames:
        return pd.DataFrame(columns=COLUMN_NAMES)
    return pd.concat(frames, ignore_index=True)[COLUMN_NAMES]


class SynthPersonnelV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
    personnel_to_dataframe = staticmethod(personnel_to_dataframe)
    assets_personnel_to_dataframe = staticmethod(assets_personnel_to_dataframe)
