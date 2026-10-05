"""SYNTH_PIPELINE_STATE_V2 — incremental / load registry for the V2 pipeline."""

from __future__ import annotations

TABLE_NAME = "SYNTH_PIPELINE_STATE_V2"

COLUMNS = [
    ("organization_id", "VARCHAR(36)"),
    ("state_json", "VARCHAR"),
]

COLUMN_NAMES = [c for c, _ in COLUMNS]


def ddl(qualified: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            ORGANIZATION_ID VARCHAR(36) NOT NULL,
            STATE_JSON VARCHAR,
            UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE (ORGANIZATION_ID)
        )
        """


class SynthPipelineStateV2:
    name = TABLE_NAME
    columns = COLUMNS
    column_names = COLUMN_NAMES
    ddl = staticmethod(ddl)
