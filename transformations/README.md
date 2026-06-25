# Transformation Layer

Exports canonical Snowflake tables into S3 parquet files used by the ML packages.

**Tenant isolation:** With `ORGANIZATION_ID` or comma-separated `ORGANIZATION_IDS`, each export runs `SELECT ... FROM WELL_* WHERE "organization_id" = %s` and writes only to `s3://$S3_BUCKET/{ORGANIZATION_ID}/sensor_data/`.

**Per-org field mapping (export):** `transformations/field_mappings.py` shapes parquet per tenant (mirrors manual ingest mappings in `APEX_PERMIAN/column_maps.py` and `RED_BUFF/column_maps.py`). Apex exports use `freq_hz` and the twin script sensor column set; Red Bluff keeps the full canonical schema with `drive_frequency` + `freq_hz`.

**Parquet shape:** Production columns are shared across tenants. Sensor parquet is profile-specific — Apex matches `generate_digital_twin._build_well_sensor_readings_df` (`freq_hz`, Rod Pump lift labels); Red Bluff keeps the full canonical ML sensor schema.

## Outputs (default: partition layout only)

**Date partitions** (primary storage — default on):

- `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet`
- `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`

**Checkpoint**:

- `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/_state/export_checkpoint.json`

**Optional root snapshots** (off by default; enable with `TRANSFORM_WRITE_SNAPSHOTS=1`):

- `well_sensor_readings.parquet`, `daily_production.parquet` at `sensor_data/` root (for legacy ML readers)

## Required Environment

```bash
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_DATABASE=AT_RISK_ASSETS
SNOWFLAKE_SCHEMA=SENSOR_DATA
SNOWFLAKE_ROLE=...

S3_BUCKET=...
ORGANIZATION_ID=...          # single tenant
ORGANIZATION_IDS=...         # comma-separated (Lambda loops each org)
AWS_REGION=us-east-1
```

`ORGANIZATION_ID` or `ORGANIZATION_IDS` is **required** unless you set `TRANSFORM_ALLOW_CROSS_TENANT_EXPORT=1` (exports all rows, no `WHERE organization_id`; use only for admin backfills).

Multi-tenant Lambda (same S3 layout per org):

```bash
ORGANIZATION_IDS=3e0b5b18-bec1-488a-82c1-a47881c69058,147a92b9-f90a-473d-b443-e6e668c4fbb7
```

Each org writes under `s3://$S3_BUCKET/{ORGANIZATION_ID}/sensor_data/` — identical folder structure, isolated by UUID prefix.

Optional:

- `TRANSFORM_FAIL_ON_EMPTY_SNOWFLAKE=1` — fail the job if preflight counts are zero for that org (avoid writing empty snapshots by mistake).
- `TRANSFORM_SKIP_WHEN_NO_NEW_DATA=1` (default) — incremental runs exit quickly when checkpoint exists and Snowflake has no rows after the watermark (`status: skipped`).
- `TRANSFORM_WRITE_SNAPSHOTS=0` (default) — do not write root `*.parquet` files; partitions only.
- `SNOWFLAKE_SOCKET_TIMEOUT=600` — read timeout for large sensor exports.

## Run

One-time full backfill (all history for the org):

```bash
python -m transformations.run --mode full
```

Daily incremental (new rows since checkpoint → new `date=` partitions only):

```bash
python -m transformations.run --mode incremental
```

Dry-run (SQL + windows only):

```bash
python -m transformations.run --dry-run --mode incremental
```

Useful overrides:

```bash
TRANSFORM_MODE=full python -m transformations.run
python -m transformations.run --tables well_sensor_readings
python -m transformations.run --output-prefix "$ORGANIZATION_ID/sensor_data"
python -m transformations.run --batch-size 50000
python -m transformations.run --no-partitions
python -m transformations.run --write-snapshots   # opt-in legacy root parquet files
```

Exports **only read** Snowflake; they do **not** insert or update Snowflake.

## SYNTH master + financial export (separate from sensor_data)

Exports Red Bluff synthetic tables from Snowflake to the same S3 layout as `APEX_PERMIAN/generate_master.py`:

| Snowflake table | S3 output |
|-----------------|-----------|
| `SYNTH_ASSETS` + `SYNTH_PERSONNEL` | `{ORG}/digital_twin/master_data/run_ts=.../master_assets.parquet` |
| `SYNTH_ASSET_DAILY_HISTORY` | `{ORG}/digital_twin/master_data/run_ts=.../master_history.parquet` |
| `SYNTH_INVOICES` | `{ORG}/financial_data/run_ts=.../invoices_v2.parquet` |

Checkpoint: `{ORG}/synth_export/_state/export_checkpoint.json`

```bash
# Full export (all history + invoices for org)
python -m transformations.run_synth_export --mode full

# Incremental (new dates since checkpoint)
python -m transformations.run_synth_export --mode incremental
```

Env: same as sensor export (`S3_BUCKET`, `ORGANIZATION_ID`, `SNOWFLAKE_*`). Optional: `SYNTH_EXPORT_MODE`, `SYNTH_EXPORT_FAIL_ON_EMPTY=1`.

The export filters by `ORGANIZATION_ID` in SQL and validates row-level `organization_id` on each batch. Incremental mode also requires `ORGANIZATION_ID`.

## Scheduled export (EventBridge + Lambda, daily)

```bash
# 1) One-time full backfill first
python -m transformations.run --mode full

# 2) Deploy daily incremental schedule
./transformations/deploy/schedule.sh
```

- EventBridge rule: `transformations-daily` (`cron(15 0 * * ? *)` — daily at 00:15 UTC)
- Lambda: `transformations`
- Logs: `/aws/lambda/transformations`
- Writes new data only under `date=YYYY-MM-DD/` folders (no root snapshot files); updates checkpoint

Deploy layout: `transformations/deploy/` (`deploy.sh`, `schedule.sh`, `Dockerfile.lambda`, `requirements.lambda.txt`).
