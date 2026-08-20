# Transformation Layer

Exports canonical Snowflake tables into S3 parquet used by the ML packages. **Read-only** on Snowflake (no inserts/updates).

**Tenant isolation:** With `ORGANIZATION_ID` or comma-separated `ORGANIZATION_IDS`, each export runs `SELECT … FROM WELL_* WHERE "organization_id" = %s` and writes only to `s3://$S3_BUCKET/{ORGANIZATION_ID}/sensor_data/`.

**Per-org Snowflake schema:** `org_snowflake.py` routes STEP (`STEP_ORGANIZATION_ID`) to `STEP_SNOWFLAKE_SCHEMA` (default `SOUTH_TEXAS_SENSOR_DATA`); Red Buff / Apex use `SNOWFLAKE_SCHEMA` (default `SENSOR_DATA`).

**Per-org field mapping:** `field_mappings.py` shapes parquet per tenant. Apex uses `freq_hz` and a subset column set; Red Bluff / STEP keep the full canonical schema (`drive_frequency` + `freq_hz`). Override profile with `TRANSFORM_ORG_PROFILE_{UUID}=apex_permian` when needed.

## Outputs (default: partition layout only)

**Date partitions** (primary — default on):

- `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet`
- `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`

**Checkpoint:** `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/_state/export_checkpoint.json`

**Optional root snapshots** (`TRANSFORM_WRITE_SNAPSHOTS=1`): root `well_sensor_readings.parquet`, `daily_production.parquet` under `sensor_data/`.

## Required environment

```bash
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...   # or SNOWFLAKE_TOKEN
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_DATABASE=AT_RISK_ASSETS
SNOWFLAKE_SCHEMA=SENSOR_DATA
SNOWFLAKE_ROLE=...

S3_BUCKET=...
ORGANIZATION_ID=...          # single tenant
ORGANIZATION_IDS=...         # comma-separated (Lambda loops each org)
AWS_REGION=us-east-1

# STEP (when that org is in ORGANIZATION_IDS)
STEP_ORGANIZATION_ID=d270f89a-aada-4ff2-adb0-68b3e8087565
STEP_SNOWFLAKE_SCHEMA=SOUTH_TEXAS_SENSOR_DATA
```

`ORGANIZATION_ID` or `ORGANIZATION_IDS` is **required** unless `TRANSFORM_ALLOW_CROSS_TENANT_EXPORT=1` (admin backfills only).

Multi-tenant example (all three orgs):

```bash
ORGANIZATION_IDS=3e0b5b18-bec1-488a-82c1-a47881c69058,147a92b9-f90a-473d-b443-e6e668c4fbb7,d270f89a-aada-4ff2-adb0-68b3e8087565
```

Optional:

- `TRANSFORM_FAIL_ON_EMPTY_SNOWFLAKE=1` — fail if preflight counts are zero
- `TRANSFORM_SKIP_WHEN_NO_NEW_DATA=1` (default) — incremental skip when no rows after watermark
- `TRANSFORM_WRITE_SNAPSHOTS=0` (default) — partitions only
- `SNOWFLAKE_SOCKET_TIMEOUT=600` — large sensor exports

## Run (sensor / production)

```bash
python -m transformations.run --mode full
python -m transformations.run --mode incremental
python -m transformations.run --dry-run --mode incremental
```

Overrides:

```bash
TRANSFORM_MODE=full python -m transformations.run
python -m transformations.run --tables well_sensor_readings
python -m transformations.run --write-snapshots
```

Snowflake sources: `WELL_SENSOR_READINGS`, `WELL_DAILY_PRODUCTION`.

## SYNTH master + financial export

Separate path from `sensor_data`. Exports SYNTH tables (Red Buff / STEP style) to S3:

| Snowflake table | S3 output |
|-----------------|-----------|
| `SYNTH_ASSETS` + `SYNTH_PERSONNEL` | `{ORG}/digital_twin/master_data/run_ts=…/master_assets.parquet` |
| `SYNTH_ASSET_DAILY_HISTORY` | `{ORG}/digital_twin/master_data/run_ts=…/master_history.parquet` |
| `SYNTH_INVOICES` | `{ORG}/financial_data/run_ts=…/invoices_v2.parquet` |

Checkpoint: `{ORG}/synth_export/_state/export_checkpoint.json`

```bash
python -m transformations.run_synth_export --mode full
python -m transformations.run_synth_export --mode incremental
```

Env: same Snowflake/S3 vars. Optional: `SYNTH_EXPORT_MODE`, `SYNTH_EXPORT_FAIL_ON_EMPTY=1`. Requires exactly one `ORGANIZATION_ID` for synth.

## Lambda

```bash
./transformations/deploy/schedule.sh
```

- EventBridge: `transformations-daily` → `cron(45 0 * * ? *)` (00:45 UTC)
- Lambda: `transformations`
- Logs: `/aws/lambda/transformations`

**Event examples:**

```json
{"mode": "incremental"}
{"mode": "full", "organization_id": "147a92b9-f90a-473d-b443-e6e668c4fbb7"}
{"mode": "full", "organization_id": "d270f89a-aada-4ff2-adb0-68b3e8087565", "synth": true}
```

`synth: true` or `export_type: "synth"` routes to SYNTH export (`run_synth_export_for_event`).

Deploy layout: `transformations/deploy/` (`deploy.sh`, `schedule.sh`, `Dockerfile.lambda`, `requirements.lambda.txt`).
