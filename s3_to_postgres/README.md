# S3 → PostgreSQL sync

Incremental (or backfill) sync of detection and production parquet from S3 into PostgreSQL. Schema name = tenant `ORGANIZATION_ID`.

## Tables synced

| Postgres table | S3 prefix |
|----------------|-----------|
| `anomaly_suggestions` | `{ORG}/detections/anomaly_detection/` |
| `anomaly_reviews` | `{ORG}/detections/anomaly_review/` |
| `operation_recommendations` | `{ORG}/detections/operations/` |
| `daily_production` | `{ORG}/sensor_data/daily_production/date=YYYY-MM-DD/` (upsert; **all** S3 history by default) |
| `monthly_production` | Aggregated from `daily_production` (all months present in daily) |

Checkpoints: `{prefix}/_state/postgres_sync_{table}.json`.

Ops sync may call schema ensure helpers (e.g. analysis text columns, oil-price related fields on `operation_recommendations`).

`sync_anomaly_violations.py` exists in the package but is **not** registered in `SYNC_HANDLERS` / CLI — violations stay on S3 only until wired.

## Required env

```bash
S3_BUCKET=...
ORGANIZATION_ID=...              # single tenant
# or
ORGANIZATION_IDS=org-a,org-b,org-c
DATABASE_URL=postgresql://...   # or DATABASE_URL_SECRET_ARN
AWS_REGION=us-east-1
```

Each tenant UUID is both the S3 prefix and the PostgreSQL schema. With `ORGANIZATION_IDS`, scheduled and `table=all` runs sync every table for every listed tenant. Pass `organization_id` in the Lambda/CLI event to target one tenant.

Skip tables via `SYNC_SKIP_TABLES` (comma-separated). Default local `.env` may skip production tables — remove them from that list or sync a single table explicitly.

## Local run

```bash
python -m s3_to_postgres.run --sync-mode incremental --table all
python -m s3_to_postgres.run --sync-mode incremental --table daily_production \
  --organization-id 147a92b9-f90a-473d-b443-e6e668c4fbb7
python -m s3_to_postgres.run --sync-mode backfill --table daily_production
# optional: limit to last N years instead of all S3 history
python -m s3_to_postgres.run --sync-mode backfill --table daily_production --backfill-years 2
```

CLI `--table` choices: `anomaly_suggestions`, `anomaly_reviews`, `operation_recommendations`, `daily_production`, `monthly_production`, `all`.

## Deploy (Lambda + EventBridge)

```bash
./s3_to_postgres/deploy/schedule.sh    # Lambda + daily 01:00 UTC
./s3_to_postgres/deploy/deploy.sh      # Lambda only
./s3_to_postgres/deploy/run_sync_step.sh   # STEP-only invoke helper
```

One multi-tenant Lambda. Include all org UUIDs in `.env` `ORGANIZATION_IDS`, then redeploy.

| Tenant | Org env |
|--------|---------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` |

- Lambda: `s3-to-postgres-sync`
- EventBridge: `s3-to-postgres-sync-daily`
- Logs: `/aws/lambda/s3-to-postgres-sync`

## Layout

```
s3_to_postgres/
  handler.py                         # Lambda; SYNC_HANDLERS router
  run.py                             # CLI
  config.py / db.py / s3_io.py / checkpoint.py / sync_common.py
  sync_anomaly_suggestions.py
  sync_anomaly_reviews.py
  sync_operation_recommendations.py
  sync_daily_production.py
  sync_monthly_production.py
  sync_anomaly_violations.py         # present; not wired to handler/CLI
  deploy/
```
