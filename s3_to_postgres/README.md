# S3 → PostgreSQL sync

Incremental sync of detection parquet from S3 into PostgreSQL (schema = tenant `ORGANIZATION_ID`).

## Tables synced

- `anomaly_suggestions` — from `detections/anomaly_detection/`
- `anomaly_reviews` — from `detections/anomaly_review/`
- `operation_recommendations` — from `detections/operations/`
- `daily_production` — from `sensor_data/daily_production/date=YYYY-MM-DD/` (upsert; keeps last 2 years only)
- `monthly_production` — monthly totals from `daily_production` (same 2-year retention)

## Required env

```bash
S3_BUCKET=...
ORGANIZATION_ID=...              # single tenant
# or
ORGANIZATION_IDS=org-a,org-b      # multiple tenants (comma-separated)
DATABASE_URL=postgresql://...   # or DATABASE_URL_SECRET_ARN
AWS_REGION=us-east-1
```

Each tenant UUID is used for both the S3 prefix (`s3://$S3_BUCKET/$ORG_ID/...`) and the PostgreSQL schema (`$ORG_ID.table_name`). Checkpoints are stored per tenant under `{ORG_ID}/.../_state/`.

When `ORGANIZATION_IDS` is set, scheduled and `table=all` runs sync every table for every listed tenant. Pass `organization_id` in the Lambda/CLI event to target one tenant only.

Skip tables without removing their sync code via `SYNC_SKIP_TABLES` (comma-separated). Default local `.env` skips production tables; remove them from that list or run a single table explicitly to sync production.

## Local run

```bash
python -m s3_to_postgres.run --sync-mode incremental --table all
python -m s3_to_postgres.run --sync-mode incremental --table daily_production --organization-id 147a92b9-f90a-473d-b443-e6e668c4fbb7
python -m s3_to_postgres.run --sync-mode backfill --table daily_production --backfill-years 2
```

## Deploy (Lambda + EventBridge)

```bash
# Deploy Lambda + daily schedule (01:00 UTC) — ORGANIZATION_IDS from .env
./s3_to_postgres/deploy/schedule.sh

# Lambda only
./s3_to_postgres/deploy/deploy.sh

# STEP-only incremental sync (after deploy includes STEP in ORGANIZATION_IDS)
./s3_to_postgres/deploy/run_sync_step.sh
```

Tenants share one Lambda. Include all org UUIDs in `.env` `ORGANIZATION_IDS` (Red Bluff, Apex, STEP), then redeploy so IAM and Lambda env pick up the new S3 prefix.

| Tenant | Org env |
|--------|---------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` |

- Lambda: `s3-to-postgres-sync`
- EventBridge rule: `s3-to-postgres-sync-daily`
- Logs: `/aws/lambda/s3-to-postgres-sync`

## Layout

```
s3_to_postgres/
  handler.py          # Lambda entrypoint
  run.py              # CLI entrypoint
  deploy/
    deploy.sh
    schedule.sh
    Dockerfile.lambda
    requirements.lambda.txt
```
