# S3 → PostgreSQL sync

Incremental sync of detection parquet from S3 into PostgreSQL (schema = `ORGANIZATION_ID`).

## Tables synced

- `anomaly_suggestions` — from `detections/anomaly_detection/`
- `anomaly_violations` — from `detections/anomaly_detection/`
- `anomaly_reviews` — from `detections/anomaly_review/`
- `operation_recommendations` — from `detections/operations/`

## Required env

```bash
S3_BUCKET=...
ORGANIZATION_ID=...
DATABASE_URL=postgresql://...   # or DATABASE_URL_SECRET_ARN
AWS_REGION=us-east-1
```

## Local run

```bash
python -m s3_to_postgres.run --sync-mode incremental --table all
```

## Deploy (Lambda + EventBridge)

```bash
# Deploy Lambda + daily schedule (01:00 UTC)
./s3_to_postgres/deploy/schedule.sh

# Lambda only
./s3_to_postgres/deploy/deploy.sh
```

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
