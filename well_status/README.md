# Well status

Computes per-well operational status (**RUNNING**, **IDLE**, **SHUTDOWN**) from sensor readings in S3, and stores results in **PostgreSQL** (`{ORGANIZATION_ID}.well_status`).

## Data flow

| Stage | Location |
|-------|----------|
| Sensor input | `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/well_sensor_readings/date=YYYY-MM-DD/` |
| Incremental checkpoint | `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/_state/well_status_checkpoint.json` |
| Output | PostgreSQL `{ORGANIZATION_ID}.well_status` |

## Run locally

```bash
export S3_BUCKET=...
export ORGANIZATION_ID=...
export DATABASE_URL=...

python -m well_status.run
```

## Deploy (Lambda + EventBridge)

From repo root (uses `.env` for `S3_BUCKET`, `ORGANIZATION_ID`, `DATABASE_URL`):

```bash
./well_status/deploy/schedule.sh
```

This builds the container image, creates/updates Lambda **`well-status`**, and wires EventBridge rule **`well-status-daily`** (`cron(0 2 * * ? *)` = daily 02:00 UTC).

**Deploy only** (no schedule):

```bash
./well_status/deploy/deploy.sh
```

**Optional env vars:** `SCHEDULE_CRON`, `LAMBDA_FUNCTION_NAME`, `SKIP_ECR_BUILD=1`, `SKIP_LAMBDA_DEPLOY=1`, `LAMBDA_VPC_ENABLED=1` (for private RDS; also set `LAMBDA_SUBNETS` and `LAMBDA_SECURITY_GROUPS`).

**Manual invoke:**

```bash
aws lambda invoke --function-name well-status --region us-east-1 \
  --cli-binary-format raw-in-base64-out --payload '{}' /tmp/well_status_out.json
```

**Logs:** CloudWatch `/aws/lambda/well-status`

## Deploy files

| File | Purpose |
|------|---------|
| `deploy/deploy.sh` | Build ECR image + create/update Lambda |
| `deploy/schedule.sh` | Daily EventBridge schedule |
| `deploy/Dockerfile.lambda` | Lambda container image |
| `deploy/requirements.lambda.txt` | Pinned Python deps |

## Module layout

| File | Purpose |
|------|---------|
| `handler.py` | Lambda entry point |
| `run.py` | CLI entry point |
| `service.py` | Status computation (S3 read, checkpoint) |
| `postgres.py` | PostgreSQL upsert and reads |
| `config.py` | Settings (`S3_BUCKET`, `ORGANIZATION_ID`, `DATABASE_URL`) |
