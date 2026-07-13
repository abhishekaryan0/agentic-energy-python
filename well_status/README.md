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

Each tenant gets its own Lambda (one `ORGANIZATION_ID` per function). From repo root (uses `.env` for `S3_BUCKET`, `DATABASE_URL`, and tenant IDs):

| Tenant | Script | Lambda | Schedule (UTC) |
|--------|--------|--------|----------------|
| Red Bluff | `./well_status/deploy/schedule.sh` | `well-status` | 02:00 |
| Apex Permian | `./well_status/deploy/schedule_apex.sh` | `well-status-apex-permian` | 02:15 |
| South Texas Energy Partners | `./well_status/deploy/schedule_step.sh` | `well-status-step` | 02:45 |

**`.env` tenant IDs:**

```bash
RED_BUFF_ORGANIZATION_ID=3e0b5b18-bec1-488a-82c1-a47881c69058
APEX_PERMIAN_ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
STEP_ORGANIZATION_ID=d270f89a-aada-4ff2-adb0-68b3e8087565
ORGANIZATION_IDS=...   # comma-separated list of all three (used by s3_to_postgres / transformations)
```

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
| `deploy/schedule.sh` | Red Bluff daily EventBridge schedule |
| `deploy/schedule_apex.sh` | Apex Permian daily schedule |
| `deploy/schedule_step.sh` | South Texas Energy Partners daily schedule |
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
