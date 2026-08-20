# Well status

Computes per-well operational status (**RUNNING**, **IDLE**, **SHUTDOWN**) from the latest sensor activity in S3, and upserts into **PostgreSQL** (`{ORGANIZATION_ID}.well_status`).

## Status logic

| Status | Rule |
|--------|------|
| **SHUTDOWN** | No sensor reading for more than `SHUTDOWN_THRESHOLD_HOURS` (default **24**) |
| **IDLE** | Recent readings, but weighted idle metrics for the lift type ≥ `IDLE_SCORE_THRESHOLD` (**4**) |
| **RUNNING** | Recent readings and not idle |

Lift aliases include `Pumping Unit` → `Rod Pump` (STEP). Idle metrics are lift-specific (e.g. low SPM/fillage for Rod Pump, low motor current for ESP, low injection rate for Gas Lift) — see `IDLE_METRICS` in `service.py`.

## Data flow

| Stage | Location |
|-------|----------|
| Sensor input | `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/well_sensor_readings/date=YYYY-MM-DD/` |
| Incremental checkpoint | `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/_state/well_status_checkpoint.json` |
| Output | PostgreSQL `{ORGANIZATION_ID}.well_status` |

Postgres columns include `well_id`, `organization_id`, `lift_type`, `status`, `last_reading_timestamp`, `last_reading_age_hours`, SPM / motor_current / injection_rate samples, `status_reason`, `updated_at`.

## Run locally

```bash
export S3_BUCKET=...
export ORGANIZATION_ID=...
export DATABASE_URL=...   # or DATABASE_URL_SECRET_ARN

python -m well_status.run
```

**Optional:** `WELL_STATUS_FORCE_FULL=1`, `WELL_STATUS_PARTITION_START_DATE` (default `2026-01-01`), `SENSOR_PARQUET_S3` (legacy snapshot override), `LOG_LEVEL`.

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
```

```bash
./well_status/deploy/deploy.sh   # deploy only, no schedule
```

**Optional:** `SCHEDULE_CRON`, `LAMBDA_FUNCTION_NAME`, `SKIP_ECR_BUILD=1`, `SKIP_LAMBDA_DEPLOY=1`, `LAMBDA_VPC_ENABLED=1` (+ `LAMBDA_SUBNETS`, `LAMBDA_SECURITY_GROUPS` for private RDS).

**Manual invoke:**

```bash
aws lambda invoke --function-name well-status --region us-east-1 \
  --cli-binary-format raw-in-base64-out --payload '{}' /tmp/well_status_out.json
```

## Module layout

| File | Purpose |
|------|---------|
| `handler.py` | Lambda entry point |
| `run.py` | CLI entry point |
| `service.py` | Status computation (S3 read, checkpoint, idle rules) |
| `postgres.py` | PostgreSQL upsert and reads |
| `config.py` | Settings (`S3_BUCKET`, `ORGANIZATION_ID`, `DATABASE_URL`) |
| `deploy/` | ECR image, Lambda, EventBridge schedules |
