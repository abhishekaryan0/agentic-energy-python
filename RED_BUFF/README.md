# Red Bluff synthetic data

Generates **60-well digital twin** synthetic data into **Snowflake** (`WELL_DAILY_PRODUCTION`, `WELL_SENSOR_READINGS`, invoices, master history).

## Run locally

```bash
# Incremental (append missing days through today)
python RED_BUFF/run_incremental_snowflake.py

# Full reset + regenerate
python RED_BUFF/run_incremental_snowflake.py --reset
```

## Local web UI (optional dev)

```bash
uvicorn RED_BUFF.web.app:app --reload --port 8765
```

Open http://localhost:8765 — manual Generate/Reset buttons (not used in production Lambda).

## Deploy (Lambda + EventBridge)

From repo root (`.env` needs `ORGANIZATION_ID`, `SNOWFLAKE_*`):

```bash
./RED_BUFF/deploy/schedule.sh
```

| Item | Value |
|------|--------|
| Lambda | `red-buff` |
| Schedule | `red-buff-daily` → `cron(0 0 * * ? *)` (00:00 UTC daily) |
| Memory / timeout | 8192 MB / 900 s |
| Logs | `/aws/lambda/red-buff` |

**Deploy only** (no schedule):

```bash
./RED_BUFF/deploy/deploy.sh
```

**Manual invoke:**

```bash
aws lambda invoke --function-name red-buff --region us-east-1 \
  --cli-binary-format raw-in-base64-out --payload '{}' /tmp/red_buff_out.json
```

**Full reset via Lambda** (wipes org Snowflake data first):

```bash
aws lambda invoke --function-name red-buff --region us-east-1 \
  --cli-binary-format raw-in-base64-out --payload '{"reset": true}' /tmp/red_buff_out.json
```

## Pipeline order (production)

1. **red-buff** (00:00 UTC) — generate new days in Snowflake  
2. **transformations** (recommend 01:00 UTC) — export Snowflake → S3 parquet  

## Deploy files

| File | Purpose |
|------|---------|
| `handler.py` | Lambda entry point |
| `pipeline.py` | Core generation logic |
| `deploy/deploy.sh` | Build ECR image + Lambda |
| `deploy/schedule.sh` | EventBridge daily schedule |
| `deploy/Dockerfile.lambda` | Container image |
| `deploy/requirements.lambda.txt` | Pinned deps |
