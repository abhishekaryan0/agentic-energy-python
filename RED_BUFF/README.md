# Red Bluff synthetic data

Generates a **60-well Permian digital twin** directly into Snowflake (`WELL_DAILY_PRODUCTION`, `WELL_SENSOR_READINGS`, `SYNTH_*` assets/personnel/history/invoices, pipeline state).

Default schema: `AT_RISK_ASSETS.SENSOR_DATA`. Org: `RED_BUFF_ORGANIZATION_ID` / `ORGANIZATION_ID` (default `3e0b5b18-bec1-488a-82c1-a47881c69058`).

For dataset design (leases, lift mix, failure modes), see `README.txt` (offline JSON rebuild path). Production path is Snowflake via `pipeline.py` / Lambda.

## Run locally

```bash
python RED_BUFF/check_snowflake.py
python RED_BUFF/run_incremental_snowflake.py
python RED_BUFF/run_incremental_snowflake.py --reset
```

## Local web UI (optional dev)

```bash
uvicorn RED_BUFF.web.app:app --reload --host 0.0.0.0 --port 8765
```

Open http://localhost:8765 — Generate / Reset against Snowflake (not used by production Lambda). See [`web/README.md`](web/README.md).

## Offline rebuild (not the Lambda path)

```bash
cd RED_BUFF
python3 generate_digital_twin.py
python3 generate_invoices.py
python3 build_master.py
```

## Deploy (Lambda + EventBridge)

From repo root (`.env` needs `RED_BUFF_ORGANIZATION_ID` or `ORGANIZATION_ID`, `SNOWFLAKE_*`):

```bash
./RED_BUFF/deploy/schedule.sh
```

| Item | Value |
|------|--------|
| Lambda | `red-buff` |
| Schedule | `red-buff-daily` → `cron(0 0 * * ? *)` (00:00 UTC) |
| Memory / timeout | 8192 MB / 900 s |
| Handler | `RED_BUFF.handler.handler` |
| Logs | `/aws/lambda/red-buff` |

```bash
./RED_BUFF/deploy/deploy.sh   # deploy only
```

**Manual invoke:**

```bash
aws lambda invoke --function-name red-buff --region us-east-1 \
  --cli-binary-format raw-in-base64-out --payload '{}' /tmp/red_buff_out.json

# Full reset
aws lambda invoke --function-name red-buff --region us-east-1 \
  --cli-binary-format raw-in-base64-out --payload '{"reset": true}' /tmp/red_buff_out.json
```

## Pipeline order (production)

1. **red-buff** (00:00 UTC) — generate new days in Snowflake  
2. **transformations** (00:15 UTC) — export Snowflake → S3 parquet  

## Key modules

| File | Purpose |
|------|---------|
| `handler.py` | Lambda entry |
| `pipeline.py` | `run_red_buff_pipeline`, reset |
| `run_incremental_snowflake.py` | CLI |
| `snowflake_io.py` / `column_maps.py` | Schema + mapping via `transformations.columns` |
| `generate_digital_twin.py` | Twin engine |
| `deploy/` | ECR + Lambda + EventBridge |
