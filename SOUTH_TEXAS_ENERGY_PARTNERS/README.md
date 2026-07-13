# South Texas Energy Partners synthetic data

Generates **100-well Eagle Ford digital twin** synthetic data **directly into Snowflake** (same pattern as `RED_BUFF`).

**Snowflake target:** `AT_RISK_ASSETS.SOUTH_TEXAS_SENSOR_DATA` (separate schema from Red Bluff / Apex).

## Run locally (production — direct to Snowflake)

```bash
# Verify connectivity + row counts
python SOUTH_TEXAS_ENERGY_PARTNERS/check_snowflake.py

# Incremental (append missing days through today)
python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py

# Full reset + regenerate
python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py --reset
```

## Environment

```bash
STEP_ORGANIZATION_ID=d270f89a-aada-4ff2-adb0-68b3e8087565
STEP_SNOWFLAKE_SCHEMA=SOUTH_TEXAS_SENSOR_DATA   # optional; this is the default
SNOWFLAKE_USER / SNOWFLAKE_ACCOUNT / SNOWFLAKE_WAREHOUSE / SNOWFLAKE_ROLE
SNOWFLAKE_TOKEN or SNOWFLAKE_PASSWORD
```

## Snowflake tables (same names as RED_BUFF, different schema)

| Table | Contents |
|-------|----------|
| `WELL_DAILY_PRODUCTION` | Daily oil/gas/water volumes |
| `WELL_SENSOR_READINGS` | Daily sensor readings per well |
| `SYNTH_ASSETS` | Asset catalog (120 assets) |
| `SYNTH_PERSONNEL` | Personnel per asset |
| `SYNTH_ASSET_DAILY_HISTORY` | Full daily JSON history (wells + supporting) |
| `SYNTH_PIPELINE_STATE` | Incremental pipeline registry |
| `SYNTH_INVOICES` | Financial ledger |

## Optional — rebuild master JSON locally

```bash
cd SOUTH_TEXAS_ENERGY_PARTNERS
python3 generate_digital_twin.py
python3 generate_invoices.py
python3 build_master.py
python3 export_to_csv.py          # optional CSV export
```

Dev shortcut to bulk-load the master JSON into Snowflake (not used in production Lambda):

```bash
python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py --from-master
```

## Deploy (Lambda + EventBridge)

```bash
export STEP_ORGANIZATION_ID=d270f89a-aada-4ff2-adb0-68b3e8087565
./SOUTH_TEXAS_ENERGY_PARTNERS/deploy/schedule.sh
```

| Item | Value |
|------|--------|
| Lambda | `south-texas-energy-partners` |
| Schedule | `south-texas-step-daily` → `cron(30 0 * * ? *)` (00:30 UTC) |
| Logs | `/aws/lambda/south-texas-energy-partners` |

## Pipeline order (production)

1. **south-texas-energy-partners** (00:30 UTC) — generate new days in Snowflake  
2. **transformations** (separate) — export Snowflake → S3 when wired for this tenant

## Layout

```
SOUTH_TEXAS_ENERGY_PARTNERS/
  config.py                  # STEP_ORGANIZATION_ID, schema defaults
  generate_digital_twin.py   # Twin engine (100 wells)
  generate_invoices.py
  build_master.py
  pipeline.py                # Direct Snowflake generation (like RED_BUFF/pipeline.py)
  snowflake_io.py
  column_maps.py
  run_incremental_snowflake.py
  handler.py
  deploy/
```
