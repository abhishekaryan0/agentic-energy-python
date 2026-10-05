# South Texas Energy Partners synthetic data

Generates a **100-well Eagle Ford digital twin** (5 leases, 20 supporting assets → **120 assets**) **directly into Snowflake**, same pattern as `RED_BUFF`.

**Snowflake target:** `AT_RISK_ASSETS.SOUTH_TEXAS_SENSOR_DATA` (separate schema from Red Bluff / Apex).  
Org: `STEP_ORGANIZATION_ID` = `d270f89a-aada-4ff2-adb0-68b3e8087565`.

## Run locally (production — direct to Snowflake)

```bash
python SOUTH_TEXAS_ENERGY_PARTNERS/check_snowflake.py
python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py
python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py --reset
```

## Environment

```bash
STEP_ORGANIZATION_ID=d270f89a-aada-4ff2-adb0-68b3e8087565
STEP_SNOWFLAKE_SCHEMA=SOUTH_TEXAS_SENSOR_DATA   # default
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
| `SYNTH_ASSET_DAILY_HISTORY` | Full daily JSON history |
| `SYNTH_PIPELINE_STATE` | Incremental pipeline registry |
| `SYNTH_INVOICES` | Financial ledger |

## Optional — rebuild master JSON locally

```bash
cd SOUTH_TEXAS_ENERGY_PARTNERS
python3 generate_digital_twin.py
python3 generate_invoices.py
python3 build_master.py
python3 export_to_csv.py
```

Bulk-load master JSON into Snowflake (dev; not production Lambda):

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
| ECR | `south-texas-step-lambda` |
| Schedule | `south-texas-step-daily` → `cron(30 0 * * ? *)` (00:30 UTC) |
| Handler | `SOUTH_TEXAS_ENERGY_PARTNERS.handler.handler` |
| Logs | `/aws/lambda/south-texas-energy-partners` |

Deploy injects `STEP_ORGANIZATION_ID`, `ORGANIZATION_ID`, and STEP schema env vars.

## Pipeline order (production)

1. **south-texas-energy-partners** (00:30 UTC) — new days in Snowflake  
2. **transformations** — include STEP in `ORGANIZATION_IDS`; `org_snowflake.py` selects `SOUTH_TEXAS_SENSOR_DATA`  
3. ML / sync packages use `*_step.sh` schedules  

Field mappings currently treat STEP like Red Bluff (full canonical columns; lift `Pumping Unit` aliased downstream).

## Layout

```
SOUTH_TEXAS_ENERGY_PARTNERS/
  config.py                  # STEP_ORGANIZATION_ID, schema, NUM_WELLS
  generate_digital_twin.py
  generate_invoices.py
  build_master.py
  pipeline.py
  snowflake_io.py
  column_maps.py
  run_incremental_snowflake.py
  handler.py
  deploy/
  v2/                        # STEP V2 (*_V2 tables, native V2 fields)
    tables/                  # one file per Snowflake table
    pipeline.py
    snowflake_io.py
    run_incremental_snowflake.py
```

## STEP V2 (same layout & pipeline as v1 → Snowflake `*_V2`)

Same structure as v1 (`generate_digital_twin` → Snowflake). Twin engine lives in
`v2/generate_digital_twin.py`. **Every row includes `organization_id`**.  
Tables are suffixed `_V2`.

```bash
# Same commands as v1
python3 -m SOUTH_TEXAS_ENERGY_PARTNERS.v2.run_incremental_snowflake
python3 -m SOUTH_TEXAS_ENERGY_PARTNERS.v2.run_incremental_snowflake --reset
```

### Deploy (Lambda + ECR) — generate all V2 data

```bash
# Build/push image + create/update Lambda
./SOUTH_TEXAS_ENERGY_PARTNERS/v2/deploy/deploy.sh

# Full backfill of all 100 wells into *_V2 (batched Lambda invokes)
./SOUTH_TEXAS_ENERGY_PARTNERS/v2/deploy/run_full_backfill.sh

# Daily incremental schedule (00:45 UTC)
./SOUTH_TEXAS_ENERGY_PARTNERS/v2/deploy/schedule.sh
```

| Item | Value |
|------|--------|
| Lambda | `south-texas-energy-partners-v2` |
| ECR | `south-texas-step-v2-lambda` |
| Handler | `SOUTH_TEXAS_ENERGY_PARTNERS.v2.handler.handler` |
| Schedule | `south-texas-step-v2-daily` → `cron(45 0 * * ? *)` |

Full load uses batched invokes (`well_offset` / `well_limit`) because one Lambda cannot finish all 100 wells in 15 minutes.

```
v2/
  generate_digital_twin.py   # twin engine (like v1)
  invoice_gen.py
  master_assets.py
  pipeline.py
  snowflake_io.py
  column/table modules in tables/
  run_incremental_snowflake.py
  handler.py
```

| V2 table | Contents |
|----------|----------|
| `WELL_DAILY_PRODUCTION_V2` | V2 volumes + `organization_id` |
| `WELL_SENSOR_READINGS_V2` | Native sensors + `organization_id` |
| `SYNTH_ASSETS_V2` | WI, NRI, lease_type + `organization_id` |
| `SYNTH_PERSONNEL_V2` | Personnel + `organization_id` |
| `SYNTH_ASSET_DAILY_HISTORY_V2` | Full history JSON + `organization_id` |
| `SYNTH_INVOICES_V2` | Invoices + `organization_id` |
| `SYNTH_REVENUE_V2` | Revenue ledger + `organization_id` |
| `SYNTH_PIPELINE_STATE_V2` | Incremental registry |

Env: `STEP_ORGANIZATION_ID`, `STEP_V2_SNOWFLAKE_SCHEMA` (default `SOUTH_TEXAS_SENSOR_DATA`).
