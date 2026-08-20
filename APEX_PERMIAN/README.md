# Apex Permian synthetic data → Snowflake

**30-well Permian** digital twin → Snowflake. Same multi-tenant schema as Red Bluff (`SENSOR_DATA`); rows isolated by `ORGANIZATION_ID`.

Default org: `APEX_PERMIAN_ORGANIZATION_ID` = `147a92b9-f90a-473d-b443-e6e668c4fbb7`.

## What the Lambda writes

Production path (`apex_permian_pipeline.py` / Lambda) writes **well tables only**:

- `WELL_DAILY_PRODUCTION`
- `WELL_SENSOR_READINGS`

It does **not** write `SYNTH_*` / invoices in the Lambda path (unlike RED_BUFF / STEP). Optional offline scripts can still upload master/invoices parquet to S3.

## Shared Snowflake tenancy

| Tenant | Lambda | Org env | Schedule (code default) |
|--------|--------|---------|-------------------------|
| Red Bluff | `red-buff` | `RED_BUFF_ORGANIZATION_ID` | 00:00 UTC |
| Apex Permian | `apex-permian` | `APEX_PERMIAN_ORGANIZATION_ID` | 00:00 UTC (`cron(0 0 * * ? *)`) |

Override Apex schedule with `SCHEDULE_CRON` if you need a staggered time.

## Run locally

```bash
export ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
# or deploy with APEX_PERMIAN_ORGANIZATION_ID

python -m APEX_PERMIAN.run_incremental_snowflake
python -m APEX_PERMIAN.run_incremental_snowflake --reset
```

Twin script (local / optional S3 upload — needs `S3_BUCKET` for uploads):

```bash
python -m APEX_PERMIAN.generate_digital_twin --organization-id 147a92b9-f90a-473d-b443-e6e668c4fbb7
```

Also available: `generate_master.py`, `generate_invoices.py` for S3 digital_twin / financial_data layouts.

## Deploy

```bash
export APEX_PERMIAN_ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
./APEX_PERMIAN/deploy/schedule.sh
```

| Item | Value |
|------|--------|
| Lambda | `apex-permian` |
| ECR | `apex-permian-lambda` |
| Schedule | `apex-permian-daily` → 00:00 UTC by default |
| Handler | `APEX_PERMIAN.handler.handler` |
| Schema | `AT_RISK_ASSETS.SENSOR_DATA` |

## Layout

```
APEX_PERMIAN/
  generate_digital_twin.py   # Twin engine (Apex sensor set / freq_hz)
  generate_master.py         # Optional S3 master parquet
  generate_invoices.py       # Optional S3 invoices
  apex_permian_pipeline.py   # Incremental → Snowflake (wells only)
  run_incremental_snowflake.py
  column_maps.py
  snowflake_io.py
  handler.py
  deploy/
```

Downstream: `transformations` exports Apex with the `apex_permian` field profile; ML packages use `*_apex.sh` deploy scripts.
