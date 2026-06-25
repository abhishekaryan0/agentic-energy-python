# Apex Permian synthetic data → Snowflake

Same **data fields and shapes** as `APEX_PERMIAN/generate_digital_twin.py`.  
Multi-tenant: one Snowflake schema (`SENSOR_DATA`), rows isolated by `ORGANIZATION_ID`.

## Lambdas (two tenants)

| Tenant | Lambda | Org env | Schedule |
|--------|--------|---------|----------|
| Red Bluff | `red-buff` | `RED_BUFF_ORGANIZATION_ID` | 00:00 UTC |
| Apex Permian | `apex-permian` | `APEX_PERMIAN_ORGANIZATION_ID` | 00:15 UTC |

## Run locally

```bash
export APEX_PERMIAN_ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
python -m APEX_PERMIAN.run_incremental_snowflake
python -m APEX_PERMIAN.run_incremental_snowflake --reset
```

```bash
python -m APEX_PERMIAN.generate_digital_twin --organization-id 147a92b9-...
```

## Deploy

```bash
export APEX_PERMIAN_ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
./APEX_PERMIAN/deploy/schedule.sh
```

## Layout

```
APEX_PERMIAN/
  generate_digital_twin.py   # Apex Permian twin engine
  generate_master.py
  generate_invoices.py
  apex_permian_pipeline.py   # incremental → Snowflake
  run_incremental_snowflake.py
  column_maps.py             # scripts parquet shape → Snowflake
  snowflake_io.py
  handler.py
  deploy/
```
