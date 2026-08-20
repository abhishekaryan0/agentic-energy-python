# Agentic Energy ML

Multi-tenant oilfield ML monorepo: synthetic digital twins write to Snowflake, `transformations` exports S3 parquet, ML packages score wells, and sync jobs land results in PostgreSQL.

## Data flow

```
Digital twins (RED_BUFF / APEX_PERMIAN / SOUTH_TEXAS_ENERGY_PARTNERS)
        ↓ Snowflake WELL_* (+ SYNTH_* where applicable)
transformations/  →  S3 {ORG}/sensor_data/ (+ digital_twin/ / financial_data/)
        ↓
anomaly_detection / anomaly_review / operation_recommendations / financial (train) / well_status
        ↓ S3 detections/*  |  well_status → Postgres
s3_to_postgres/  →  PostgreSQL schema = {ORGANIZATION_ID}
```

## Packages

| Package | Role | Primary entry points |
|---------|------|----------------------|
| [`RED_BUFF/`](RED_BUFF/README.md) | Red Bluff twin → Snowflake (`SENSOR_DATA`, 60 wells + `SYNTH_*`) | `handler.py`, `pipeline.py`, `run_incremental_snowflake.py` |
| [`APEX_PERMIAN/`](APEX_PERMIAN/README.md) | Apex Permian twin → Snowflake (`SENSOR_DATA`, 30 wells only on Lambda) | `handler.py`, `apex_permian_pipeline.py`, `run_incremental_snowflake.py` |
| [`SOUTH_TEXAS_ENERGY_PARTNERS/`](SOUTH_TEXAS_ENERGY_PARTNERS/README.md) | STEP twin → Snowflake (`SOUTH_TEXAS_SENSOR_DATA`, 100 wells + `SYNTH_*`) | `handler.py`, `pipeline.py`, `run_incremental_snowflake.py` |
| [`transformations/`](transformations/README.md) | Snowflake → org-scoped S3 parquet | `run.py`, `run_synth_export.py`, `handler.py` |
| [`anomaly_detection/`](anomaly_detection/README.md) | Daily sensor anomaly ensemble → S3 | `run.py`, `bulk_detect.py`, `detector.py`, `train_models.py` |
| [`anomaly_review/`](anomaly_review/README.md) | Weekly physics/ML review → S3 | `run.py`, `service.py`, `checkers.py`, `models.py` |
| [`operation_recommendations/`](operation_recommendations/README.md) | Weekly action recommendations + oil-price economics → S3 | `run.py`, `run_pipeline.py`, `train_models.py`, `suggestions.py` |
| [`financial/`](financial/README.md) | Implementation-cost model (SageMaker train only) | `run.py`, `train_models.py`, `persistence.py` |
| [`well_status/`](well_status/README.md) | RUNNING / IDLE / SHUTDOWN → Postgres | `run.py`, `handler.py`, `service.py`, `postgres.py` |
| [`s3_to_postgres/`](s3_to_postgres/README.md) | Detection + production parquet → Postgres | `run.py`, `handler.py`, `sync_*.py` |

Package-level run/deploy detail lives in each package’s `README.md`. Deploy scripts: `./{package}/deploy/schedule*.sh` (tenant variants `*_apex.sh`, `*_step.sh`).

## Tenants

| Tenant | Env | Default org UUID | Snowflake schema | Notes |
|--------|-----|------------------|------------------|-------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` | `3e0b5b18-bec1-488a-82c1-a47881c69058` | `SENSOR_DATA` | Wells + `SYNTH_*` + invoices |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` | `147a92b9-f90a-473d-b443-e6e668c4fbb7` | `SENSOR_DATA` | Lambda writes `WELL_*` only; optional master/invoices via scripts |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` | `d270f89a-aada-4ff2-adb0-68b3e8087565` | `SOUTH_TEXAS_SENSOR_DATA` | Wells + `SYNTH_*`; lift label `Pumping Unit` |

Each org UUID is both the **S3 prefix** and the **PostgreSQL schema**.

### Lift-type aliases

| Source label (parquet / twin) | Treated as | Packages |
|-------------------------------|------------|----------|
| `Pumping Unit` | `Rod Pump` | `anomaly_detection`, `anomaly_review`, `operation_recommendations`, `well_status` |
| `Rod Pump` / `ESP` / `Gas Lift` | As-is | Apex twin and Apex sensor profile |

Red Buff / STEP rod-pump wells are often labeled `Pumping Unit` in Snowflake/S3; ML packages alias them to `Rod Pump` for rules and features. Apex uses native `Rod Pump`.

## Common environment

Shared config lives in repo-root `.env` (not committed). There is no root `requirements.txt` — use each package’s `deploy/requirements*.txt` or Dockerfiles for prod parity.

```bash
# AWS / multi-tenant
S3_BUCKET=...
AWS_REGION=us-east-1
ORGANIZATION_ID=...                    # single tenant
ORGANIZATION_IDS=org-a,org-b,org-c     # multi-tenant loops (transformations, s3_to_postgres)

RED_BUFF_ORGANIZATION_ID=3e0b5b18-bec1-488a-82c1-a47881c69058
APEX_PERMIAN_ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
STEP_ORGANIZATION_ID=d270f89a-aada-4ff2-adb0-68b3e8087565

# Snowflake (twins + transformations)
SNOWFLAKE_USER=...
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_ROLE=...
SNOWFLAKE_TOKEN=...                    # or SNOWFLAKE_PASSWORD
SNOWFLAKE_DATABASE=AT_RISK_ASSETS
SNOWFLAKE_SCHEMA=SENSOR_DATA
STEP_SNOWFLAKE_SCHEMA=SOUTH_TEXAS_SENSOR_DATA

# Postgres (well_status, s3_to_postgres, ops oil price)
DATABASE_URL=postgresql://...          # or DATABASE_URL_SECRET_ARN

# Optional
OPENAI_API_KEY=...                     # anomaly / ops suggestions
EIA_API_KEY=...                        # Brent / WTI for operations
SAGEMAKER_ROLE_ARN=...                 # training jobs
```

## Local smoke path

From repo root (with `.env` loaded):

```bash
# 1) Twin → Snowflake (pick one tenant)
python RED_BUFF/run_incremental_snowflake.py
# python -m APEX_PERMIAN.run_incremental_snowflake
# python SOUTH_TEXAS_ENERGY_PARTNERS/run_incremental_snowflake.py

# 2) Snowflake → S3
python -m transformations.run --mode incremental
# optional SYNTH master/invoices (Red Buff / STEP style):
# python -m transformations.run_synth_export --mode incremental

# 3) ML / status
python -m anomaly_detection.run
python -m well_status.run
# python -m anomaly_review.run
# python -m operation_recommendations.run
# python -m financial.run

# 4) S3 → Postgres
python -m s3_to_postgres.run --sync-mode incremental --table all
```

---

## Schedules (deploy script defaults, UTC)

Source of truth: `*/deploy/schedule*.sh`. Override with `SCHEDULE_CRON` / `LAMBDA_FUNCTION_NAME` / `TASK_FAMILY`.

### Daily

| UTC | Job | Name | Type | Deploy script |
|-----|-----|------|------|---------------|
| 00:00 | Red Bluff twin | `red-buff` | Lambda | `RED_BUFF/deploy/schedule.sh` |
| 00:00 | Apex twin | `apex-permian` | Lambda | `APEX_PERMIAN/deploy/schedule.sh` |
| 00:30 | STEP twin | `south-texas-energy-partners` | Lambda | `SOUTH_TEXAS_ENERGY_PARTNERS/deploy/schedule.sh` |
| 00:30 | Anomaly detection (Red Buff) | `anomaly-detection-batch` | ECS | `anomaly_detection/deploy/schedule_detection.sh` |
| 00:45 | Transformations (Snowflake → S3) | `transformations` | Lambda | `transformations/deploy/schedule.sh` |
| 00:45 | Anomaly detection (Apex) | `anomaly-detection-apex-permian` | ECS | `…/schedule_detection_apex.sh` |
| 01:00 | Anomaly detection (STEP) | `anomaly-detection-step` | ECS | `…/schedule_detection_step.sh` |
| 02:00 | Well status (Red Buff) | `well-status` | Lambda | `well_status/deploy/schedule.sh` |
| 02:15 | Well status (Apex) | `well-status-apex-permian` | Lambda | `…/schedule_apex.sh` |
| 02:45 | Well status (STEP) | `well-status-step` | Lambda | `…/schedule_step.sh` |
| 03:30 | S3 → Postgres sync | `s3-to-postgres-sync` | Lambda | `s3_to_postgres/deploy/schedule.sh` |

### Monday weekly (train + review/ops)

| UTC | Job | ECS family | Deploy script |
|-----|-----|------------|---------------|
| 01:15 | Review detection (STEP) | `anomaly-review-detection-step` | `anomaly_review/deploy/schedule_detection_step.sh` |
| 01:30 | Ops detection (STEP) | `operations-detection-step` | `operation_recommendations/deploy/schedule_detection_step.sh` |
| 02:00 | Anomaly train (Red Buff) | `anomaly-detection-sagemaker-train-launch` | `anomaly_detection/deploy/schedule_sagemaker_training.sh` |
| 02:00 | Ops detection (Red Buff) | `operations-detection-batch` | `operation_recommendations/deploy/schedule_detection.sh` |
| 02:10 | Review train (Red Buff) | `anomaly-review-sagemaker-train-launch` | `anomaly_review/deploy/schedule_sagemaker_training.sh` |
| 02:15 | Review train (Apex) | `anomaly-review-apex-sagemaker-train-launch` | `…/schedule_sagemaker_training_apex.sh` |
| 02:20 | Ops train (Red Buff) | `operations-sagemaker-train-launch` | `operation_recommendations/deploy/schedule_sagemaker_training.sh` |
| 02:30 | Anomaly train (Apex) | `anomaly-detection-apex-sagemaker-train-launch` | `anomaly_detection/deploy/schedule_sagemaker_training_apex.sh` |
| 02:30 | Review train (STEP) | `anomaly-review-step-sagemaker-train-launch` | `anomaly_review/deploy/schedule_sagemaker_training_step.sh` |
| 02:30 | Financial train (Red Buff) | `financial-sagemaker-train-launch` | `financial/deploy/schedule_sagemaker_training.sh` |
| 02:40 | Ops train (Apex) | `operations-apex-sagemaker-train-launch` | `…/schedule_sagemaker_training_apex.sh` |
| 02:50 | Financial train (Apex) | `financial-apex-sagemaker-train-launch` | `financial/deploy/schedule_sagemaker_training_apex.sh` |
| 03:00 | Review detection (Red Buff) | `anomaly-review-detection-batch` | `anomaly_review/deploy/schedule_detection.sh` |
| 03:00 | Ops detection (Apex) | `operations-detection-apex-permian` | `operation_recommendations/deploy/schedule_detection_apex.sh` |
| 03:00 | Anomaly train (STEP) | `anomaly-detection-step-sagemaker-train-launch` | `anomaly_detection/deploy/schedule_sagemaker_training_step.sh` |
| 03:10 | Ops train (STEP) | `operations-step-sagemaker-train-launch` | `…/schedule_sagemaker_training_step.sh` |
| 03:15 | Review detection (Apex) | `anomaly-review-detection-apex-permian` | `anomaly_review/deploy/schedule_detection_apex.sh` |
| 03:30 | Financial train (STEP) | `financial-step-sagemaker-train-launch` | `financial/deploy/schedule_sagemaker_training_step.sh` |

---

## Key files by package

### Digital twins

| Package | Key files |
|---------|-----------|
| `RED_BUFF/` | `generate_digital_twin.py`, `generate_invoices.py`, `build_master.py`, `snowflake_io.py`, `column_maps.py`, `check_snowflake.py`, `web/app.py` (dev UI) |
| `APEX_PERMIAN/` | `generate_digital_twin.py`, `generate_master.py`, `generate_invoices.py`, `apex_permian_pipeline.py`, `snowflake_io.py`, `column_maps.py` |
| `SOUTH_TEXAS_ENERGY_PARTNERS/` | `generate_digital_twin.py`, `generate_invoices.py`, `build_master.py`, `config.py`, `snowflake_io.py`, `column_maps.py`, `check_snowflake.py` |

Snowflake tables (names shared; STEP uses a different schema): `WELL_DAILY_PRODUCTION`, `WELL_SENSOR_READINGS`, and (Red Buff / STEP) `SYNTH_ASSETS`, `SYNTH_PERSONNEL`, `SYNTH_ASSET_DAILY_HISTORY`, `SYNTH_PIPELINE_STATE`, `SYNTH_INVOICES`.

### `transformations/`

| File | Purpose |
|------|---------|
| `run.py` | Sensor / daily production export CLI |
| `run_synth_export.py` | SYNTH master + invoices export CLI |
| `export.py` / `synth_export.py` | Export orchestration |
| `columns.py` / `field_mappings.py` | Canonical columns + per-tenant profiles (`apex_permian` vs full) |
| `org_snowflake.py` | Per-org Snowflake schema routing |
| `handler.py` | Lambda entry (`mode`, `organization_id`, `synth: true`) |

### `anomaly_detection/`

| File | Purpose |
|------|---------|
| `run.py` | CLI (`ANOMALY_DETECTION_MODE`: incremental / recent / bulk) |
| `bulk_detect.py` | Detection runners |
| `detector.py` | Ensemble scoring |
| `ml_models.py` | PyOD + MAD helpers |
| `tenant_sensors.py` | Profile resolve (`ANOMALY_TENANT_PROFILE` or org UUID) |
| `red_buff_sensors.py` / `apex_permian_sensors.py` / `step_sensors.py` | Per-tenant bands |
| `train_models.py` / `sagemaker_train.py` | Training |
| `s3_history.py` / `llm_suggestions.py` | History parquet + optional suggestions |
| `validate_red_buff.py` | SHUT-IN lead-time validation |

### `anomaly_review/`

| File | Purpose |
|------|---------|
| `run.py` | Weekly incremental detection |
| `service.py` | `detect_anomalies` / `save_reviews` |
| `checkers.py` | Domain checkers (fluid pound, pump wear, pressure instability, …) |
| `models.py` | RF + LSTM autoencoder |
| `data.py` / `partition_data.py` | S3 partition load |
| `charts.py` / `llm.py` | Chart payloads, OpenAI |
| `sagemaker_train.py` | SageMaker training entrypoint |

### `operation_recommendations/`

| File | Purpose |
|------|---------|
| `run.py` | Weekly checkpointed detection |
| `run_pipeline.py` | Local train + recommend pipeline (dev) |
| `train_models.py` / `sagemaker_train.py` | Action / category models |
| `recommendations/analyzer.py` | Metrics from parquet |
| `recommendations/rules.py` | Rules + impact |
| `recommendations/persistence.py` | Recommend, economics, S3 write |
| `org_settings.py` / `oil_price.py` / `eia_prices.py` | Oil $/bbl (Manual / Brent / WTI) |
| `tenant_config.py` | Per-tenant sensor lists |
| `suggestions.py` | Optional OpenAI suggestion text |
| `s3_inputs.py` / `s3_models.py` | Parquet + model loaders |

### `financial/`

| File | Purpose |
|------|---------|
| `run.py` | Local train-from-S3 (+ optional `--upload`) |
| `train_models.py` | Build frame + train/save joblib |
| `persistence.py` | Load + predict (used by operations) |
| `s3_inputs.py` / `partition_channel.py` | Invoice + production loaders |
| `sagemaker_train.py` | SageMaker container entrypoint |

Artifact: `financial_implementation_cost.joblib` inside `model/financial-train-…/output/model.tar.gz`.

### `well_status/`

| File | Purpose |
|------|---------|
| `handler.py` / `run.py` | Lambda / CLI |
| `service.py` | Status computation (S3 + idle rules) |
| `postgres.py` | Upsert / reads |
| `config.py` | Settings |

Statuses: **SHUTDOWN** (no reading > 24h), **IDLE** (lift-specific idle score), **RUNNING**.

### `s3_to_postgres/`

| File | Purpose |
|------|---------|
| `handler.py` / `run.py` | Lambda / CLI (`SYNC_HANDLERS`) |
| `sync_anomaly_suggestions.py` | → `{ORG}.anomaly_suggestions` |
| `sync_anomaly_reviews.py` | → `{ORG}.anomaly_reviews` |
| `sync_operation_recommendations.py` | → `{ORG}.operation_recommendations` |
| `sync_daily_production.py` | → `{ORG}.daily_production` |
| `sync_monthly_production.py` | → `{ORG}.monthly_production` |
| `sync_anomaly_violations.py` | Present; **not** wired to handler/CLI |
| `db.py` / `s3_io.py` / `checkpoint.py` / `sync_common.py` | Shared helpers |

CLI `--table` choices: `anomaly_suggestions`, `anomaly_reviews`, `operation_recommendations`, `daily_production`, `monthly_production`, `all`.

---

## ML models overview

### `anomaly_detection/`

Ensemble on well sensor parquet: **tenant sensor bands + Isolation Forest + PyOD (ECOD/COPOD) + MAD + LSTM autoencoder**, optional OpenAI suggestions.

| Component | Library / model | Role |
|-----------|-----------------|------|
| Rules | Tenant bands (`tenant_sensors.py`) | Domain min/max / deviation |
| Outliers | scikit-learn `IsolationForest` | Multivariate point outliers |
| Density | PyOD ECOD / COPOD | Additional unsupervised scores |
| Stats | MAD | Robust univariate deviation |
| Sequence | PyTorch LSTM autoencoder | Temporal reconstruction error |

Weighted ensemble (default ≈ rules 0.30, iso 0.15, pyod 0.20, mad 0.10, lstm 0.25). Output sub-scores: `band_score`, `iso_score`, `pyod_score`, `mad_score`, `lstm_score`.

### `anomaly_review/`

Weekly “worth human review” layer: physics/statistical checkers + RF production baseline + LSTM autoencoder.

| Component | Library / model | Role |
|-----------|-----------------|------|
| Production baseline | `RandomForestRegressor` | Expected oil / flow |
| Pattern anomaly | PyTorch `LSTMAutoencoder` | High reconstruction error |
| Checkers | Domain rules in `checkers.py` | Fluid pound, pump wear, pressure instability, etc. |

### `operation_recommendations/`

Rules + calibrated **RandomForestClassifier** action/category models. Economics use financial cost model + oil price from Postgres `{ORG}.org_settings` (Manual / Brent / WTI via EIA). Categories: Production / Optimization / Maintenance.

### `financial/`

**RandomForestRegressor** predicting implementation cost USD/day from production + invoices. Consumed by operations at inference (no detection ECS of its own).

---

## S3 Parquet datasets (folder layout + fields)

Org-scoped prefixes: `s3://$S3_BUCKET/$ORGANIZATION_ID/`.

- **Primary export:** `transformations/` (`python -m transformations.run`)
- **Readers:** anomaly detection/review, operations, well_status, financial training
- **Master / invoices:** SYNTH export (`python -m transformations.run_synth_export`) or Apex `generate_master.py` / twin scripts

### Sensor data (Snowflake → S3)

**Partitioned (preferred)**

- `…/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`
- `…/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet`

**Legacy root snapshots** (optional; `TRANSFORM_WRITE_SNAPSHOTS=1`)

- `…/sensor_data/well_sensor_readings.parquet`
- `…/sensor_data/daily_production.parquet`

**Checkpoint:** `…/sensor_data/_state/export_checkpoint.json`

#### `well_sensor_readings` columns

Canonical set from `transformations/columns.py` (`WELL_SENSOR_READINGS_COLUMNS`), plus Apex alias `freq_hz` (coalesced with `drive_frequency`).

- **Identity / time:** `organization_id`, `well_id`, `timestamp`, `lift_type`, `well_status`
- **Rod pump / mechanical:** `strokes_per_minute`, `torque`, `polish_rod_load`, `pump_fillage`, `surface_stroke_length`, `downhole_gross_stroke`, `downhole_net_stroke`, `runtime`, `cycles`, `structural_load`, `inferred_production`, `pump_intake_pressure`, `dry_rod_weight`, `buoyant_rod_weight`, `pump_friction`, `pump_diameter`, `barrel_length`, `pump_depth`, `idle_time_setpoint`
- **ESP / electrical:** `motor_temp`, `motor_current`, `discharge_pressure`, `motor_voltage`, `intake_fluid_temp`, `vibration_x`, `vibration_y`, `discharge_temp`, `downhole_flow_rate`, `drive_frequency`, `tubing_pressure`, `casing_pressure`, `drive_input_voltage`, `output_voltage`, `input_current`, `motor_speed`, `vsd_temp`, `total_harmonic_distortion`, `motor_load`, `set_frequency`, `acceleration_ramp_time`
- **Gas lift / plunger-style:** `injection_rate`, `injection_temperature`, `bottomhole_pressure`, `injection_pressure`, `cycle_time`, `bottomhole_temp`, `plunger_arrival_time`, `plunger_velocity`, `arrival_count`, `missed_arrivals`, `shutin_time`, `afterflow_time`, `flow_time`, `plunger_drop_time`, `min_shutin_pressure`, `max_shutin_pressure`, `open_diff_pressure`, `well_open`, `well_close`, `velocity_limit`, `wellhead_temp`, `surface_pressure`
- **Alias:** `freq_hz`

#### `daily_production` columns

From `WELL_DAILY_PRODUCTION_COLUMNS`: `organization_id`, `well_id`, `date`, `oil_volume`, `gas_volume`, `water_volume`, `boe`, `lift_type`, `status`, `notes`, `test_oil`, `test_water`, `test_gas`, `facility_id`

### Master / financial (digital twin)

From SYNTH export or Apex master scripts:

- `…/digital_twin/master_data/run_ts=…/master_assets.parquet`
- `…/digital_twin/master_data/run_ts=…/master_history.parquet`
- `…/financial_data/run_ts=…/invoices_v2.parquet` (also `invoices_v1.parquet` where present)

Checkpoint: `{ORG}/synth_export/_state/export_checkpoint.json`

### Detection outputs (ML → S3)

| Package | Path |
|---------|------|
| Anomaly detection | `{ORG}/detections/anomaly_detection/history/well_anomalies_run_ts=….parquet` |
| Anomaly violations | `{ORG}/detections/anomaly_detection/history/anomaly_violations_run_ts=….parquet` (S3 only until sync is wired) |
| Anomaly suggestions | `{ORG}/detections/anomaly_detection/anomaly_suggestions_run_ts=….parquet` |
| Anomaly review | `{ORG}/detections/anomaly_review/run_ts=…/anomaly_reviews.parquet` |
| Operations | `{ORG}/detections/operations/run_ts=…/operations_recommendations.parquet` |
| Models | `{ORG}/model/{anomaly\|review\|operations\|financial}-train-…/output/model.tar.gz` |
| Checkpoints | `{ORG}/detections/{anomaly_detection\|anomaly_review\|operations}/checkpoints/last_processed.json` |
| Well status checkpoint | `{ORG}/sensor_data/_state/well_status_checkpoint.json` |

### Postgres sync (`s3_to_postgres`)

| Postgres table | Sync module | S3 source |
|----------------|-------------|-----------|
| `anomaly_suggestions` | `sync_anomaly_suggestions.py` | `{ORG}/detections/anomaly_detection/` |
| `anomaly_reviews` | `sync_anomaly_reviews.py` | `{ORG}/detections/anomaly_review/` |
| `operation_recommendations` | `sync_operation_recommendations.py` | `{ORG}/detections/operations/` |
| `daily_production` | `sync_daily_production.py` | `{ORG}/sensor_data/daily_production/date=…/` |
| `monthly_production` | `sync_monthly_production.py` | Aggregated from daily |
| `well_status` | written by `well_status/` (not this sync) | Latest sensor activity |

```bash
python -m s3_to_postgres.run --sync-mode incremental --table all
python -m s3_to_postgres.run --sync-mode backfill --table daily_production --backfill-years 2
```

Skip tables via `SYNC_SKIP_TABLES` (comma-separated).
