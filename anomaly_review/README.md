# Anomaly review

Weekly middle-layer review: physics/statistical checkers + RF production baseline + LSTM autoencoder reconstruction, with optional LLM enrichment. Reads **partitioned** sensor/production data from S3 and writes review parquet to S3.

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | Weekly incremental detection (`python -m anomaly_review.run`) |
| `batch_run.py` | Legacy parallel batch over all wells (local dev) |
| `service.py` | Core `detect_anomalies` / `save_reviews` |
| `checkers.py` | Domain checkers (fluid pound, pump wear, etc.) |
| `models.py` | RF + LSTM autoencoder train/load/score |
| `config.py` | Thresholds, lift features, suppression hierarchy |
| `data.py` / `partition_data.py` | S3 partition load |
| `charts.py` / `llm.py` | Chart payloads, OpenAI |
| `sagemaker_train.py` | SageMaker training entrypoint |
| `deploy/` | ECS Fargate detection + SageMaker training |

## What it detects

Per well (then suppression / severity):

- Alarm gatekeeper from sensor min/max metadata
- `check_fluid_pound`, `check_pump_wear`, `check_gas_injection_inefficiency`
- `check_energy_efficiency_drift`, `check_energy_production_divergence`
- `check_model_divergence`, `check_pressure_instability`
- Pattern shift / event frequency, monthly production decline
- `check_ml_anomaly` (LSTM reconstruction), `check_sensor_state_mismatch`

STEP lift label `Pumping Unit` is aliased to `Rod Pump` for checkers and ML features.

## Run locally

```bash
python -m anomaly_review.run

# First/full window (e.g. 180 days)
ANOMALY_REVIEW_FORCE_WINDOW=1 python -m anomaly_review.run
```

**Useful env:** `S3_BUCKET`, `ORGANIZATION_ID`, `AWS_REGION`, `ANOMALY_REVIEW_FETCH_DAYS` (180), `ANOMALY_REVIEW_INITIAL_DAYS` (180), `ANOMALY_REVIEW_FORCE_WINDOW`, `ANOMALY_REVIEW_MAX_WORKERS` (5), `ANOMALY_REVIEW_CHECKPOINT_S3_KEY`, `ANOMALY_REVIEW_MODEL_S3_URI`, `OPENAI_*`, `REVIEW_TRAINING_PARTITION_DAYS` (250).

## Deploy

From repo root (with `.env` or `S3_BUCKET`, `ORGANIZATION_ID`):

```bash
./anomaly_review/deploy/schedule_detection.sh
./anomaly_review/deploy/schedule_detection_apex.sh
./anomaly_review/deploy/schedule_detection_step.sh

./anomaly_review/deploy/run_sagemaker_training.sh
./anomaly_review/deploy/schedule_sagemaker_training.sh
./anomaly_review/deploy/schedule_sagemaker_training_apex.sh
./anomaly_review/deploy/schedule_sagemaker_training_step.sh
```

## S3 layout

| Kind | Path |
|------|------|
| Input | `{ORG}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet` |
| Input | `{ORG}/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet` |
| Output | `{ORG}/detections/anomaly_review/run_ts=…/anomaly_reviews.parquet` |
| Models | `{ORG}/model/review-train-…/output/model.tar.gz` |
| Checkpoint | `{ORG}/detections/anomaly_review/checkpoints/last_processed.json` |

## Tenants

| Tenant | Org env | ECS task | Detection schedule (UTC) |
|--------|---------|----------|--------------------------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` | `anomaly-review-detection-batch` | Mon 03:00 |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` | `anomaly-review-detection-apex-permian` | Mon 03:15 |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` | `anomaly-review-detection-step` | Mon 01:15 |

Training schedules: Mon ~02:10 (Red Buff), 02:15 (Apex), 02:30 (STEP).
