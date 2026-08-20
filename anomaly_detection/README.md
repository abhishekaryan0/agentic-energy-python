# Anomaly detection

Incremental and bulk sensor anomaly scoring (**tenant band profiles + Isolation Forest + PyOD ECOD/COPOD + MAD + LSTM**), with optional LLM suggestions. Red Buff, Apex Permian, and South Texas Energy Partners use separate sensor bands via `tenant_sensors.py`. Writes detection history parquet to S3 (no direct Postgres writes).

Training filters **PRODUCING** rows, drops all-zero shut-in sensors, and excludes the latest 15% of history by default (`ANOMALY_TRAIN_EXCLUDE_RECENT_PCT`). LSTM `seq_len` defaults to **24**; incremental history window defaults to **48** rows (`ANOMALY_INCREMENTAL_LAST_N`).

SHUT-IN / all-zero sensor rows are skipped at detection time. Output includes sub-scores: `band_score`, `iso_score`, `pyod_score`, `mad_score`, `lstm_score`.

**Threshold env vars** (ECS task): `ANOMALY_THRESHOLD` (default 0.75), `ANOMALY_ML_ONLY_THRESHOLD` (0.85), `ANOMALY_RULE_DEVIATION_PCT` (30), `ANOMALY_VIOLATION_MIN_BEYOND_PCT` (0), `ANOMALY_MIN_SCORE_WITH_VIOLATIONS` (0). Apex Permian `schedule_detection_apex.sh` sets stricter defaults (0.85 / 0.92 / 45 / 15 / 0.55).

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | CLI; mode from `ANOMALY_DETECTION_MODE` |
| `bulk_detect.py` | Incremental / recent / bulk runners |
| `detector.py` | `AnomalyDetector`, ensemble scoring |
| `train_models.py` | Local / launch training |
| `sagemaker_train.py` | SageMaker training entrypoint |
| `ml_models.py` | PyOD + MAD helpers |
| `sensor_data.py` / `training_data.py` | Partition load + train filters |
| `s3_history.py` | Write/read detection history parquet |
| `llm_suggestions.py` | Optional OpenAI suggestions |
| `tenant_sensors.py` | Profile resolve (`ANOMALY_TENANT_PROFILE` or org UUID) |
| `validate_red_buff.py` | SHUT-IN lead-time validation CLI |
| `deploy/` | ECS Fargate detect + SageMaker train |

## Run locally

```bash
# Incremental (default): new rows since checkpoint
python -m anomaly_detection.run

# Last N sensor rows (no checkpoint)
ANOMALY_DETECTION_MODE=recent ANOMALY_LAST_ROWS=100 python -m anomaly_detection.run

# Bulk backfill
ANOMALY_DETECTION_MODE=bulk python -m anomaly_detection.run

# Train models from local/S3 parquet
python -m anomaly_detection.train_models

# Red Buff SHUT-IN lead-time validation
python -m anomaly_detection.validate_red_buff
```

Modes: `incremental` | `recent` | `bulk` via `ANOMALY_DETECTION_MODE`.

**Other useful env:** `ANOMALY_WINDOW_MINUTES` (15), `ANOMALY_SAFETY_LAG_SECONDS` (90), `ANOMALY_BOOTSTRAP_ROWS`, `ANOMALY_RESET_CHECKPOINT`, `ANOMALY_BULK_LAST_N`, `ANOMALY_BULK_MAX_WELLS`, `ANOMALY_SENSOR_PARTITION_DAYS` (250), `ANOMALY_MODEL_S3_URI` / latest `anomaly-train-*`, `ANOMALY_ENABLE_LLM_SUGGESTIONS`, `OPENAI_*`, `S3_BUCKET`, `ORGANIZATION_ID`.

## S3 I/O

**Input:** `{ORG}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`  
**Models:** `{ORG}/model/anomaly-train-*/output/model.tar.gz`

**Output:**

| Artifact | Key pattern |
|----------|-------------|
| Well anomalies | `{ORG}/detections/anomaly_detection/history/well_anomalies_run_ts=….parquet` |
| Violations | `{ORG}/detections/anomaly_detection/history/anomaly_violations_run_ts=….parquet` |
| LLM suggestions | `{ORG}/detections/anomaly_detection/anomaly_suggestions_run_ts=….parquet` |
| Checkpoint | `{ORG}/detections/anomaly_detection/checkpoints/last_processed.json` |

Suggestions are synced to Postgres by `s3_to_postgres` (`anomaly_suggestions`). Violations history is written to S3; Postgres sync for violations is not wired in the current handler.

## Tenants

| Tenant | Org ID env | ECS task | Detection schedule (UTC) | Sensor profile |
|--------|------------|----------|--------------------------|----------------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` | `anomaly-detection-batch` | 00:30 | Full canonical parquet, Red Bluff bands |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` | `anomaly-detection-apex-permian` | 00:45 | Apex subset parquet, Permian bands |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` | `anomaly-detection-step` | 01:00 | Full canonical parquet, Eagle Ford bands |

STEP S3 sensor parquet uses lift types `Pumping Unit`, `ESP`, `Gas Lift` (same columns as Red Bluff). Models auto-resolve from the latest `anomaly-train-*` tarball under `s3://{BUCKET}/{STEP_ORGANIZATION_ID}/model/`.

## Deploy (`anomaly_detection/deploy/`)

| File | Purpose |
|------|---------|
| `schedule_detection.sh` | ECS Fargate + EventBridge daily incremental (Red Buff) |
| `schedule_detection_apex.sh` | Apex Permian ECS task |
| `schedule_detection_step.sh` | STEP ECS task |
| `run_detection_ecs*.sh` | One-off / bootstrap runs |
| `Dockerfile.fargate.detect` | Batch detection image |
| `ecs_task_definition.json` | ECS task template |
| `requirements.batch.txt` | Detection deps (pandas, sklearn, pyod, torch, openai, …) |
| `run_sagemaker_training*.sh` / `schedule_sagemaker_training*.sh` | SageMaker train (per tenant) |
| `launch_sagemaker_training.py` | ECS launcher for a training job |
| `Dockerfile.sagemaker.train` | SageMaker training image |

SageMaker entrypoint: `python -m anomaly_detection.sagemaker_train`
