# Anomaly detection

Incremental and bulk sensor anomaly scoring (tenant band profiles + Isolation Forest + PyOD ECOD/COPOD + MAD + LSTM), with optional LLM suggestions. Red Buff and Apex Permian use separate sensor bands via `tenant_sensors.py`. Writes detection history parquet to S3.

Training filters **PRODUCING** rows, drops all-zero shut-in sensors, and excludes the latest 15% of history by default (`ANOMALY_TRAIN_EXCLUDE_RECENT_PCT`). LSTM `seq_len` defaults to **24** hours; incremental detection history window defaults to **48** rows (`ANOMALY_INCREMENTAL_LAST_N`).

SHUT-IN / all-zero sensor rows are skipped at detection time. Detector output includes sub-scores: `band_score`, `iso_score`, `pyod_score`, `mad_score`, `lstm_score`.

**Threshold env vars** (ECS task): `ANOMALY_THRESHOLD` (default 0.75), `ANOMALY_ML_ONLY_THRESHOLD` (0.85), `ANOMALY_RULE_DEVIATION_PCT` (30), `ANOMALY_VIOLATION_MIN_BEYOND_PCT` (0), `ANOMALY_MIN_SCORE_WITH_VIOLATIONS` (0). Apex Permian `schedule_detection_apex.sh` sets stricter defaults (0.85 / 0.92 / 45 / 15 / 0.55).

## Run locally

```bash
# Incremental (default): new rows since checkpoint
python -m anomaly_detection.run

# Last N sensor rows (no checkpoint; loads fewer partition days than incremental)
ANOMALY_DETECTION_MODE=recent ANOMALY_LAST_ROWS=100 python -m anomaly_detection.run

# Incremental bootstrap (first run: last ANOMALY_BOOTSTRAP_ROWS rows, writes checkpoint)
# Apex: ./anomaly_detection/deploy/run_detection_ecs_apex.sh

# Bulk backfill
ANOMALY_DETECTION_MODE=bulk python -m anomaly_detection.run

# Train models from local/S3 parquet
python -m anomaly_detection.train_models

# Red Buff SHUT-IN lead-time validation (scores in 14–120 days before failure)
python -m anomaly_detection.validate_red_buff
```

## Deploy (`anomaly_detection/deploy/`)

| File | Purpose |
|------|---------|
| `schedule_detection.sh` | ECS Fargate + EventBridge daily incremental detection (Red Buff default) |
| `schedule_detection_apex.sh` | Same image, separate ECS task for Apex Permian (`anomaly-detection-apex-permian`) |
| `Dockerfile.fargate.detect` | Batch detection container image |
| `ecs_task_definition.json` | ECS task template |
| `requirements.batch.txt` | Python deps for detection batch image |
| `run_sagemaker_training.sh` | SageMaker training job (build image + start) |
| `schedule_sagemaker_training.sh` | Weekly schedule (own EventBridge rule + ECS launcher) |
| `launch_sagemaker_training.py` | ECS launcher: starts one anomaly_detection training job |
| `Dockerfile.sagemaker.train` | SageMaker training container image |
| `Dockerfile.fargate.sagemaker.launch` | ECS launcher container image |

SageMaker training entrypoint: `python -m anomaly_detection.sagemaker_train`
