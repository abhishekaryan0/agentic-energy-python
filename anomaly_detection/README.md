# Anomaly detection

Incremental and bulk sensor anomaly scoring (Red Bluff bands + Isolation Forest + PyOD ECOD/COPOD + MAD + LSTM), with optional LLM suggestions. Writes detection history parquet to S3.

Training filters **PRODUCING** rows, drops all-zero shut-in sensors, and excludes the latest 15% of history by default (`ANOMALY_TRAIN_EXCLUDE_RECENT_PCT`). LSTM `seq_len` defaults to **24** hours; incremental detection history window defaults to **48** rows (`ANOMALY_INCREMENTAL_LAST_N`).

SHUT-IN / all-zero sensor rows are skipped at detection time. Detector output includes sub-scores: `band_score`, `iso_score`, `pyod_score`, `mad_score`, `lstm_score`.

## Run locally

```bash
# Incremental (default): new rows since checkpoint
python -m anomaly_detection.run

# Last N sensor rows (runs LLM on each flagged anomaly when OPENAI_API_KEY is set)
ANOMALY_DETECTION_MODE=recent ANOMALY_LAST_ROWS=100 python -m anomaly_detection.run

# Bulk backfill
ANOMALY_DETECTION_MODE=bulk python -m anomaly_detection.run

# Train models from local/S3 parquet
python -m anomaly_detection.train_models

# Red Bluff SHUT-IN lead-time validation (scores in 14–120 days before failure)
python -m anomaly_detection.validate_red_buff
```

## Deploy (`anomaly_detection/deploy/`)

| File | Purpose |
|------|---------|
| `schedule_detection.sh` | ECS Fargate + EventBridge daily incremental detection |
| `Dockerfile.fargate.detect` | Batch detection container image |
| `ecs_task_definition.json` | ECS task template |
| `requirements.batch.txt` | Python deps for detection batch image |
| `run_sagemaker_training.sh` | SageMaker training job (build image + start) |
| `schedule_sagemaker_training.sh` | Weekly schedule (own EventBridge rule + ECS launcher) |
| `launch_sagemaker_training.py` | ECS launcher: starts one anomaly_detection training job |
| `Dockerfile.sagemaker.train` | SageMaker training container image |
| `Dockerfile.fargate.sagemaker.launch` | ECS launcher container image |

SageMaker training entrypoint: `python -m anomaly_detection.sagemaker_train`
