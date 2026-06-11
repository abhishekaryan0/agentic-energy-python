# Anomaly detection

Incremental and bulk sensor anomaly scoring (rules + isolation forest + LSTM), with optional LLM suggestions. Writes detection history parquet to S3.

## Run locally

```bash
# Incremental (default): new rows since checkpoint
python -m anomaly_detection.run

# Bulk backfill
ANOMALY_DETECTION_MODE=bulk python -m anomaly_detection.run

# Train models from local/S3 parquet
python -m anomaly_detection.train_models
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
