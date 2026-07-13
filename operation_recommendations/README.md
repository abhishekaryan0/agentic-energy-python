# Operation recommendations

ML-driven operational action recommendations for wells. Reads partitioned sensor and production data from S3, runs trained models, and writes recommendation parquet to S3.

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | Weekly detection entrypoint (`python -m operation_recommendations.run`) |
| `sagemaker_train.py` | SageMaker training entrypoint |
| `train_models.py` | Local / SageMaker model training |
| `recommendations/` | Analyzer, rules, persistence |
| `deploy/` | ECS Fargate detection + SageMaker training scripts |

## Deploy

From repo root (with `.env` or exported `S3_BUCKET`, `ORGANIZATION_ID`).

- **Detection** loads joblibs directly from `s3://.../operations-train-.../output/model.tar.gz` (auto-detected or `OPERATIONS_MODEL_TARBALL_S3_URI`).
- **Training** reads partitioned sensor data from S3 (`train_action_s3` or SageMaker channels).

```bash
# Train from S3 partitions (local output dir)
python -m operation_recommendations.train_models train_action_s3 --output-dir /tmp/ops_models

# Detection (weekly window from S3 partitions)
python -m operation_recommendations.run
```

Deploy / run:

```bash
# Weekly detection (ECS + EventBridge, Mondays 02:00 UTC)
./operation_recommendations/deploy/schedule_detection.sh

# Apex Permian (separate ECS task + schedule)
./operation_recommendations/deploy/schedule_detection_apex.sh

# SageMaker training (one-off)
./operation_recommendations/deploy/run_sagemaker_training.sh

# Weekly SageMaker training schedule (operations only, Mon 02:20 UTC)
./operation_recommendations/deploy/schedule_sagemaker_training.sh

# Apex Permian weekly training (Mon 02:40 UTC)
./operation_recommendations/deploy/schedule_sagemaker_training_apex.sh

# South Texas Energy Partners
./operation_recommendations/deploy/run_sagemaker_training_step.sh
./operation_recommendations/deploy/schedule_sagemaker_training_step.sh
./operation_recommendations/deploy/schedule_detection_step.sh
```

## S3 layout

- Input: `s3://BUCKET/{ORG}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`
- Output: `s3://BUCKET/{ORG}/detections/operations/run_ts=.../operations_recommendations.parquet`
- Models: `s3://BUCKET/{ORG}/model/operations-train-.../output/model.tar.gz`
