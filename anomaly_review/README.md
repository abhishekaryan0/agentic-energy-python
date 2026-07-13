# Anomaly review

Middle-layer anomaly review: forecast vs actual, reconstruction error, and checkers. Reads partitioned sensor/production data from S3 and writes review parquet to S3.

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | Weekly incremental detection (`python -m anomaly_review.run`) |
| `batch_run.py` | Legacy parallel batch over all wells (local dev) |
| `sagemaker_train.py` | SageMaker training entrypoint |
| `service.py` | Core `detect_anomalies` orchestration |
| `deploy/` | ECS Fargate detection + SageMaker training scripts |

## Deploy

From repo root (with `.env` or `S3_BUCKET`, `ORGANIZATION_ID`):

```bash
# Weekly detection (ECS + EventBridge, Mondays 03:00 UTC)
# First run: ANOMALY_REVIEW_FORCE_WINDOW=1 for 180-day backfill
./anomaly_review/deploy/schedule_detection.sh

# Apex Permian (separate ECS task + schedule)
./anomaly_review/deploy/schedule_detection_apex.sh

# South Texas Energy Partners (separate ECS task + schedule)
./anomaly_review/deploy/schedule_detection_step.sh

# SageMaker training (one-off)
./anomaly_review/deploy/run_sagemaker_training.sh

# Weekly SageMaker training schedule (review only, Mon 02:10 UTC)
./anomaly_review/deploy/schedule_sagemaker_training.sh

# Apex Permian weekly training (Mon 02:15 UTC)
./anomaly_review/deploy/schedule_sagemaker_training_apex.sh

# STEP weekly training (Mon 02:30 UTC)
./anomaly_review/deploy/schedule_sagemaker_training_step.sh
```

## S3 layout

- Input: `s3://BUCKET/{ORG}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`
- Input (training): `s3://BUCKET/{ORG}/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet`
- Output: `s3://BUCKET/{ORG}/detections/anomaly_review/run_ts=.../anomaly_reviews.parquet`
- Models: `s3://BUCKET/{ORG}/model/review-train-.../output/model.tar.gz` (auto-detected at runtime)

## Tenants

| Tenant | Org env | ECS task | Detection schedule (UTC) |
|--------|---------|----------|--------------------------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` | `anomaly-review-detection-batch` | Mon 03:00 |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` | `anomaly-review-detection-apex-permian` | Mon 03:15 |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` | `anomaly-review-detection-step` | Mon 03:30 |

STEP parquet uses lift type `Pumping Unit` (aliased to `Rod Pump` for checkers and ML features).
