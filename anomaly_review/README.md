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

# SageMaker training (one-off)
./anomaly_review/deploy/run_sagemaker_training.sh

# Weekly SageMaker training schedule (review only, Mon 02:10 UTC)
./anomaly_review/deploy/schedule_sagemaker_training.sh

# Apex Permian weekly training (Mon 02:15 UTC)
./anomaly_review/deploy/schedule_sagemaker_training_apex.sh
```

## S3 layout

- Input: `s3://BUCKET/{ORG}/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`
- Output: `s3://BUCKET/{ORG}/detections/anomaly_review/run_ts=.../anomaly_reviews.parquet`
- Models: `s3://BUCKET/{ORG}/model/review-train-.../output/model.tar.gz`
