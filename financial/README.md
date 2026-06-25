# Financial model

Trains a model that estimates implementation cost (USD/day) per well from daily production and invoice parquet on S3.

## Layout

```
financial/
  train_models.py       # training logic
  persistence.py        # load + predict
  s3_inputs.py          # S3 invoice + production loaders
  sagemaker_train.py    # SageMaker container entrypoint
  partition_channel.py  # merge date partitions for training
  run.py                # local train-from-S3 (+ optional upload)
  deploy/
    run_sagemaker_training.sh
    schedule_sagemaker_training.sh
    launch_sagemaker_training.py
    Dockerfile.sagemaker.train
    Dockerfile.fargate.sagemaker.launch
    requirements.training.txt
```

## Inputs (S3)

- `{ORG}/sensor_data/daily_production/date=*/part-0000.parquet` (or legacy `daily_production.parquet`)
- `{ORG}/financial_data/run_ts=*/invoices_v1.parquet` and `invoices_v2.parquet` (latest run used)

## Output

- SageMaker: `s3://{BUCKET}/{ORG}/model/{job}/output/model.tar.gz` containing `financial_implementation_cost.joblib`
- Local: `models/financial_local_<ts>/financial_implementation_cost.joblib`

`operation_recommendations` loads this artifact at inference via `s3_models.load_financial_joblib()`.

## Local training

```bash
source .env
python -m financial.run
python -m financial.run --upload
```

## SageMaker training

```bash
# One-off (build training image + start job)
./financial/deploy/run_sagemaker_training.sh

# Apex Permian one-off
./financial/deploy/run_sagemaker_training_apex.sh

# Weekly schedule (Mon 02:30 UTC) — own EventBridge rule, starts financial-train-* only
./financial/deploy/schedule_sagemaker_training.sh

# Apex Permian weekly training (Mon 02:50 UTC)
./financial/deploy/schedule_sagemaker_training_apex.sh
```

Run `run_sagemaker_training.sh` once first so the training image exists in ECR (`financial-sagemaker-train:latest`).
