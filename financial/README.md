# Financial model

Trains a **RandomForestRegressor** that estimates **implementation cost (USD/day)** per well from daily production and invoice parquet on S3. The artifact is loaded by `operation_recommendations` via `s3_models.load_financial_joblib()`.

There is **no** detection ECS job — training only (local or SageMaker).

## Layout

```
financial/
  train_models.py       # build frame + train/save joblib
  persistence.py        # load + predict
  s3_inputs.py          # S3 invoice + production loaders
  sagemaker_train.py    # SageMaker container entrypoint
  partition_channel.py  # merge date partitions for training
  run.py                # local train-from-S3 (+ optional --upload)
  deploy/
    run_sagemaker_training.sh / _apex.sh / _step.sh
    schedule_sagemaker_training.sh / _apex.sh / _step.sh
    launch_sagemaker_training.py
    Dockerfile.sagemaker.train
    Dockerfile.fargate.sagemaker.launch
    requirements.training.txt
```

## Inputs (S3)

- `{ORG}/sensor_data/daily_production/date=*/part-0000.parquet` (or legacy `daily_production.parquet`)
- `{ORG}/financial_data/run_ts=*/invoices_v1.parquet` and `invoices_v2.parquet` (newest run used)

Training window: `FINANCIAL_TRAIN_PARTITION_DAYS` / `FINANCIAL_TRAINING_PARTITION_DAYS` (default **365**).

## Output

- SageMaker: `s3://{BUCKET}/{ORG}/model/{job}/output/model.tar.gz` containing `financial_implementation_cost.joblib`
- Local: `models/financial_local_<ts>/financial_implementation_cost.joblib`

Joblib payload: `{model, features, target, metadata}` (RF ~300 trees, depth 12; label from workover-like invoice sum / day).

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
./financial/deploy/run_sagemaker_training_apex.sh
./financial/deploy/run_sagemaker_training_step.sh

# Weekly schedules (own EventBridge rules)
./financial/deploy/schedule_sagemaker_training.sh
./financial/deploy/schedule_sagemaker_training_apex.sh
./financial/deploy/schedule_sagemaker_training_step.sh
```

Run `run_sagemaker_training.sh` once first so the training image exists in ECR (`financial-sagemaker-train:latest`). Requires `SAGEMAKER_ROLE_ARN` (and related deploy env from `.env`).
