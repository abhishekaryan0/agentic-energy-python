# Deploy: SageMaker training + ECS/Fargate (AWS CLI)

Deployment uses **AWS CLI only**. SageMaker is used **only for training jobs**; online inference is handled by Flask/ECS/Fargate, not SageMaker endpoints or Lambda.

**Models on S3 and detection:** Trained models are stored on S3 in their own directory; detection jobs (operations, anomaly review, well status) read S3 parquet and model tarballs directly. See [docs/MODELS_S3_AND_DETECTION.md](../docs/MODELS_S3_AND_DETECTION.md) for the flow.

## Train locally and upload models to S3 (required)

**S3_BUCKET** must be set in your `.env`. Every time you run anomaly detection training (locally or via `automate/run_training.py`), models are saved to `models/anomaly_detection/` and then **required** to be packed and uploaded to S3; training exits with an error if S3_BUCKET is missing or upload fails.

**.env (required):**
```bash
S3_BUCKET=your-models-bucket
# optional:
# S3_PREFIX=models/anomaly
# MODEL_VERSION=v1
```

Run training from repo root:
```bash
python -m anomaly_detection.train_models
# or scheduled: python automate/run_training.py
```

Upload goes to `s3://BUCKET/PREFIX/anomaly_detection_VERSION.tar.gz`. AWS CLI must be configured. Then deploy with `./deploy/deploy_sagemaker_lambda.sh` (it will use the uploaded tarball).

## SageMaker training (sensor data from S3)

To train anomaly detection **on SageMaker** using sensor data stored in S3 (e.g. from `scripts/generate_digital_twin.py --s3`):

1. **Data:** Ensure parquet exists at `s3://BUCKET/{ORGANIZATION_ID}/sensor_data/well_sensor_readings.parquet`.
2. **Run training job** (from repo root):

   ```bash
   export S3_BUCKET=agentic-energy-ml-models-200937443240
   export ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
   ./deploy/run_sagemaker_training.sh
   ```

3. **Output:** When the job completes, the model is at  
   `s3://BUCKET/{ORGANIZATION_ID}/model/<training_job_name>/output/model.tar.gz`  
   (same layout as local training: `anomaly_detection/wells/<safe_well_id>/`).

Optional env: `TRAINING_JOB_NAME`, `INSTANCE_TYPE` (default `ml.m5.xlarge`), `PER_WELL_LIMIT` (default 5000), `AWS_REGION`.

**Local training from parquet (no SageMaker):** You can also train from a parquet file (local or `s3://`) and write to a directory:

```python
from anomaly_detection.train_models import train_all_models_from_parquet
train_all_models_from_parquet("/path/to/well_sensor_readings.parquet", output_dir="models", per_well_limit=5000)
```

**Anomaly review (SageMaker training from S3):** Same bucket layout; uses `daily_production.parquet` to train per-well RF + autoencoder. Run:

```bash
export S3_BUCKET=... ORGANIZATION_ID=...
./deploy/run_sagemaker_training_review.sh
```

Output: `s3://BUCKET/{ORG_ID}/model/review-train-.../output/model.tar.gz` (per-well dirs at top level).

**Operations (SageMaker training from S3):** Uses `well_sensor_readings.parquet` and `daily_production.parquet` to train action recommendation models (per lift-type + fallback). Run:

```bash
export S3_BUCKET=... ORGANIZATION_ID=...
./deploy/run_sagemaker_training_operations.sh
```

Output: `s3://BUCKET/{ORG_ID}/model/operations-train-.../output/model.tar.gz` (decision_tree_*.joblib, etc.).

## Required

- **S3_BUCKET** – Required for deploy and for training (training always uploads to S3 after saving models). All other variables have defaults.
- **AWS CLI** configured (credentials, region).
- **Docker** (for building the SageMaker image unless `SKIP_ECR_BUILD=1`).
- **Python 3** (for `pack_and_upload_sagemaker_model.py`).

## Optional env overrides

| Variable | Default |
|----------|---------|
| AWS_REGION | us-east-1 |
| ECR_REPO_NAME | anomaly-detection-sagemaker |
| SAGEMAKER_MODEL_NAME | anomaly-detection-model |
| SAGEMAKER_ENDPOINT_NAME | anomaly-detection-endpoint |
| SAGEMAKER_INSTANCE_TYPE | ml.m5.xlarge |
| LAMBDA_FUNCTION_NAME | anomaly-detection-invoke |
| MODEL_VERSION | v1 |
| SKIP_ECR_BUILD | 0 (set to 1 in CI to use existing image) |

## After deploy

- Wait for the SageMaker endpoint to be **InService** (5–10 min):  
  `aws sagemaker describe-endpoint --endpoint-name anomaly-detection-endpoint --region $AWS_REGION`
- Invoke Lambda (use raw payload; set region if needed):  
  `aws lambda invoke --function-name anomaly-detection-invoke --cli-binary-format raw-in-base64-out --payload '{"readings":{"motor_temp":100},"lift_type":"ESP"}' --region us-east-1 /tmp/out.json && cat /tmp/out.json`

## CI (GitHub Actions)

Set secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ACCOUNT_ID`, `ECR_REPOSITORY`, and optionally `S3_BUCKET`.  
If `S3_BUCKET` is set, the workflow runs the full deploy script with `SKIP_ECR_BUILD=1` after building and pushing the image.

## If model/endpoint/Lambda are missing (script failed partway)

1. **Upload model tarball to S3** (from repo root; need `models/anomaly_detection/` or script creates minimal):
   ```bash
   python3 scripts/pack_and_upload_sagemaker_model.py --bucket agentic-energy-ml-models-200937443240 --prefix models/anomaly --version v1
   ```
2. **Re-run full deploy** so SageMaker model, endpoint, and Lambda get created (image will be rebuilt with Docker V2 manifest):
   ```bash
   cd /path/to/agentic-energy-ml
   export AWS_REGION=us-east-1
   export AWS_ACCOUNT_ID=200937443240
   export S3_BUCKET=agentic-energy-ml-models-200937443240
   ./deploy/deploy_sagemaker_lambda.sh
   ```
3. Wait for endpoint **InService**, then invoke Lambda with `--cli-binary-format raw-in-base64-out` (see above).

## Anomaly review on SageMaker (second endpoint)

The **anomaly review** pipeline (per-well: fetch data, run physics/ML checkers, LLM, optional DB write) can be deployed as a separate SageMaker endpoint so you can invoke it as a model.

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET=your-models-bucket

# Optional: for full pipeline (Snowflake + Postgres + OpenAI) inside the container
export DATABASE_URL=postgresql://...
export OPENAI_API_KEY=sk-...
export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...  # etc.

./deploy/deploy_sagemaker_review.sh
```

This script will:

1. **ECR** – Create repo `anomaly-review-sagemaker`, build `Dockerfile.sagemaker.review`, push.
2. **S3** – Pack per-well models from `models/` (excluding `anomaly_detection`) and upload.
3. **SageMaker** – Create model + endpoint `anomaly-review-endpoint` (reuses same SageMaker execution role if present).
4. **Lambda** – Create `anomaly-review-invoke` that calls the review endpoint.

**Input:** `{"well_id": "Well_001", "save_to_db": false}`  
**Output:** `{"well_id": "...", "anomalies": [...], "count": N}`

- Pack review models only:  
  `python3 scripts/pack_and_upload_review_models.py --bucket $S3_BUCKET --prefix models/review --version v1`
- Overrides: `ECR_REPO_NAME_REVIEW`, `SAGEMAKER_ENDPOINT_NAME_REVIEW`, `LAMBDA_FUNCTION_NAME_REVIEW`, etc.

## Operations detector on SageMaker

The **operational recommendations** detector can be deployed as a separate SageMaker endpoint (metrics-in, no Snowflake in container):

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET=your-models-bucket
# Train first: python -m operations.train_models train_category && python -m operations.train_models train_action
./deploy/deploy_sagemaker_operations.sh
```

**Input:** `{"well_id": "Well_001", "lift_type": "ESP", "metrics": { "motor_temp_slope": 0.1, ... } }`  
**Output:** `{"well_id": "...", "recommendations": [...], "count": N}`

- Pack operations models: `python3 scripts/pack_and_upload_operations_models.py --bucket $S3_BUCKET --prefix models/operations --version v1`
- The Flask app has been removed; operations inference should run via scheduled jobs or dedicated services.

## Operations detection on schedule (production)

To run **operations detection** (ML + OpenAI suggestions) on a **schedule** and write results to **S3 as Parquet** (no Postgres), use ECS Fargate + EventBridge:

1. **One-time setup** (from repo root; uses `.env` if present):

   ```bash
   export S3_BUCKET=agentic-energy-ml-models-200937443240
   export ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
   export OPERATIONS_MODEL_TARBALL_S3_URI="s3://agentic-energy-ml-models-200937443240/147a92b9-f90a-473d-b443-e6e668c4fbb7/model/operations-train-20260309-1351/output/model.tar.gz"
   export OPENAI_API_KEY=sk-...
   export OPENAI_API_BASE=https://openrouter.ai/api/v1   # optional

   ./deploy/schedule_operations_detection.sh
   ```

2. **What the script does**:
   - Builds `Dockerfile.fargate.operations.detect` and pushes to ECR (`operations-fargate-detect`).
   - Creates ECS cluster `operations-detection-cluster`, log group `/ecs/operations-detection`, and IAM roles.
   - Registers Fargate task definition `operations-detection-batch` with your env vars.
   - Creates an **EventBridge rule** `operations-detection-daily` that runs the task **daily at 02:00 UTC** (`cron(0 2 * * ? *)`).

3. **Optional overrides**:
   - `SCHEDULE_CRON="cron(0 6 * * ? *)"` – run at 6 AM UTC.
   - `ECS_CLUSTER=my-cluster` – use another cluster.
   - `ECS_SUBNETS=subnet-1,subnet-2` and `ECS_SECURITY_GROUPS=sg-1` – use specific VPC subnets and security groups (e.g. private subnets + NAT for OpenAI).
   - `SKIP_ECR_BUILD=1` – do not rebuild/push the image (e.g. in CI).

4. **Run once manually** (after deploy):
   ```bash
   aws ecs run-task --cluster operations-detection-cluster --task-definition operations-detection-batch \
     --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
     --region us-east-1
   ```

5. **Output**: Parquet files under  
   `s3://BUCKET/ORG_ID/detections/operations/run_ts=YYYYMMDDTHHMMSSZ/operations_recommendations.parquet`  
   (includes ML recommendations + OpenAI-generated suggestion fields).

6. **Logs**: CloudWatch log group `/ecs/operations-detection`.

## Well status on schedule (production)

To run **well status update** (read sensor parquet from S3, compute RUNNING/IDLE/SHUTDOWN, write `well_status.parquet` to S3) **daily** via EventBridge + ECS Fargate:

1. **One-time setup** (from repo root; uses `.env` if present):

   ```bash
   export S3_BUCKET=agentic-energy-ml-models-200937443240
   export ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
   ./deploy/schedule_well_status.sh
   ```

2. The script will: build and push `Dockerfile.fargate.well_status` to ECR, create ECS cluster `well-status-cluster`, register task `well-status-batch`, and create EventBridge rule **well-status-daily** (default: `cron(0 2 * * ? *)` = daily 02:00 UTC).

3. **Optional**: `SCHEDULE_CRON`, `ECS_CLUSTER`, `AWS_REGION`, `SKIP_ECR_BUILD=1`.

4. **Run once manually**:  
   `aws ecs run-task --cluster well-status-cluster --task-definition well-status-batch --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[...],securityGroups=[...],assignPublicIp=ENABLED}" --region us-east-1`

5. **Logs**: CloudWatch log group `/ecs/well-status`.

## Anomaly review (S3 only, no Snowflake)

Anomaly review uses **S3 only** for input and output (Snowflake removed).

**Input (same as operations and well status):**
- `s3://BUCKET/{ORGANIZATION_ID}/sensor_data/well_sensor_readings.parquet`
- `s3://BUCKET/{ORGANIZATION_ID}/sensor_data/daily_production.parquet`

**Output:**
- `s3://BUCKET/{ORGANIZATION_ID}/detections/anomaly_review/run_ts=YYYYMMDDTHHMMSSZ/anomaly_reviews.parquet`

**Inference:** ECS/SageMaker or Flask call `detect_anomalies(well_id)`; data is loaded from S3 via `fetch_well_data(well_id)`. Set `S3_BUCKET` and `ORGANIZATION_ID` (or `ANOMALY_REVIEW_INPUT_S3_PREFIX` / `ANOMALY_REVIEW_OUTPUT_S3_PREFIX`).

**Batch run (local or Fargate):**
```bash
export S3_BUCKET=your-bucket ORGANIZATION_ID=your-org-id
python scripts/run_anomaly_review_s3.py
# or: python -m anomaly_review.run
```

**History:** Prior anomaly events for a well (e.g. for LLM context) are read from existing run parquets under the output prefix (no separate state table).

**Optional:** `ANOMALY_REVIEW_SENSOR_DEFINITIONS_S3_URI` = `s3://bucket/key` to a parquet/CSV with columns `field_name`, `min_value`, `max_value` for sensor limits.

## S3 data layout summary

| Use case              | Input                                                                 | Output |
|------------------------|-----------------------------------------------------------------------|--------|
| Operations detection   | `{ORG}/sensor_data/well_sensor_readings.parquet`, `daily_production.parquet` | `{ORG}/detections/operations/run_ts=.../operations_recommendations.parquet` |
| Well status            | `{ORG}/sensor_data/well_sensor_readings.parquet`                     | `{ORG}/sensor_data/well_status.parquet` (or similar) |
| Anomaly review         | Same sensor + production parquet                                     | `{ORG}/detections/anomaly_review/run_ts=.../anomaly_reviews.parquet` |

## Files

- **Dockerfile** – Flask app (legacy/local).
- **Dockerfile.fargate.operations.detect** – Fargate batch image for operations detection (S3 in/out, OpenAI).
- **Dockerfile.fargate.well_status** – Fargate batch image for well status update (S3 in/out).
- **Dockerfile.sagemaker.train** – SageMaker training container for anomaly detection.
- **Dockerfile.sagemaker.train.review** – SageMaker training container for anomaly review.
- **Dockerfile.sagemaker.train.operations** – SageMaker training container for operations.
- **ecs_operations_detection_task_definition.json** – ECS Fargate task definition template for operations detection.
- **ecs_well_status_task_definition.json** – ECS Fargate task definition template for well status.
- **schedule_operations_detection.sh** – Register task + EventBridge schedule for operations detection (daily run).
- **schedule_well_status.sh** – Register task + EventBridge schedule for well status (daily run).
- **sagemaker/** – SageMaker training entrypoints; see `sagemaker/README.md`.
- **scripts/pack_and_upload_sagemaker_model.py** – Pack anomaly_detection → upload tarball to S3.
- **scripts/pack_and_upload_review_models.py** – Pack anomaly_review per-well models → upload tarball to S3.
- **scripts/pack_and_upload_operations_models.py** – Pack operations models → upload tarball to S3.
- **scripts/sync_models_to_s3.py** – Optional: sync local `models/` to S3 (per-module directories).
