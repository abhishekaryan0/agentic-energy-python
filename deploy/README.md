# Deploy: SageMaker + Lambda (AWS CLI)

Deployment uses **AWS CLI only**. No ECS.

**Models on S3 and detection:** Trained models are stored on S3 in their own directory; detection uses that S3 model. See [docs/MODELS_S3_AND_DETECTION.md](../docs/MODELS_S3_AND_DETECTION.md) for the flow for all three (anomaly detection, anomaly review, operations).

## Role of SageMaker and why you use it

**SageMaker** hosts your ML models as **HTTPS inference endpoints** in AWS. You use it so that:

- **Anomaly detection**, **anomaly review**, and **operations** recommendations can be called as APIs (e.g. from other services, Lambda, or the web app) without running the full Flask app.
- Models run on managed instances (e.g. `ml.m5.xlarge`), scale independently, and stay available 24/7.
- **Lambda** in front of each endpoint gives you a simple invoke interface (and can be wired to API Gateway for REST).

So: **Flask app** = full UI + in-process detection when you run it locally; **SageMaker + Lambda** = same models exposed as serverless APIs in AWS.

## How to check SageMaker is working

1. **Endpoint status** (all three endpoints):

   ```bash
   ./deploy/check_endpoints.sh
   # Or with region: ./deploy/check_endpoints.sh us-east-1
   ```

   You should see `InService` for any endpoint you have deployed. If you see `Creating` or `Updating`, wait 5–10 minutes. `Failed` or `not found` means the endpoint does not exist or needs to be recreated.

2. **Invoke the anomaly-detection Lambda** (after endpoint is InService):

   ```bash
   aws lambda invoke --function-name anomaly-detection-invoke \
     --cli-binary-format raw-in-base64-out \
     --payload '{"readings":{"motor_temp":100,"motor_current":40},"lift_type":"ESP"}' \
     --region us-east-1 /tmp/out.json && cat /tmp/out.json
   ```

   You should get JSON with `is_anomaly`, `anomaly_score`, etc.

3. **Invoke the endpoint directly** (optional):

   ```bash
   aws sagemaker-runtime invoke-endpoint \
     --endpoint-name anomaly-detection-endpoint \
     --content-type application/json \
     --body '{"readings":{"motor_temp":100},"lift_type":"ESP"}' \
     --region us-east-1 /tmp/out.json && cat /tmp/out.json
   ```

4. **Review and operations** (if deployed): use `anomaly-review-invoke` and `anomaly-operations-invoke` with the payloads described later in this README.

## One-shot deploy (from repo root)

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET=your-models-bucket

./deploy/deploy_sagemaker_lambda.sh
```

The script will:

1. **ECR** – Create repo (if missing), build `Dockerfile.sagemaker`, push image.
2. **S3** – Create bucket (if missing), pack `models/anomaly_detection/` and upload tarball.
3. **IAM** – Create SageMaker execution role (with ECR pull, S3 read) and Lambda role (InvokeEndpoint + logs).
4. **SageMaker** – Create model (image + model data URL), endpoint config, endpoint.
5. **Lambda** – Create/update function that invokes the endpoint; env `SAGEMAKER_ENDPOINT_NAME` set.

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
- The Flask app (`app.py`) also uses operations in-process for `/api/operation-suggestions`, `/api/operation-recommendations`, `/api/run-pipeline`.

## Files

- **Dockerfile** – Flask app (legacy/local).
- **Dockerfile.sagemaker** – SageMaker inference container (anomaly_detection).
- **Dockerfile.sagemaker.review** – SageMaker inference container (anomaly_review).
- **Dockerfile.sagemaker.operations** – SageMaker inference container (operations detector).
- **deploy_sagemaker_lambda.sh** – Deploy anomaly_detection endpoint + Lambda.
- **deploy_sagemaker_review.sh** – Deploy anomaly_review endpoint + Lambda.
- **deploy_sagemaker_operations.sh** – Deploy operations detector endpoint + Lambda.
- **sagemaker/** – Inference modules and Lambda examples; see `sagemaker/README.md`.
- **scripts/pack_and_upload_sagemaker_model.py** – Pack anomaly_detection → upload tarball to S3.
- **scripts/pack_and_upload_review_models.py** – Pack anomaly_review per-well models → upload tarball to S3.
- **scripts/pack_and_upload_operations_models.py** – Pack operations models → upload tarball to S3.
- **scripts/sync_models_to_s3.py** – Optional: sync local `models/` to S3 (per-module directories).
