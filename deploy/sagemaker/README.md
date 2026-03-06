# SageMaker deployment (Option B: SageMaker endpoint + Lambda)

This folder contains the inference entrypoint used when the anomaly detection model is deployed as a **SageMaker endpoint** (with Lambda in front).

## Files

- **inference.py** – SageMaker contract for **anomaly_detection**: `model_fn`, `input_fn`, `predict_fn`, `output_fn`. Loads `AnomalyDetector` from `model_dir` and runs `check_anomaly()` on each request.
- **review_inference.py** – SageMaker contract for **anomaly_review**: same contract; sets `MODEL_DIR` to `model_dir` and runs `detect_anomalies(well_id, save_to_db=...)` per request. Use with `Dockerfile.sagemaker.review` and `deploy_sagemaker_review.sh`.

## Model tarball layout

SageMaker extracts the model tarball to `model_dir` (e.g. `/opt/ml/model`). The tarball must contain **anomaly_detection/** at the top level:

```
model_dir/
  anomaly_detection/
    wells/
      <well_id>/
        isolation_forest.joblib
        statistical_model.joblib
        lstm_autoencoder.pt
        lstm_scaler.joblib
        lstm_metadata.joblib
    well_types/   (optional)
      rod_pump/
      esp/
      gas_lift/
```

Pack from project root after training:

```bash
# From repo root
cd models
tar -czvf anomaly_detection.tar.gz anomaly_detection/
# Upload to S3
aws s3 cp anomaly_detection.tar.gz s3://YOUR_BUCKET/YOUR_PREFIX/anomaly_detection.tar.gz
```

Or use the script:

```bash
python scripts/pack_and_upload_sagemaker_model.py --bucket YOUR_BUCKET --prefix models --version v1
```

## Request / response (for Lambda)

**Request body (JSON):**

```json
{
  "readings": { "motor_temp": 250, "motor_current": 45, ... },
  "lift_type": "ESP",
  "well_id": "well-001",
  "historical_data": []
}
```

**Response (JSON):** same as your current API: `is_anomaly`, `anomaly_score`, `rule_score`, `ml_score`, `lstm_score`, `violations`, `summary`, `detection_method`.

## Using your Docker image with SageMaker

Your app Dockerfile (`deploy/Dockerfile`) copies the app to `/app`. To use it as a **SageMaker endpoint** you need a server that:

1. Loads the model once (calls `model_fn(model_dir)`).
2. For each POST to `/invocations`, runs `input_fn` → `predict_fn` → `output_fn` and returns the result.

**Option A – SageMaker PyTorch container**

Use a SageMaker PyTorch pre-built image, add your code and this module, and set the inference script to `deploy.sagemaker.inference` (or the path SageMaker expects). Then create the SageMaker Model with that image and the model tarball URL.

**Option B – Custom image with SageMaker Inference Toolkit**

Install `sagemaker-inference` in your Dockerfile and set:

- `SAGEMAKER_PROGRAM` or the entrypoint so the toolkit loads `deploy.sagemaker.inference`.
- Ensure `PYTHONPATH` or `sys.path` includes `/app` so `from anomaly_detection.detector import ...` works.

**Option C – Minimal custom server**

Run a small Flask/Gunicorn app that:

- On startup: reads `SAGEMAKER_MODEL_DIR` or `/opt/ml/model`, calls `model_fn(model_dir)`, keeps the detector in memory.
- On `POST /invocations`: reads body, calls `input_fn` → `predict_fn` → `output_fn`, returns JSON.

When **SAGEMAKER_REGION** is set (as in the deploy script), the detector runs **ML/LSTM only** (no Snowflake). For rule-based detection outside SageMaker, configure **SNOWFLAKE_***.

## Environment variables (anomaly_detection)

- **ANOMALY_MODELS_DIR** – Set by inference.py from `model_dir` so the detector loads from `/opt/ml/model`.
- **SAGEMAKER_REGION** – Set by deploy script; detector then skips Snowflake and uses ML/LSTM only.
- **SNOWFLAKE_*** – Optional in SageMaker; required for rule-based detection when not in SageMaker.
