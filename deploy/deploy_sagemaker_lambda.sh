#!/usr/bin/env bash
# Deploy anomaly detection: ECR, S3, IAM, SageMaker model/endpoint, Lambda (all via AWS CLI).
# Run from repo root. Requires: aws cli, docker, jq.
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
#   export S3_BUCKET=my-models-bucket
#   ./deploy/deploy_sagemaker_lambda.sh
#
# Optional overrides:
#   ECR_REPO_NAME, SAGEMAKER_MODEL_NAME, SAGEMAKER_ENDPOINT_NAME, LAMBDA_FUNCTION_NAME, LAMBDA_ROLE_NAME
#   SKIP_ECR_BUILD=1  skip Docker build and push (use existing image in ECR, e.g. from CI)

set -euo pipefail
# Prevent AWS CLI from opening a pager (script would appear to "stop" at first JSON output)
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET (e.g. my-models-bucket)}"
MODEL_PREFIX="${MODEL_PREFIX:-models/anomaly}"
MODEL_VERSION="${MODEL_VERSION:-v1}"

ECR_REPO_NAME="${ECR_REPO_NAME:-anomaly-detection-sagemaker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SKIP_ECR_BUILD="${SKIP_ECR_BUILD:-0}"
SAGEMAKER_MODEL_NAME="${SAGEMAKER_MODEL_NAME:-anomaly-detection-model}"
SAGEMAKER_ENDPOINT_CONFIG_NAME="${SAGEMAKER_ENDPOINT_CONFIG_NAME:-anomaly-detection-config}"
SAGEMAKER_ENDPOINT_NAME="${SAGEMAKER_ENDPOINT_NAME:-anomaly-detection-endpoint}"
SAGEMAKER_INSTANCE_TYPE="${SAGEMAKER_INSTANCE_TYPE:-ml.m5.xlarge}"
SAGEMAKER_INSTANCE_COUNT="${SAGEMAKER_INSTANCE_COUNT:-1}"

LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-anomaly-detection-invoke}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-anomaly-detection-lambda-role}"
LAMBDA_HANDLER="${LAMBDA_HANDLER:-lambda_function.lambda_handler}"
LAMBDA_RUNTIME="${LAMBDA_RUNTIME:-python3.11}"
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-60}"
LAMBDA_MEMORY="${LAMBDA_MEMORY:-256}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1) ECR repository ==="
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
echo "ECR_URI=$ECR_URI"

echo ""
echo "=== 2) Build and push SageMaker inference image ==="
if [[ "$SKIP_ECR_BUILD" == "1" ]]; then
  echo "Skipping (SKIP_ECR_BUILD=1). Using existing image $ECR_URI"
else
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  # Single-platform build so ECR gets Docker V2 manifest (SageMaker rejects OCI index)
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f deploy/Dockerfile.sagemaker -t "$ECR_URI" --push .
fi

echo ""
echo "=== 3) S3 bucket and model tarball ==="
aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null || aws s3 mb "s3://${S3_BUCKET}" --region "$REGION"
if [[ ! -d "models/anomaly_detection" ]]; then
  echo "WARN: models/anomaly_detection not found. Run: python -m anomaly_detection.train_models"
  echo "Creating a minimal tarball for testing..."
  mkdir -p models/anomaly_detection/wells models/anomaly_detection/well_types
  touch models/anomaly_detection/wells/.keep models/anomaly_detection/well_types/.keep
fi
python3 scripts/pack_and_upload_sagemaker_model.py --bucket "$S3_BUCKET" --prefix "$MODEL_PREFIX" --version "$MODEL_VERSION" || true
MODEL_DATA_URL="s3://${S3_BUCKET}/${MODEL_PREFIX}/anomaly_detection_${MODEL_VERSION}.tar.gz"
echo "Model data URL: $MODEL_DATA_URL"

echo ""
echo "=== 4) SageMaker execution role ==="
SAGEMAKER_ROLE_NAME="${SAGEMAKER_ROLE_NAME:-anomaly-detection-sagemaker-role}"
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
if ! aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" &>/dev/null; then
  cat > /tmp/sagemaker-trust.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF
  aws iam create-role --role-name "$SAGEMAKER_ROLE_NAME" --assume-role-policy-document file:///tmp/sagemaker-trust.json
  aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
  aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  echo "Waiting for IAM role to propagate..."
  sleep 15
fi
# Ensure SageMaker can pull our ECR image (idempotent)
cat > /tmp/sagemaker-ecr-policy.json << ECRPOL
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"],
    "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/${ECR_REPO_NAME}"
  }]
}
ECRPOL
aws iam put-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-name "ECRPull" --policy-document file:///tmp/sagemaker-ecr-policy.json
echo "SageMaker role: $SAGEMAKER_ROLE_ARN"

echo ""
echo "=== 5) SageMaker model ==="
aws sagemaker describe-model --model-name "$SAGEMAKER_MODEL_NAME" --region "$REGION" 2>/dev/null && \
  aws sagemaker delete-model --model-name "$SAGEMAKER_MODEL_NAME" --region "$REGION" || true
sleep 2
cat > /tmp/sagemaker-primary-container.json << EOF
{
  "Image": "$ECR_URI",
  "ModelDataUrl": "$MODEL_DATA_URL",
  "Environment": {
    "SAGEMAKER_PROGRAM": "deploy.sagemaker.inference",
    "SAGEMAKER_REGION": "$REGION"
  }
}
EOF
aws sagemaker create-model \
  --model-name "$SAGEMAKER_MODEL_NAME" \
  --execution-role-arn "$SAGEMAKER_ROLE_ARN" \
  --primary-container file:///tmp/sagemaker-primary-container.json \
  --region "$REGION"

echo ""
echo "=== 6) SageMaker endpoint config and endpoint ==="
aws sagemaker describe-endpoint-config --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" --region "$REGION" 2>/dev/null && \
  aws sagemaker delete-endpoint-config --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" --region "$REGION" || true
sleep 2
aws sagemaker create-endpoint-config \
  --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" \
  --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"$SAGEMAKER_MODEL_NAME\",\"InstanceType\":\"$SAGEMAKER_INSTANCE_TYPE\",\"InitialInstanceCount\":$SAGEMAKER_INSTANCE_COUNT}]" \
  --region "$REGION"

ENDPOINT_STATUS=""
if aws sagemaker describe-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$REGION" &>/dev/null; then
  ENDPOINT_STATUS=$(aws sagemaker describe-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$REGION" --query 'EndpointStatus' --output text 2>/dev/null || echo "")
fi
if [[ "$ENDPOINT_STATUS" == "Failed" ]]; then
  echo "Endpoint $SAGEMAKER_ENDPOINT_NAME is Failed; deleting it (cannot update a failed endpoint)..."
  aws sagemaker delete-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$REGION"
  echo "Waiting for endpoint to be deleted..."
  while aws sagemaker describe-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$REGION" &>/dev/null; do sleep 15; done
  echo "Creating new endpoint $SAGEMAKER_ENDPOINT_NAME..."
  aws sagemaker create-endpoint \
    --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" \
    --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" \
    --region "$REGION"
elif [[ -n "$ENDPOINT_STATUS" ]]; then
  echo "Updating existing endpoint $SAGEMAKER_ENDPOINT_NAME..."
  aws sagemaker update-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" --region "$REGION"
else
  aws sagemaker create-endpoint \
    --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" \
    --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" \
    --region "$REGION"
fi
echo "Endpoint $SAGEMAKER_ENDPOINT_NAME is creating/updating. Wait 5–10 min, then check: aws sagemaker describe-endpoint --endpoint-name $SAGEMAKER_ENDPOINT_NAME --region $REGION"

echo ""
echo "=== 7) Lambda execution role ==="
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"
if ! aws iam get-role --role-name "$LAMBDA_ROLE_NAME" &>/dev/null; then
  cat > /tmp/lambda-trust.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "lambda.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF
  aws iam create-role --role-name "$LAMBDA_ROLE_NAME" --assume-role-policy-document file:///tmp/lambda-trust.json
  # Inline policy: InvokeEndpoint + basic Lambda logs
  cat > /tmp/lambda-inline-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:InvokeEndpoint",
      "Resource": "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:endpoint/${SAGEMAKER_ENDPOINT_NAME}"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
    }
  ]
}
EOF
  aws iam put-role-policy --role-name "$LAMBDA_ROLE_NAME" --policy-name "InvokeEndpointAndLogs" --policy-document file:///tmp/lambda-inline-policy.json
  echo "Waiting for IAM role to propagate..."
  sleep 15
fi
echo "Lambda role: $LAMBDA_ROLE_ARN"

echo ""
echo "=== 8) Lambda deployment package and function ==="
LAMBDA_DIR="/tmp/lambda_anomaly_deploy"
rm -rf "$LAMBDA_DIR" && mkdir -p "$LAMBDA_DIR"
cp deploy/sagemaker/lambda_invoke_example.py "$LAMBDA_DIR/lambda_function.py"
(cd "$LAMBDA_DIR" && zip -q -r /tmp/lambda_deploy.zip .)
if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$REGION" &>/dev/null; then
  aws lambda update-function-code --function-name "$LAMBDA_FUNCTION_NAME" --zip-file fileb:///tmp/lambda_deploy.zip --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY" \
    --environment "Variables={SAGEMAKER_ENDPOINT_NAME=$SAGEMAKER_ENDPOINT_NAME}" \
    --region "$REGION"
  echo "Lambda function $LAMBDA_FUNCTION_NAME updated."
else
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --runtime "$LAMBDA_RUNTIME" \
    --role "$LAMBDA_ROLE_ARN" \
    --handler "$LAMBDA_HANDLER" \
    --zip-file fileb:///tmp/lambda_deploy.zip \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY" \
    --environment "Variables={SAGEMAKER_ENDPOINT_NAME=$SAGEMAKER_ENDPOINT_NAME}" \
    --region "$REGION"
  echo "Lambda function $LAMBDA_FUNCTION_NAME created."
fi

echo ""
echo "=== Done ==="
echo "SageMaker endpoint: $SAGEMAKER_ENDPOINT_NAME (wait for InService before invoking)"
echo "Lambda function:   $LAMBDA_FUNCTION_NAME"
echo "Invoke Lambda:     aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --cli-binary-format raw-in-base64-out --payload '{\"readings\":{\"motor_temp\":100},\"lift_type\":\"ESP\"}' --region $REGION /tmp/out.json && cat /tmp/out.json"
