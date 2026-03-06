#!/usr/bin/env bash
# Deploy operational recommendations detector to SageMaker + Lambda.
# Run from repo root. Requires: aws cli, docker.
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
#   export S3_BUCKET=my-models-bucket
#   ./deploy/deploy_sagemaker_operations.sh
#
# Detector is stateless: input = well_id, lift_type, metrics (no Snowflake in container).
# Train models first: python -m operations.train_models train_category && python -m operations.train_models train_action
#
# Optional: ECR_REPO_NAME_OPERATIONS, SKIP_ECR_BUILD=1, OPERATIONS_MODEL_PREFIX, OPERATIONS_MODEL_VERSION

set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
OPERATIONS_MODEL_PREFIX="${OPERATIONS_MODEL_PREFIX:-models/operations}"
OPERATIONS_MODEL_VERSION="${OPERATIONS_MODEL_VERSION:-v1}"

ECR_REPO_NAME="${ECR_REPO_NAME_OPERATIONS:-anomaly-operations-sagemaker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SKIP_ECR_BUILD="${SKIP_ECR_BUILD:-0}"
SAGEMAKER_MODEL_NAME="${SAGEMAKER_MODEL_NAME_OPERATIONS:-anomaly-operations-model}"
SAGEMAKER_ENDPOINT_CONFIG_NAME="${SAGEMAKER_ENDPOINT_CONFIG_NAME_OPERATIONS:-anomaly-operations-config}"
SAGEMAKER_ENDPOINT_NAME="${SAGEMAKER_ENDPOINT_NAME_OPERATIONS:-anomaly-operations-endpoint}"
SAGEMAKER_INSTANCE_TYPE="${SAGEMAKER_INSTANCE_TYPE_OPERATIONS:-ml.m5.large}"
SAGEMAKER_INSTANCE_COUNT="${SAGEMAKER_INSTANCE_COUNT_OPERATIONS:-1}"

LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME_OPERATIONS:-anomaly-operations-invoke}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME_OPERATIONS:-anomaly-operations-lambda-role}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1) ECR repository (operations) ==="
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
echo "ECR_URI=$ECR_URI"

echo ""
echo "=== 2) Build and push operations image ==="
if [[ "$SKIP_ECR_BUILD" == "1" ]]; then
  echo "Skipping (SKIP_ECR_BUILD=1). Using existing image $ECR_URI"
else
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f deploy/Dockerfile.sagemaker.operations -t "$ECR_URI" --push .
fi

echo ""
echo "=== 3) S3 bucket and operations model tarball ==="
aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null || aws s3 mb "s3://${S3_BUCKET}" --region "$REGION"
python3 scripts/pack_and_upload_operations_models.py --bucket "$S3_BUCKET" --prefix "$OPERATIONS_MODEL_PREFIX" --version "$OPERATIONS_MODEL_VERSION" || true
MODEL_DATA_URL="s3://${S3_BUCKET}/${OPERATIONS_MODEL_PREFIX}/operations_models_${OPERATIONS_MODEL_VERSION}.tar.gz"
echo "Model data URL: $MODEL_DATA_URL"

echo ""
echo "=== 4) SageMaker execution role ==="
SAGEMAKER_ROLE_NAME="${SAGEMAKER_ROLE_NAME:-anomaly-detection-sagemaker-role}"
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
if ! aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" &>/dev/null; then
  echo "Create the SageMaker role first (e.g. run deploy_sagemaker_lambda.sh once)."
  exit 1
fi
echo "SageMaker role: $SAGEMAKER_ROLE_ARN"

echo ""
echo "=== 5) SageMaker model (operations) ==="
aws sagemaker describe-model --model-name "$SAGEMAKER_MODEL_NAME" --region "$REGION" 2>/dev/null && \
  aws sagemaker delete-model --model-name "$SAGEMAKER_MODEL_NAME" --region "$REGION" || true
sleep 2
aws sagemaker create-model \
  --model-name "$SAGEMAKER_MODEL_NAME" \
  --execution-role-arn "$SAGEMAKER_ROLE_ARN" \
  --primary-container "{\"Image\":\"$ECR_URI\",\"ModelDataUrl\":\"$MODEL_DATA_URL\",\"Environment\":{\"SAGEMAKER_PROGRAM\":\"deploy.sagemaker.operations_inference\",\"SAGEMAKER_REGION\":\"$REGION\"}}" \
  --region "$REGION"

echo ""
echo "=== 6) Endpoint config and endpoint ==="
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
  echo "Endpoint $SAGEMAKER_ENDPOINT_NAME is Failed; deleting..."
  aws sagemaker delete-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$REGION"
  while aws sagemaker describe-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$REGION" &>/dev/null; do sleep 15; done
  aws sagemaker create-endpoint \
    --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" \
    --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" \
    --region "$REGION"
elif [[ -n "$ENDPOINT_STATUS" ]]; then
  aws sagemaker update-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" --region "$REGION"
else
  aws sagemaker create-endpoint \
    --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" \
    --endpoint-config-name "$SAGEMAKER_ENDPOINT_CONFIG_NAME" \
    --region "$REGION"
fi
echo "Endpoint $SAGEMAKER_ENDPOINT_NAME is creating/updating. Wait 5–10 min."

echo ""
echo "=== 7) Lambda (optional) ==="
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"
if ! aws iam get-role --role-name "$LAMBDA_ROLE_NAME" &>/dev/null; then
  cat > /tmp/lambda-ops-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{ "Effect": "Allow", "Principal": { "Service": "lambda.amazonaws.com" }, "Action": "sts:AssumeRole" }]
}
EOF
  aws iam create-role --role-name "$LAMBDA_ROLE_NAME" --assume-role-policy-document file:///tmp/lambda-ops-trust.json
  cat > /tmp/lambda-ops-policy.json << LAMBDAPOL
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint", "Resource": "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:endpoint/${SAGEMAKER_ENDPOINT_NAME}" },
    { "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*" }
  ]
}
LAMBDAPOL
  aws iam put-role-policy --role-name "$LAMBDA_ROLE_NAME" --policy-name "InvokeAndLogs" --policy-document file:///tmp/lambda-ops-policy.json
  sleep 15
fi
LAMBDA_DIR="/tmp/lambda_operations_deploy"
rm -rf "$LAMBDA_DIR" && mkdir -p "$LAMBDA_DIR"
cat > "$LAMBDA_DIR/lambda_function.py" << 'LAMBDACODE'
import json
import os
import boto3

def lambda_handler(event, context):
    endpoint = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint:
        return {"statusCode": 500, "body": json.dumps({"error": "SAGEMAKER_ENDPOINT_NAME not set"})}
    body = event.get("body", event) if isinstance(event.get("body"), str) else (event.get("body") or event)
    if isinstance(body, dict) and (not body.get("well_id") or not body.get("lift_type") or not body.get("metrics")):
        return {"statusCode": 400, "body": json.dumps({"error": "well_id, lift_type, and metrics required"})}
    payload = json.dumps(body) if isinstance(body, dict) else body
    runtime = boto3.client("sagemaker-runtime")
    resp = runtime.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=payload)
    result = json.loads(resp["Body"].read().decode())
    return {"statusCode": 200, "body": json.dumps(result)}
LAMBDACODE
(cd "$LAMBDA_DIR" && zip -q -r /tmp/lambda_operations.zip .)
if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$REGION" &>/dev/null; then
  aws lambda update-function-code --function-name "$LAMBDA_FUNCTION_NAME" --zip-file fileb:///tmp/lambda_operations.zip --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --timeout 30 --memory-size 256 \
    --environment "Variables={SAGEMAKER_ENDPOINT_NAME=$SAGEMAKER_ENDPOINT_NAME}" \
    --region "$REGION"
  echo "Lambda $LAMBDA_FUNCTION_NAME updated."
else
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --runtime python3.11 --role "$LAMBDA_ROLE_ARN" --handler lambda_function.lambda_handler \
    --zip-file fileb:///tmp/lambda_operations.zip \
    --timeout 30 --memory-size 256 \
    --environment "Variables={SAGEMAKER_ENDPOINT_NAME=$SAGEMAKER_ENDPOINT_NAME}" \
    --region "$REGION"
  echo "Lambda $LAMBDA_FUNCTION_NAME created."
fi

echo ""
echo "=== Done ==="
echo "Operations endpoint: $SAGEMAKER_ENDPOINT_NAME"
echo "Input: well_id, lift_type, metrics (e.g. motor_temp_slope, motor_temp_roll_mean, oil_volume_slope, ...)"
echo "Invoke: aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --cli-binary-format raw-in-base64-out --payload '{\"well_id\":\"Well_001\",\"lift_type\":\"ESP\",\"metrics\":{\"motor_temp_slope\":0.1,\"motor_temp_roll_mean\":120}}' --region $REGION /tmp/out.json && cat /tmp/out.json"
