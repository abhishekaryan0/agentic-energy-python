#!/usr/bin/env bash
#
# Build and deploy South Texas Energy Partners pipeline Lambda (synthetic data → Snowflake).
#
# Usage (from repo root):
#   ./SOUTH_TEXAS_ENERGY_PARTNERS/deploy/deploy.sh
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
ORGANIZATION_ID="${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}"
SNOWFLAKE_USER="${SNOWFLAKE_USER:?Set SNOWFLAKE_USER}"
SNOWFLAKE_ACCOUNT="${SNOWFLAKE_ACCOUNT:?Set SNOWFLAKE_ACCOUNT}"
SNOWFLAKE_WAREHOUSE="${SNOWFLAKE_WAREHOUSE:?Set SNOWFLAKE_WAREHOUSE}"
SNOWFLAKE_DATABASE="${SNOWFLAKE_DATABASE:-AT_RISK_ASSETS}"
SNOWFLAKE_SCHEMA="${STEP_SNOWFLAKE_SCHEMA:-SOUTH_TEXAS_SENSOR_DATA}"
SNOWFLAKE_ROLE="${SNOWFLAKE_ROLE:?Set SNOWFLAKE_ROLE}"
SNOWFLAKE_PASSWORD="${SNOWFLAKE_PASSWORD:-}"
SNOWFLAKE_TOKEN="${SNOWFLAKE_TOKEN:-}"

if [[ -z "$SNOWFLAKE_TOKEN" && -z "$SNOWFLAKE_PASSWORD" ]]; then
  echo "ERROR: Set SNOWFLAKE_TOKEN or SNOWFLAKE_PASSWORD in .env"
  exit 1
fi

LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-south-texas-energy-partners}"
ECR_REPO_NAME="${ECR_REPO_STEP:-south-texas-step-lambda}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-south-texas-step-lambda-role}"
LAMBDA_MEMORY="${LAMBDA_MEMORY:-8192}"
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-900}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${LAMBDA_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

echo "=== 1) ECR repository and image ==="
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$LAMBDA_REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$LAMBDA_REGION" >/dev/null
fi

if [[ "${SKIP_ECR_BUILD:-0}" != "1" ]]; then
  aws ecr get-login-password --region "$LAMBDA_REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${LAMBDA_REGION}.amazonaws.com"
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f SOUTH_TEXAS_ENERGY_PARTNERS/deploy/Dockerfile.lambda -t "$ECR_URI" --push .
else
  echo "Skipping ECR build (SKIP_ECR_BUILD=1)"
fi

echo ""
echo "=== 2) IAM role for Lambda ==="
if ! aws iam get-role --role-name "$LAMBDA_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$LAMBDA_ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' >/dev/null
  echo "Created role $LAMBDA_ROLE_NAME; waiting for propagation..."
  sleep 10
fi

aws iam attach-role-policy --role-name "$LAMBDA_ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true

echo ""
echo "=== 3) Lambda function ==="
ENV_JSON_FILE="$(mktemp)"
export ENV_JSON_FILE ORGANIZATION_ID LAMBDA_REGION LOG_LEVEL
export SNOWFLAKE_USER SNOWFLAKE_PASSWORD SNOWFLAKE_TOKEN SNOWFLAKE_ACCOUNT
export SNOWFLAKE_WAREHOUSE SNOWFLAKE_DATABASE SNOWFLAKE_SCHEMA SNOWFLAKE_ROLE
export SNOWFLAKE_NETWORK_POLICY="${SNOWFLAKE_NETWORK_POLICY:-}"
python3 - <<'PY'
import json
import os

variables = {
    "STEP_ORGANIZATION_ID": os.environ["ORGANIZATION_ID"],
    "ORGANIZATION_ID": os.environ["ORGANIZATION_ID"],
    "STEP_SNOWFLAKE_SCHEMA": os.environ.get("STEP_SNOWFLAKE_SCHEMA", "SOUTH_TEXAS_SENSOR_DATA"),
    "LOG_LEVEL": os.environ["LOG_LEVEL"],
    "SNOWFLAKE_SOCKET_TIMEOUT": os.environ.get("SNOWFLAKE_SOCKET_TIMEOUT", "600"),
    "SNOWFLAKE_USER": os.environ["SNOWFLAKE_USER"],
    "SNOWFLAKE_ACCOUNT": os.environ["SNOWFLAKE_ACCOUNT"],
    "SNOWFLAKE_WAREHOUSE": os.environ["SNOWFLAKE_WAREHOUSE"],
    "SNOWFLAKE_DATABASE": os.environ["SNOWFLAKE_DATABASE"],
    "SNOWFLAKE_SCHEMA": os.environ.get("STEP_SNOWFLAKE_SCHEMA", "SOUTH_TEXAS_SENSOR_DATA"),
    "SNOWFLAKE_ROLE": os.environ["SNOWFLAKE_ROLE"],
}
token = os.environ.get("SNOWFLAKE_TOKEN", "").strip()
password = os.environ.get("SNOWFLAKE_PASSWORD", "").strip()
use_password = os.environ.get("SNOWFLAKE_USE_PASSWORD", "").strip().lower() in (
    "1", "true", "yes",
)
if token and not use_password:
    variables["SNOWFLAKE_TOKEN"] = token
elif password:
    variables["SNOWFLAKE_PASSWORD"] = password
elif token:
    variables["SNOWFLAKE_TOKEN"] = token
for key in ("SNOWFLAKE_NETWORK_POLICY",):
    val = os.environ.get(key, "").strip()
    if val:
        variables[key] = val

with open(os.environ["ENV_JSON_FILE"], "w", encoding="utf-8") as fh:
    json.dump({"Variables": variables}, fh)
PY

if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$LAMBDA_REGION" &>/dev/null; then
  aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --image-uri "$ECR_URI" \
    --region "$LAMBDA_REGION" >/dev/null
  echo "Waiting for code update..."
  aws lambda wait function-updated --function-name "$LAMBDA_FUNCTION_NAME" --region "$LAMBDA_REGION"

  aws lambda update-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --role "$LAMBDA_ROLE_ARN" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY" \
    --environment "file://${ENV_JSON_FILE}" \
    --region "$LAMBDA_REGION" >/dev/null
  echo "Updated Lambda: $LAMBDA_FUNCTION_NAME"
else
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --package-type Image \
    --code "ImageUri=${ECR_URI}" \
    --role "$LAMBDA_ROLE_ARN" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY" \
    --environment "file://${ENV_JSON_FILE}" \
    --region "$LAMBDA_REGION" >/dev/null
  echo "Created Lambda: $LAMBDA_FUNCTION_NAME"
fi
rm -f "$ENV_JSON_FILE"

echo ""
echo "Done."
echo "Function : $LAMBDA_FUNCTION_NAME ($LAMBDA_REGION)"
echo "Image    : $ECR_URI"
echo "Target   : Snowflake ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA}"
echo ""
echo "Test invoke:"
echo "  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $LAMBDA_REGION \\"
echo "    --cli-binary-format raw-in-base64-out --payload '{\"from_master\": true}' /tmp/step_out.json"
