#!/usr/bin/env bash
#
# Build and deploy well status Lambda (S3 sensor parquet -> PostgreSQL).
#
# Usage (from repo root):
#   export S3_BUCKET=...
#   export ORGANIZATION_ID=...
#   export DATABASE_URL=...   # or DATABASE_URL_SECRET_ARN
#   ./well_status/deploy/deploy.sh
#
# Apex Permian (separate Lambda — do not overwrite Red Bluff well-status):
#   ./well_status/deploy/schedule_apex.sh
#
# South Texas Energy Partners (separate Lambda — uses STEP_ORGANIZATION_ID from .env):
#   ./well_status/deploy/schedule_step.sh
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

_SAVED_ORGANIZATION_ID="${ORGANIZATION_ID:-}"
_SAVED_LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-}"
_SAVED_LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-}"
_SAVED_RULE_NAME="${RULE_NAME:-}"
_SAVED_WELL_STATUS_FORCE_FULL="${WELL_STATUS_FORCE_FULL:-}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -n "$_SAVED_ORGANIZATION_ID" ]]; then
  export ORGANIZATION_ID="$_SAVED_ORGANIZATION_ID"
fi
if [[ -n "$_SAVED_LAMBDA_FUNCTION_NAME" ]]; then
  export LAMBDA_FUNCTION_NAME="$_SAVED_LAMBDA_FUNCTION_NAME"
fi
if [[ -n "$_SAVED_LAMBDA_ROLE_NAME" ]]; then
  export LAMBDA_ROLE_NAME="$_SAVED_LAMBDA_ROLE_NAME"
fi
if [[ -n "$_SAVED_WELL_STATUS_FORCE_FULL" ]]; then
  export WELL_STATUS_FORCE_FULL="$_SAVED_WELL_STATUS_FORCE_FULL"
fi

LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
S3_BUCKET="${S3_BUCKET// /}"
ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-well-status}"
ECR_REPO_NAME="${ECR_REPO_WELL_STATUS:-well-status-lambda}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-well-status-lambda-role}"
LAMBDA_MEMORY="${LAMBDA_MEMORY:-4096}"
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-900}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LAMBDA_VPC_ENABLED="${LAMBDA_VPC_ENABLED:-0}"

if [[ -z "${DATABASE_URL:-}" && -z "${DATABASE_URL_SECRET_ARN:-}" ]]; then
  echo "ERROR: Set DATABASE_URL or DATABASE_URL_SECRET_ARN"
  exit 1
fi

if [[ "$LAMBDA_VPC_ENABLED" == "1" ]]; then
  if [[ -z "${LAMBDA_SUBNETS:-}" || -z "${LAMBDA_SECURITY_GROUPS:-}" ]]; then
    echo "ERROR: LAMBDA_VPC_ENABLED=1 requires LAMBDA_SUBNETS and LAMBDA_SECURITY_GROUPS."
    exit 1
  fi
fi

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
    -f well_status/deploy/Dockerfile.lambda -t "$ECR_URI" --push .
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
aws iam attach-role-policy --role-name "$LAMBDA_ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole 2>/dev/null || true

POLICY_DOC="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::${S3_BUCKET}",
        "arn:aws:s3:::${S3_BUCKET}/${ORGANIZATION_ID}/*"
      ]
    }
  ]
}
EOF
)"
if [[ -n "${DATABASE_URL_SECRET_ARN:-}" ]]; then
  POLICY_DOC="$(python3 - <<PY
import json, os
doc = json.loads('''$POLICY_DOC''')
doc["Statement"].append({
    "Effect": "Allow",
    "Action": ["secretsmanager:GetSecretValue"],
    "Resource": [os.environ["DATABASE_URL_SECRET_ARN"]],
})
print(json.dumps(doc))
PY
)"
fi
aws iam put-role-policy --role-name "$LAMBDA_ROLE_NAME" \
  --policy-name "WellStatusAccess" --policy-document "$POLICY_DOC"

echo ""
echo "=== 3) Lambda function ==="
ENV_JSON_FILE="$(mktemp)"
export ENV_JSON_FILE S3_BUCKET ORGANIZATION_ID LAMBDA_REGION LOG_LEVEL
export DATABASE_URL_SECRET_ARN="${DATABASE_URL_SECRET_ARN:-}"
export DATABASE_URL="${DATABASE_URL:-}"
python3 - <<'PY'
import json
import os

variables = {
    "S3_BUCKET": os.environ["S3_BUCKET"],
    "ORGANIZATION_ID": os.environ["ORGANIZATION_ID"],
    "LOG_LEVEL": os.environ["LOG_LEVEL"],
    "WELL_STATUS_PARTITION_START_DATE": os.environ.get("WELL_STATUS_PARTITION_START_DATE", "2026-01-01"),
    "WELL_STATUS_FORCE_FULL": os.environ.get("WELL_STATUS_FORCE_FULL", "0"),
}
secret_arn = os.environ.get("DATABASE_URL_SECRET_ARN", "").strip()
database_url = os.environ.get("DATABASE_URL", "").strip()
if secret_arn:
    variables["DATABASE_URL_SECRET_ARN"] = secret_arn
elif database_url:
    variables["DATABASE_URL"] = database_url
else:
    raise SystemExit("DATABASE_URL or DATABASE_URL_SECRET_ARN is required")

with open(os.environ["ENV_JSON_FILE"], "w", encoding="utf-8") as fh:
    json.dump({"Variables": variables}, fh)
PY

VPC_ARGS=()
if [[ "$LAMBDA_VPC_ENABLED" == "1" ]]; then
  SUBNET_JSON="[\"$(echo "$LAMBDA_SUBNETS" | sed 's/,/","/g')\"]"
  SG_JSON="[\"$(echo "$LAMBDA_SECURITY_GROUPS" | sed 's/,/","/g')\"]"
  VPC_ARGS=(--vpc-config "SubnetIds=${SUBNET_JSON},SecurityGroupIds=${SG_JSON}")
fi

if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$LAMBDA_REGION" &>/dev/null; then
  aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --image-uri "$ECR_URI" \
    --region "$LAMBDA_REGION" >/dev/null
  echo "Waiting for code update..."
  aws lambda wait function-updated --function-name "$LAMBDA_FUNCTION_NAME" --region "$LAMBDA_REGION"

  UPDATE_ARGS=(
    --function-name "$LAMBDA_FUNCTION_NAME"
    --role "$LAMBDA_ROLE_ARN"
    --timeout "$LAMBDA_TIMEOUT"
    --memory-size "$LAMBDA_MEMORY"
    --environment "file://${ENV_JSON_FILE}"
    --region "$LAMBDA_REGION"
  )
  if [[ "$LAMBDA_VPC_ENABLED" == "1" ]]; then
    UPDATE_ARGS+=("${VPC_ARGS[@]}")
  else
    UPDATE_ARGS+=(--vpc-config "SubnetIds=[],SecurityGroupIds=[]")
  fi
  aws lambda update-function-configuration "${UPDATE_ARGS[@]}" >/dev/null
  echo "Updated Lambda: $LAMBDA_FUNCTION_NAME"
else
  CREATE_ARGS=(
    --function-name "$LAMBDA_FUNCTION_NAME"
    --package-type Image
    --code "ImageUri=${ECR_URI}"
    --role "$LAMBDA_ROLE_ARN"
    --timeout "$LAMBDA_TIMEOUT"
    --memory-size "$LAMBDA_MEMORY"
    --environment "file://${ENV_JSON_FILE}"
    --region "$LAMBDA_REGION"
  )
  if [[ "$LAMBDA_VPC_ENABLED" == "1" ]]; then
    CREATE_ARGS+=("${VPC_ARGS[@]}")
  fi
  aws lambda create-function "${CREATE_ARGS[@]}" >/dev/null
  echo "Created Lambda: $LAMBDA_FUNCTION_NAME"
fi
rm -f "$ENV_JSON_FILE"

echo ""
echo "Done."
echo "Function : $LAMBDA_FUNCTION_NAME ($LAMBDA_REGION)"
echo "Image    : $ECR_URI"
echo "Target   : PostgreSQL schema \"${ORGANIZATION_ID}\".well_status"
echo ""
echo "Test invoke:"
echo "  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $LAMBDA_REGION \\"
echo "    --cli-binary-format raw-in-base64-out --payload '{}' /tmp/well_status_out.json"
