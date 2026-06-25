#!/usr/bin/env bash
# Launch a SageMaker training job for operation recommendations (action models).
#
# Usage (from repo root):
#   ./operation_recommendations/deploy/run_sagemaker_training.sh
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-operations-train-$(date +%Y%m%d-%H%M)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.xlarge}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

_SAVED_ORGANIZATION_ID="${ORGANIZATION_ID:-}"
_SAVED_TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -n "$_SAVED_ORGANIZATION_ID" ]]; then
  export ORGANIZATION_ID="$_SAVED_ORGANIZATION_ID"
fi
if [[ -n "$_SAVED_TRAINING_JOB_NAME" ]]; then
  export TRAINING_JOB_NAME="$_SAVED_TRAINING_JOB_NAME"
fi

ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID}"

ECR_REPO_NAME="${ECR_REPO_TRAIN_OPERATIONS:-operations-sagemaker-train}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
SAGEMAKER_ROLE_NAME="${SAGEMAKER_ROLE_NAME:-anomaly-detection-sagemaker-role}"
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"

SENSOR_READINGS_S3="s3://${S3_BUCKET}/${ORGANIZATION_ID}/sensor_data/well_sensor_readings"
PRODUCTION_S3="s3://${S3_BUCKET}/${ORGANIZATION_ID}/sensor_data/daily_production"
MODEL_OUTPUT_S3="s3://${S3_BUCKET}/${ORGANIZATION_ID}/model"

echo "=== 1) ECR repository for operations training image ==="
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION"
echo "ECR_URI=$ECR_URI"

echo ""
echo "=== 2) Build and push operations training image ==="
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -f operation_recommendations/deploy/Dockerfile.sagemaker.train -t "$ECR_URI" --push .

echo ""
echo "=== 3) Ensure SageMaker role exists and has S3 write ==="
if ! aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" &>/dev/null; then
  echo "Creating role $SAGEMAKER_ROLE_NAME ..."
  cat > /tmp/sagemaker-trust.json << 'EOF'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
EOF
  aws iam create-role --role-name "$SAGEMAKER_ROLE_NAME" --assume-role-policy-document file:///tmp/sagemaker-trust.json
  aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
  aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  sleep 15
fi
aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true

echo ""
echo "=== 4) Create operations training job ==="
echo "  Job: $TRAINING_JOB_NAME"
echo "  Sensor: $SENSOR_READINGS_S3"
echo "  Production: $PRODUCTION_S3"
echo "  Output: $MODEL_OUTPUT_S3"

aws sagemaker create-training-job \
  --training-job-name "$TRAINING_JOB_NAME" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --algorithm-specification "TrainingImage=${ECR_URI},TrainingInputMode=File" \
  --input-data-config \
    "ChannelName=sensor_data,DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=${SENSOR_READINGS_S3}/,S3DataDistributionType=FullyReplicated}}" \
    "ChannelName=production_data,DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=${PRODUCTION_S3}/,S3DataDistributionType=FullyReplicated}}" \
  --output-data-config "S3OutputPath=${MODEL_OUTPUT_S3}/" \
  --resource-config "InstanceType=${INSTANCE_TYPE},InstanceCount=1,VolumeSizeInGB=30" \
  --stopping-condition "MaxRuntimeInSeconds=3600" \
  --hyper-parameters "ORGANIZATION_ID=${ORGANIZATION_ID}" \
  --region "$REGION"

echo ""
echo "Operations training job started: $TRAINING_JOB_NAME"
echo "Model will be at: ${MODEL_OUTPUT_S3}/${TRAINING_JOB_NAME}/output/model.tar.gz"
