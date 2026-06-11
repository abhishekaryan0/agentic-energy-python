#!/usr/bin/env bash
# Launch a SageMaker training job for financial model (implementation cost).
#
# Usage (from repo root):
#   ./financial/deploy/run_sagemaker_training.sh
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID}"
TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-financial-train-$(date +%Y%m%d-%H%M)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.xlarge}"
FINANCIAL_TRAINING_PARTITION_DAYS="${FINANCIAL_TRAINING_PARTITION_DAYS:-365}"

ECR_REPO_NAME="${ECR_REPO_TRAIN_FINANCIAL:-financial-sagemaker-train}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

SAGEMAKER_ROLE_NAME="${SAGEMAKER_ROLE_NAME:-anomaly-detection-sagemaker-role}"
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PRODUCTION_S3="s3://${S3_BUCKET}/${ORGANIZATION_ID}/sensor_data/daily_production"
FINANCIAL_DATA_S3="s3://${S3_BUCKET}/${ORGANIZATION_ID}/financial_data"
MODEL_OUTPUT_S3="s3://${S3_BUCKET}/${ORGANIZATION_ID}/model"

echo "=== 1) ECR repository for financial training image ==="
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION"
echo "ECR_URI=$ECR_URI"

echo ""
echo "=== 2) Build and push financial training image ==="
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -f financial/deploy/Dockerfile.sagemaker.train -t "$ECR_URI" --push .

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
echo "=== 4) Create financial training job ==="
echo "  Job: $TRAINING_JOB_NAME"
echo "  Production: $PRODUCTION_S3"
echo "  Financial input: $FINANCIAL_DATA_S3"
echo "  Output: $MODEL_OUTPUT_S3"
echo "  Partition days: $FINANCIAL_TRAINING_PARTITION_DAYS"

aws sagemaker create-training-job \
  --training-job-name "$TRAINING_JOB_NAME" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --algorithm-specification "TrainingImage=${ECR_URI},TrainingInputMode=File" \
  --input-data-config \
    "ChannelName=sensor_data,DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=${PRODUCTION_S3}/,S3DataDistributionType=FullyReplicated}}" \
    "ChannelName=financial_data,DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=${FINANCIAL_DATA_S3}/,S3DataDistributionType=FullyReplicated}}" \
  --output-data-config "S3OutputPath=${MODEL_OUTPUT_S3}/" \
  --resource-config "InstanceType=${INSTANCE_TYPE},InstanceCount=1,VolumeSizeInGB=30" \
  --stopping-condition "MaxRuntimeInSeconds=3600" \
  --hyper-parameters "ORGANIZATION_ID=${ORGANIZATION_ID},FINANCIAL_TRAINING_PARTITION_DAYS=${FINANCIAL_TRAINING_PARTITION_DAYS}" \
  --region "$REGION"

echo ""
echo "Financial training job started: $TRAINING_JOB_NAME"
echo "Model will be at: ${MODEL_OUTPUT_S3}/${TRAINING_JOB_NAME}/output/model.tar.gz"
