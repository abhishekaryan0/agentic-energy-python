#!/usr/bin/env bash
#
# Schedule weekly SageMaker training for anomaly_detection only.
# EventBridge -> ECS Fargate launcher -> CreateTrainingJob (one model).
#
# Prerequisite: training image in ECR (run ./anomaly_detection/deploy/run_sagemaker_training.sh once).
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${ECS_CLUSTER:-anomaly-detection-sagemaker-train-cluster}"
TASK_FAMILY="anomaly-detection-sagemaker-train-launch"
LOG_GROUP="/ecs/anomaly-detection-sagemaker-train"
RULE_NAME="${RULE_NAME:-anomaly-detection-sagemaker-train-weekly}"
EVENTBRIDGE_ROLE_NAME="ecs-anomaly-detection-sagemaker-train-events-role"
EXECUTION_ROLE_NAME="ecs-anomaly-detection-sagemaker-train-execution-role"
TASK_ROLE_NAME="ecs-anomaly-detection-sagemaker-train-task-role"
ECR_REPO_LAUNCHER="${ECR_REPO_SAGEMAKER_TRAIN_LAUNCHER:-anomaly-detection-sagemaker-train-launcher}"
ECR_REPO_TRAIN="${ECR_REPO_TRAIN:-anomaly-detection-sagemaker-train}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 2 ? * MON *)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.xlarge}"
PER_WELL_LIMIT="${PER_WELL_LIMIT:-5000}"
ANOMALY_SENSOR_PARTITION_DAYS="${ANOMALY_SENSOR_PARTITION_DAYS:-250}"
SAGEMAKER_ROLE_NAME="${SAGEMAKER_ROLE_NAME:-anomaly-detection-sagemaker-role}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET or add to .env}"
ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID or add to .env}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
LAUNCHER_ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_LAUNCHER}:${IMAGE_TAG}"

echo "=== 0) Ensure SageMaker training role exists ==="
if ! aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" &>/dev/null; then
  cat > /tmp/sagemaker-trust.json << 'EOF'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
EOF
  aws iam create-role --role-name "$SAGEMAKER_ROLE_NAME" --assume-role-policy-document file:///tmp/sagemaker-trust.json
  aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
  aws iam attach-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  sleep 15
fi

echo "=== 1) Launcher ECR image ==="
aws ecr describe-repositories --repository-names "$ECR_REPO_LAUNCHER" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_LAUNCHER" --region "$REGION"

if [[ "${SKIP_ECR_BUILD:-0}" != "1" ]]; then
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f anomaly_detection/deploy/Dockerfile.fargate.sagemaker.launch -t "$LAUNCHER_ECR_URI" --push .
else
  echo "Skipping launcher ECR build (SKIP_ECR_BUILD=1)"
fi

echo ""
echo "=== 2) ECS cluster and log group ==="
aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" 2>/dev/null | grep -q '"status": "ACTIVE"' || \
  aws ecs create-cluster --cluster-name "$CLUSTER" --region "$REGION"
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true

echo ""
echo "=== 3) IAM roles ==="
EXECUTION_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}"
if ! aws iam get-role --role-name "$EXECUTION_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$EXECUTION_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  aws iam attach-role-policy --role-name "$EXECUTION_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  sleep 10
fi

TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
if ! aws iam get-role --role-name "$TASK_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$TASK_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  sleep 10
fi
aws iam put-role-policy --role-name "$TASK_ROLE_NAME" --policy-name "LaunchSageMakerTraining" --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"sagemaker:CreateTrainingJob\", \"sagemaker:DescribeTrainingJob\"], \"Resource\": \"*\"},
    {\"Effect\": \"Allow\", \"Action\": [\"iam:PassRole\"], \"Resource\": [\"${SAGEMAKER_ROLE_ARN}\"]}
  ]
}" >/dev/null

EVENTS_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EVENTBRIDGE_ROLE_NAME}"
if ! aws iam get-role --role-name "$EVENTBRIDGE_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$EVENTBRIDGE_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"events.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  aws iam put-role-policy --role-name "$EVENTBRIDGE_ROLE_NAME" --policy-name "RunScheduledTrainingTask" --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Effect\": \"Allow\", \"Action\": [\"ecs:RunTask\"], \"Resource\": [\"arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/${TASK_FAMILY}:*\"]},
      {\"Effect\": \"Allow\", \"Action\": [\"iam:PassRole\"], \"Resource\": [\"${EXECUTION_ROLE_ARN}\", \"${TASK_ROLE_ARN}\"],
       \"Condition\": {\"StringLike\": {\"iam:PassedToService\": \"ecs-tasks.amazonaws.com\"}}}
    ]
  }" >/dev/null
  sleep 10
fi

echo ""
echo "=== 4) Register ECS task definition ==="
TASK_DEF_PATH="$SCRIPT_DIR/ecs_sagemaker_launch_task_definition.json"
TASK_JSON=$(sed -e "s|EXECUTION_ROLE_ARN|${EXECUTION_ROLE_ARN}|g" \
  -e "s|TASK_ROLE_ARN|${TASK_ROLE_ARN}|g" \
  -e "s|IMAGE_URI|${LAUNCHER_ECR_URI}|g" \
  -e "s|S3_BUCKET_VALUE|${S3_BUCKET}|g" \
  -e "s|ORGANIZATION_ID_VALUE|${ORGANIZATION_ID}|g" \
  -e "s|AWS_REGION_VALUE|${REGION}|g" \
  -e "s|AWS_ACCOUNT_ID_VALUE|${ACCOUNT_ID}|g" \
  -e "s|SAGEMAKER_ROLE_ARN_VALUE|${SAGEMAKER_ROLE_ARN}|g" \
  -e "s|ECR_REPO_TRAIN_VALUE|${ECR_REPO_TRAIN}|g" \
  -e "s|IMAGE_TAG_VALUE|${IMAGE_TAG}|g" \
  -e "s|INSTANCE_TYPE_VALUE|${INSTANCE_TYPE}|g" \
  -e "s|PER_WELL_LIMIT_VALUE|${PER_WELL_LIMIT}|g" \
  -e "s|ANOMALY_SENSOR_PARTITION_DAYS_VALUE|${ANOMALY_SENSOR_PARTITION_DAYS}|g" \
  -e "s|LOG_GROUP_NAME|${LOG_GROUP}|g" \
  "$TASK_DEF_PATH")
aws ecs register-task-definition --region "$REGION" --cli-input-json "$TASK_JSON" >/dev/null

echo ""
echo "=== 5) EventBridge schedule ==="
SUBNETS="${ECS_SUBNETS:-}"
SECURITY_GROUPS="${ECS_SECURITY_GROUPS:-}"
if [[ -z "$SUBNETS" ]] || [[ -z "$SECURITY_GROUPS" ]]; then
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region "$REGION" 2>/dev/null || true)
  if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    echo "WARN: Set ECS_SUBNETS and ECS_SECURITY_GROUPS for Fargate."
    SUBNETS="subnet-00000000000000000"
    SECURITY_GROUPS="sg-00000000000000000"
  else
    SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region "$REGION" | tr '\t' ',')
    SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text --region "$REGION" 2>/dev/null || true)
    SECURITY_GROUPS="${SG:-}"
  fi
fi

aws events put-rule --name "$RULE_NAME" --schedule-expression "$SCHEDULE_CRON" --state ENABLED --region "$REGION" \
  --description "Weekly SageMaker training for anomaly_detection only"

_subnets_json="[\"$(echo "$SUBNETS" | sed 's/,/","/g')\"]"
_sgs_json="[\"$(echo "$SECURITY_GROUPS" | sed 's/,/","/g')\"]"
TARGET_JSON=$(cat <<TARGET
[{"Id":"SageMakerTrain","Arn":"arn:aws:ecs:${REGION}:${ACCOUNT_ID}:cluster/${CLUSTER}","RoleArn":"${EVENTS_ROLE_ARN}",
  "EcsParameters":{"TaskDefinitionArn":"arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/${TASK_FAMILY}",
    "TaskCount":1,"LaunchType":"FARGATE","NetworkConfiguration":{"awsvpcConfiguration":{"Subnets":$_subnets_json,"SecurityGroups":$_sgs_json,"AssignPublicIp":"ENABLED"}}}}]
TARGET
)
aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$REGION"

echo ""
echo "Done. anomaly_detection SageMaker training: $SCHEDULE_CRON (rule: $RULE_NAME)"
echo "Training image ECR: ${ECR_REPO_TRAIN}:${IMAGE_TAG} | Logs: $LOG_GROUP"
