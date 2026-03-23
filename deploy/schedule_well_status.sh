#!/usr/bin/env bash
#
# Schedule well status update batch job for production (ECS Fargate + EventBridge).
#
# What this does:
#  1. Builds and pushes the well-status Docker image to ECR.
#  2. Ensures ECS cluster, log group, and IAM roles exist.
#  3. Registers the Fargate task definition with S3_BUCKET and ORGANIZATION_ID from env.
#  4. Creates an EventBridge rule to run the task daily (default: 02:00 UTC).
#
# Usage (from repo root):
#   export S3_BUCKET=agentic-energy-ml-models-200937443240
#   export ORGANIZATION_ID=147a92b9-f90a-473d-b443-e6e668c4fbb7
#   ./deploy/schedule_well_status.sh
#
# Optional overrides:
#   SCHEDULE_CRON="cron(0 2 * * ? *)"   # default: daily 2 AM UTC
#   ECS_CLUSTER=default
#   AWS_REGION=us-east-1
#   SKIP_ECR_BUILD=1
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${ECS_CLUSTER:-well-status-cluster}"
TASK_FAMILY="well-status-batch"
LOG_GROUP="/ecs/well-status"
EVENTBRIDGE_ROLE_NAME="ecs-well-status-events-role"
EXECUTION_ROLE_NAME="ecs-well-status-execution-role"
TASK_ROLE_NAME="ecs-well-status-task-role"
ECR_REPO_NAME="${ECR_REPO_WELL_STATUS:-well-status-fargate}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 2 * * ? *)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET or add to .env}"
ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID or add to .env}"

ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo "=== 1) ECR image ==="
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION"

if [[ "${SKIP_ECR_BUILD:-0}" != "1" ]]; then
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f deploy/Dockerfile.fargate.well_status -t "$ECR_URI" --push .
else
  echo "Skipping ECR build (SKIP_ECR_BUILD=1)"
fi

echo ""
echo "=== 2) ECS cluster and log group ==="
aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" 2>/dev/null | grep -q '"status": "ACTIVE"' || \
  aws ecs create-cluster --cluster-name "$CLUSTER" --region "$REGION"
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true

echo ""
echo "=== 3) IAM roles for ECS and EventBridge ==="
EXECUTION_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}"
if ! aws iam get-role --role-name "$EXECUTION_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$EXECUTION_ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }'
  aws iam attach-role-policy --role-name "$EXECUTION_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  echo "Created execution role $EXECUTION_ROLE_NAME; wait 10s for propagation."
  sleep 10
fi

TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
if ! aws iam get-role --role-name "$TASK_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$TASK_ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }'
  aws iam attach-role-policy --role-name "$TASK_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  echo "Created task role $TASK_ROLE_NAME; wait 10s for propagation."
  sleep 10
fi

EVENTS_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EVENTBRIDGE_ROLE_NAME}"
if ! aws iam get-role --role-name "$EVENTBRIDGE_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$EVENTBRIDGE_ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "events.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }'
  aws iam put-role-policy --role-name "$EVENTBRIDGE_ROLE_NAME" --policy-name "RunWellStatusTask" --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"ecs:RunTask\"],
        \"Resource\": [\"arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/${TASK_FAMILY}:*\"]
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"iam:PassRole\"],
        \"Resource\": [\"${EXECUTION_ROLE_ARN}\", \"${TASK_ROLE_ARN}\"],
        \"Condition\": {\"StringLike\": {\"iam:PassedToService\": \"ecs-tasks.amazonaws.com\"}}
      }
    ]
  }"
  echo "Created EventBridge role $EVENTBRIDGE_ROLE_NAME; wait 10s for propagation."
  sleep 10
fi

echo ""
echo "=== 4) Register ECS task definition ==="
TASK_DEF_PATH="$SCRIPT_DIR/ecs_well_status_task_definition.json"
TASK_JSON=$(sed -e "s|EXECUTION_ROLE_ARN|${EXECUTION_ROLE_ARN}|g" \
  -e "s|TASK_ROLE_ARN|${TASK_ROLE_ARN}|g" \
  -e "s|IMAGE_URI|${ECR_URI}|g" \
  -e "s|S3_BUCKET_VALUE|${S3_BUCKET}|g" \
  -e "s|ORGANIZATION_ID_VALUE|${ORGANIZATION_ID}|g" \
  -e "s|AWS_REGION_VALUE|${REGION}|g" \
  -e "s|LOG_GROUP_NAME|${LOG_GROUP}|g" \
  "$TASK_DEF_PATH")

aws ecs register-task-definition --region "$REGION" --cli-input-json "$TASK_JSON" >/dev/null
echo "Registered task definition: $TASK_FAMILY"

echo ""
echo "=== 5) EventBridge schedule ==="
SUBNETS="${ECS_SUBNETS:-}"
SECURITY_GROUPS="${ECS_SECURITY_GROUPS:-}"
if [[ -z "$SUBNETS" ]] || [[ -z "$SECURITY_GROUPS" ]]; then
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region "$REGION" 2>/dev/null || true)
  if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    echo "WARN: No default VPC. Set ECS_SUBNETS and ECS_SECURITY_GROUPS for Fargate."
    SUBNETS="subnet-00000000000000000"
    SECURITY_GROUPS="sg-00000000000000000"
  else
    SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region "$REGION" | tr '\t' ',')
    SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text --region "$REGION" 2>/dev/null || true)
    SECURITY_GROUPS="${SG:-}"
  fi
fi

RULE_NAME="well-status-daily"
aws events put-rule --name "$RULE_NAME" --schedule-expression "$SCHEDULE_CRON" --state ENABLED --region "$REGION" --description "Run well status update batch job (Fargate) daily"

_subnets_json="[\"$(echo "$SUBNETS" | sed 's/,/","/g')\"]"
_sgs_json="[\"$(echo "$SECURITY_GROUPS" | sed 's/,/","/g')\"]"
TARGET_JSON=$(cat <<TARGET
[
  {
    "Id": "WellStatusTask",
    "Arn": "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:cluster/${CLUSTER}",
    "RoleArn": "${EVENTS_ROLE_ARN}",
    "EcsParameters": {
      "TaskDefinitionArn": "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/${TASK_FAMILY}",
      "TaskCount": 1,
      "LaunchType": "FARGATE",
      "NetworkConfiguration": {
        "awsvpcConfiguration": {
          "Subnets": $_subnets_json,
          "SecurityGroups": $_sgs_json,
          "AssignPublicIp": "ENABLED"
        }
      }
    }
  }
]
TARGET
)

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$REGION"
echo "Schedule: $SCHEDULE_CRON (rule: $RULE_NAME) -> ECS RunTask on $CLUSTER"

echo ""
echo "Done. Well status will run daily on schedule."
echo "To run once now: aws ecs run-task --cluster $CLUSTER --task-definition $TASK_FAMILY --launch-type FARGATE --network-configuration \"awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=ENABLED}\" --region $REGION"
echo "Logs: CloudWatch log group $LOG_GROUP"
