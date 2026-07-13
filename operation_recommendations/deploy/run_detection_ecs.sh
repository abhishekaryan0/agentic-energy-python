#!/usr/bin/env bash
#
# One-off ECS Fargate run for operation recommendations (real S3 write).
# Override window/checkpoint env vars; scheduled task definition stays unchanged.
#
# Usage (from repo root):
#   source .env
#   ./operation_recommendations/deploy/run_detection_ecs.sh
#
# Bootstrap last N days (writes checkpoint for incremental schedule after):
#   OPS_DETECTION_DAYS=7 OPS_DETECTION_FORCE_WINDOW=1 ./operation_recommendations/deploy/run_detection_ecs.sh
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${ECS_CLUSTER:-operations-detection-cluster}"
TASK_FAMILY="${ECS_TASK_FAMILY:-operations-detection-batch}"
LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection}"
DETECTION_DAYS="${OPS_DETECTION_DAYS:-7}"
FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-1}"
RESET_CHECKPOINT="${OPS_RESET_CHECKPOINT:-0}"
MIN_SENSOR_ROWS="${OPS_MIN_SENSOR_ROWS_PER_WELL:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

_SAVED_ORGANIZATION_ID="${ORGANIZATION_ID:-}"
_SAVED_ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-}"
_SAVED_LOG_GROUP="${LOG_GROUP:-}"
_SAVED_DETECTION_DAYS="${OPS_DETECTION_DAYS:-}"
_SAVED_FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-}"
_SAVED_RESET_CHECKPOINT="${OPS_RESET_CHECKPOINT:-}"
_SAVED_MIN_SENSOR_ROWS="${OPS_MIN_SENSOR_ROWS_PER_WELL:-}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -n "$_SAVED_ORGANIZATION_ID" ]]; then
  export ORGANIZATION_ID="$_SAVED_ORGANIZATION_ID"
fi
if [[ -n "$_SAVED_ECS_TASK_FAMILY" ]]; then
  export ECS_TASK_FAMILY="$_SAVED_ECS_TASK_FAMILY"
  TASK_FAMILY="$_SAVED_ECS_TASK_FAMILY"
fi
if [[ -n "$_SAVED_LOG_GROUP" ]]; then
  LOG_GROUP="$_SAVED_LOG_GROUP"
fi
if [[ -n "$_SAVED_DETECTION_DAYS" ]]; then
  DETECTION_DAYS="$_SAVED_DETECTION_DAYS"
fi
if [[ -n "$_SAVED_FORCE_WINDOW" ]]; then
  FORCE_WINDOW="$_SAVED_FORCE_WINDOW"
fi
if [[ -n "$_SAVED_RESET_CHECKPOINT" ]]; then
  RESET_CHECKPOINT="$_SAVED_RESET_CHECKPOINT"
fi
if [[ -n "$_SAVED_MIN_SENSOR_ROWS" ]]; then
  MIN_SENSOR_ROWS="$_SAVED_MIN_SENSOR_ROWS"
fi

S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID}"

SUBNETS="${ECS_SUBNETS:-}"
SECURITY_GROUPS="${ECS_SECURITY_GROUPS:-}"
if [[ -z "$SUBNETS" ]] || [[ -z "$SECURITY_GROUPS" ]]; then
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region "$REGION" 2>/dev/null || true)
  if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    echo "ERROR: Set ECS_SUBNETS and ECS_SECURITY_GROUPS" >&2
    exit 1
  fi
  SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region "$REGION" | tr '\t' ',')
  SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text --region "$REGION" 2>/dev/null || true)
  SECURITY_GROUPS="${SG:-}"
fi

TASK_DEF=$(aws ecs describe-task-definition --task-definition "$TASK_FAMILY" --region "$REGION" --query 'taskDefinition.taskDefinitionArn' --output text)

ENV_OVERRIDES=$(cat <<EOF
      {"name": "OPS_DETECTION_DAYS", "value": "${DETECTION_DAYS}"},
      {"name": "OPS_DETECTION_FORCE_WINDOW", "value": "${FORCE_WINDOW}"},
      {"name": "OPS_RESET_CHECKPOINT", "value": "${RESET_CHECKPOINT}"}
EOF
)
if [[ -n "$MIN_SENSOR_ROWS" ]]; then
  ENV_OVERRIDES+=$(cat <<EOF
,
      {"name": "OPS_MIN_SENSOR_ROWS_PER_WELL", "value": "${MIN_SENSOR_ROWS}"}
EOF
)
fi

OVERRIDES=$(cat <<EOF
{
  "containerOverrides": [{
    "name": "operations-detection",
    "environment": [
${ENV_OVERRIDES}
    ]
  }]
}
EOF
)

echo "One-off ECS task: org=$ORGANIZATION_ID cluster=$CLUSTER family=$TASK_FAMILY"
echo "days=$DETECTION_DAYS force_window=$FORCE_WINDOW reset_checkpoint=$RESET_CHECKPOINT min_sensor_rows=${MIN_SENSOR_ROWS:-default}"
TASK_ARN=$(aws ecs run-task \
  --cluster "$CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$(echo "$SUBNETS" | sed 's/,/,/g')],securityGroups=[${SECURITY_GROUPS}],assignPublicIp=ENABLED}" \
  --overrides "$OVERRIDES" \
  --region "$REGION" \
  --query 'tasks[0].taskArn' \
  --output text)

echo "Task started: $TASK_ARN"
echo "Logs: aws logs tail ${LOG_GROUP} --follow --region ${REGION}"
