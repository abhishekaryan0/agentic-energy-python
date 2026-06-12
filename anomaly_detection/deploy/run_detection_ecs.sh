#!/usr/bin/env bash
#
# One-off ECS Fargate run: score the last N sensor rows + LLM on flags.
# Does NOT read or update the incremental checkpoint — scheduled daily
# incremental jobs continue unchanged after this.
#
# Usage (from repo root):
#   source .env
#   ./anomaly_detection/deploy/run_detection_ecs.sh
#
# Optional:
#   ANOMALY_LAST_ROWS=100
#   ANOMALY_MODEL_TRAINING_JOB=anomaly-train-...
#   ECS_SUBNETS=...  ECS_SECURITY_GROUPS=...
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${ECS_CLUSTER:-anomaly-detection-cluster}"
TASK_FAMILY="${ECS_TASK_FAMILY:-anomaly-detection-batch}"
MODE="recent"
LAST_ROWS="${ANOMALY_LAST_ROWS:-100}"
MODEL_JOB="${ANOMALY_MODEL_TRAINING_JOB:-anomaly-train-20260612-1337}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
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

OVERRIDES=$(cat <<EOF
{
  "containerOverrides": [{
    "name": "anomaly-detection",
    "environment": [
      {"name": "ANOMALY_DETECTION_MODE", "value": "${MODE}"},
      {"name": "ANOMALY_LAST_ROWS", "value": "${LAST_ROWS}"},
      {"name": "ANOMALY_ENABLE_LLM_SUGGESTIONS", "value": "1"},
      {"name": "ANOMALY_MODEL_TRAINING_JOB", "value": "${MODEL_JOB}"},
      {"name": "ANOMALY_RESET_CHECKPOINT", "value": "0"}
    ]
  }]
}
EOF
)

echo "One-off ECS task (no checkpoint): cluster=$CLUSTER last_rows=$LAST_ROWS model_job=$MODEL_JOB"
echo "Scheduled incremental detection is unchanged."
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
echo "Logs: aws logs tail /ecs/anomaly-detection --follow --region $REGION"
