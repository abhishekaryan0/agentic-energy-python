#!/usr/bin/env bash
#
# Schedule weekly operation recommendations detection (ECS Fargate + EventBridge).
#
# Usage (from repo root):
#   export S3_BUCKET=...
#   export ORGANIZATION_ID=...
#   ./operation_recommendations/deploy/schedule_detection.sh
#
set -euo pipefail
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${ECS_CLUSTER:-operations-detection-cluster}"
TASK_FAMILY="${TASK_FAMILY:-operations-detection-batch}"
LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection}"
RULE_NAME="${RULE_NAME:-operations-detection-weekly}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 2 ? * MON *)}"

EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-operations-detection-events-role}"
EXECUTION_ROLE_NAME="${EXECUTION_ROLE_NAME:-ecs-operations-detection-execution-role}"
TASK_ROLE_NAME="${TASK_ROLE_NAME:-ecs-operations-detection-task-role}"

ECR_REPO_NAME="${ECR_REPO_OPERATIONS_DETECTION:-operations-detection-fargate}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

_SAVED_ORGANIZATION_ID="${ORGANIZATION_ID:-}"
_SAVED_TASK_FAMILY="${TASK_FAMILY:-}"
_SAVED_RULE_NAME="${RULE_NAME:-}"
_SAVED_LOG_GROUP="${LOG_GROUP:-}"
_SAVED_EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-}"
_SAVED_OPERATIONS_MODEL_TARBALL_S3_URI="${OPERATIONS_MODEL_TARBALL_S3_URI:-}"
_SAVED_OPS_DETECTION_DAYS="${OPS_DETECTION_DAYS:-}"
_SAVED_OPS_DETECTION_FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -n "$_SAVED_ORGANIZATION_ID" ]]; then
  export ORGANIZATION_ID="$_SAVED_ORGANIZATION_ID"
fi
if [[ -n "$_SAVED_TASK_FAMILY" ]]; then
  export TASK_FAMILY="$_SAVED_TASK_FAMILY"
fi
if [[ -n "$_SAVED_RULE_NAME" ]]; then
  export RULE_NAME="$_SAVED_RULE_NAME"
fi
if [[ -n "$_SAVED_LOG_GROUP" ]]; then
  export LOG_GROUP="$_SAVED_LOG_GROUP"
fi
if [[ -n "$_SAVED_EVENTBRIDGE_ROLE_NAME" ]]; then
  export EVENTBRIDGE_ROLE_NAME="$_SAVED_EVENTBRIDGE_ROLE_NAME"
fi
if [[ -n "$_SAVED_OPERATIONS_MODEL_TARBALL_S3_URI" ]]; then
  export OPERATIONS_MODEL_TARBALL_S3_URI="$_SAVED_OPERATIONS_MODEL_TARBALL_S3_URI"
fi
if [[ -n "$_SAVED_OPS_DETECTION_DAYS" ]]; then
  export OPS_DETECTION_DAYS="$_SAVED_OPS_DETECTION_DAYS"
fi
if [[ -n "$_SAVED_OPS_DETECTION_FORCE_WINDOW" ]]; then
  export OPS_DETECTION_FORCE_WINDOW="$_SAVED_OPS_DETECTION_FORCE_WINDOW"
fi

TASK_FAMILY="${TASK_FAMILY:-operations-detection-batch}"
LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection}"
EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-operations-detection-events-role}"
OPS_DETECTION_DAYS="${OPS_DETECTION_DAYS:-7}"
OPS_DETECTION_FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-0}"

: "${S3_BUCKET:?Set S3_BUCKET}"
: "${ORGANIZATION_ID:?Set ORGANIZATION_ID}"

ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

if [[ -z "${OPERATIONS_MODEL_TARBALL_S3_URI:-}" ]]; then
  latest_key="$(aws s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "${ORGANIZATION_ID}/model/" \
    --query 'Contents[?contains(Key, `operations-train`) && ends_with(Key, `model.tar.gz`)]|sort_by(@,&LastModified)[-1].Key' --output text --region "$REGION")"
  if [[ -z "$latest_key" || "$latest_key" == "None" ]]; then
    echo "ERROR: OPERATIONS_MODEL_TARBALL_S3_URI not set and no model.tar.gz found under s3://${S3_BUCKET}/${ORGANIZATION_ID}/model/"
    exit 1
  fi
  export OPERATIONS_MODEL_TARBALL_S3_URI="s3://${S3_BUCKET}/${latest_key}"
fi

echo "=== 1) ECR image ==="
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" >/dev/null 2>&1 || true
fi

if [[ "${SKIP_ECR_BUILD:-0}" != "1" ]]; then
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f operation_recommendations/deploy/Dockerfile.fargate.detect -t "$ECR_URI" --push .
else
  echo "Skipping ECR build (SKIP_ECR_BUILD=1)"
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
  aws iam attach-role-policy --role-name "$EXECUTION_ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  sleep 10
fi

TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
if ! aws iam get-role --role-name "$TASK_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$TASK_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  aws iam attach-role-policy --role-name "$TASK_ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  sleep 10
fi

if [[ -n "${DATABASE_URL_SECRET_ARN:-}" ]]; then
  aws iam put-role-policy --role-name "$TASK_ROLE_NAME" --policy-name "OperationsDetectionSecrets" \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [{
        \"Effect\": \"Allow\",
        \"Action\": [\"secretsmanager:GetSecretValue\"],
        \"Resource\": [\"${DATABASE_URL_SECRET_ARN}\"]
      }]
    }" 2>/dev/null || true
fi

EVENTS_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EVENTBRIDGE_ROLE_NAME}"
if ! aws iam get-role --role-name "$EVENTBRIDGE_ROLE_NAME" &>/dev/null; then
  aws iam create-role --role-name "$EVENTBRIDGE_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"events.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  aws iam put-role-policy --role-name "$EVENTBRIDGE_ROLE_NAME" --policy-name "RunOperationsDetectionTask" --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Effect\": \"Allow\", \"Action\": [\"ecs:RunTask\"], \"Resource\": [\"arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/${TASK_FAMILY}:*\"]},
      {\"Effect\": \"Allow\", \"Action\": [\"iam:PassRole\"], \"Resource\": [\"${EXECUTION_ROLE_ARN}\", \"${TASK_ROLE_ARN}\"],
       \"Condition\": {\"StringLike\": {\"iam:PassedToService\": \"ecs-tasks.amazonaws.com\"}}}
    ]
  }"
  sleep 10
fi

echo ""
echo "=== 4) Register ECS task definition ==="
TASK_DEF_PATH="$SCRIPT_DIR/ecs_task_definition.json"
export LOG_GROUP REGION TASK_FAMILY OPS_DETECTION_DAYS OPS_DETECTION_FORCE_WINDOW
TASK_JSON="$(python3 - "$TASK_DEF_PATH" "$ECR_URI" "$EXECUTION_ROLE_ARN" "$TASK_ROLE_ARN" <<'PY'
import json, os, sys
path, ecr, exec_arn, task_arn = sys.argv[1:5]
text = open(path).read()
repl = {
  "TASK_FAMILY_VALUE": os.environ.get("TASK_FAMILY", "operations-detection-batch"),
  "EXECUTION_ROLE_ARN": exec_arn,
  "TASK_ROLE_ARN": task_arn,
  "IMAGE_URI": ecr,
  "LOG_GROUP_NAME": os.environ["LOG_GROUP"],
  "AWS_REGION_VALUE": os.environ["REGION"],
  "S3_BUCKET_VALUE": os.environ["S3_BUCKET"],
  "ORGANIZATION_ID_VALUE": os.environ["ORGANIZATION_ID"],
  "OPERATIONS_MODEL_TARBALL_S3_URI_VALUE": os.environ["OPERATIONS_MODEL_TARBALL_S3_URI"],
  "OPS_DETECTION_DAYS_VALUE": os.environ.get("OPS_DETECTION_DAYS", "7"),
  "OPS_DETECTION_FORCE_WINDOW_VALUE": os.environ.get("OPS_DETECTION_FORCE_WINDOW", "0"),
  "EIA_API_KEY_VALUE": os.environ.get("EIA_API_KEY", ""),
  "DATABASE_URL_VALUE": os.environ.get("DATABASE_URL", ""),
  "DATABASE_URL_SECRET_ARN_VALUE": os.environ.get("DATABASE_URL_SECRET_ARN", ""),
  "OPENAI_API_KEY_VALUE": os.environ.get("OPENAI_API_KEY", ""),
  "OPENAI_API_BASE_VALUE": os.environ.get("OPENAI_API_BASE", ""),
}
for k, v in repl.items():
  text = text.replace(k, v)
json.loads(text)
print(text)
PY
)"
aws ecs register-task-definition --region "$REGION" --cli-input-json "$TASK_JSON" >/dev/null
echo "Registered task definition: $TASK_FAMILY"

echo ""
echo "=== 5) EventBridge schedule ==="
SUBNETS="${ECS_SUBNETS:-}"
SECURITY_GROUPS="${ECS_SECURITY_GROUPS:-}"
if [[ -z "$SUBNETS" ]] || [[ -z "$SECURITY_GROUPS" ]]; then
  VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$REGION" 2>/dev/null || true)"
  SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --query 'Subnets[*].SubnetId' --output text --region "$REGION" | tr '\t' ',')"
  SG="$(aws ec2 describe-security-groups --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values=default --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || true)"
  SECURITY_GROUPS="${SG:-}"
fi

aws events put-rule --name "$RULE_NAME" --schedule-expression "$SCHEDULE_CRON" --state ENABLED --region "$REGION" \
  --description "Run weekly operation recommendations detection (Fargate)"

_subnets_json="[\"$(echo "$SUBNETS" | sed 's/,/","/g')\"]"
_sgs_json="[\"$(echo "$SECURITY_GROUPS" | sed 's/,/","/g')\"]"
TARGET_JSON=$(cat <<TARGET
[
  {
    "Id": "OperationsDetectionTask",
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

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$REGION" >/dev/null

echo ""
echo "Scheduled operation recommendations detection"
echo "  Rule:     $RULE_NAME"
echo "  Cron:     $SCHEDULE_CRON"
echo "  Cluster:  $CLUSTER"
echo "  Task:     $TASK_FAMILY"
echo "  Model:    $OPERATIONS_MODEL_TARBALL_S3_URI"
echo "  Window:   OPS_DETECTION_DAYS=$OPS_DETECTION_DAYS FORCE_WINDOW=$OPS_DETECTION_FORCE_WINDOW"
echo "  Logs:     aws logs tail $LOG_GROUP --follow --region $REGION"
