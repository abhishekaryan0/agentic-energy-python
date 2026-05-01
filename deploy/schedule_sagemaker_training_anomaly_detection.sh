#!/usr/bin/env bash
#
# Schedule weekly SageMaker training for anomaly_detection only (EventBridge -> ECS Fargate).
#
set -euo pipefail
export AWS_PAGER=""

MODEL_KEY="anomaly_detection"
DEFAULT_RULE_NAME="sagemaker-train-anomaly-detection"
DEFAULT_TASK_FAMILY="sagemaker-train-anomaly-detection-batch"
DEFAULT_CLUSTER="sagemaker-training-cluster"
DEFAULT_LOG_GROUP="/ecs/sagemaker-train-anomaly-detection"
DEFAULT_EVENTBRIDGE_ROLE_NAME="ecs-sagemaker-train-anomaly-detection-events-role"
DEFAULT_EXECUTION_ROLE_NAME="ecs-sagemaker-train-anomaly-detection-execution-role"
DEFAULT_TASK_ROLE_NAME="ecs-sagemaker-train-anomaly-detection-task-role"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

export MODEL_KEY
export RULE_NAME="${RULE_NAME:-$DEFAULT_RULE_NAME}"
export TASK_FAMILY="${TASK_FAMILY:-$DEFAULT_TASK_FAMILY}"
export ECS_CLUSTER="${ECS_CLUSTER:-$DEFAULT_CLUSTER}"
export LOG_GROUP="${LOG_GROUP:-$DEFAULT_LOG_GROUP}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-$DEFAULT_EVENTBRIDGE_ROLE_NAME}"
export EXECUTION_ROLE_NAME="${EXECUTION_ROLE_NAME:-$DEFAULT_EXECUTION_ROLE_NAME}"
export TASK_ROLE_NAME="${TASK_ROLE_NAME:-$DEFAULT_TASK_ROLE_NAME}"

# Default: Mondays 02:00 UTC (override any time)
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 2 ? * MON *)}"

./deploy/schedule_sagemaker_training.sh

