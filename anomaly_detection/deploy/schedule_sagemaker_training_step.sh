#!/usr/bin/env bash
#
# Weekly SageMaker training schedule for South Texas Energy Partners anomaly detection.
#
# Usage (from repo root):
#   ./anomaly_detection/deploy/schedule_sagemaker_training_step.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export ORGANIZATION_ID="${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}"
export TASK_FAMILY="${TASK_FAMILY:-anomaly-detection-step-sagemaker-train-launch}"
export RULE_NAME="${RULE_NAME:-anomaly-detection-step-sagemaker-train-weekly}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-detection-step-sagemaker-train}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-anomaly-detection-step-sagemaker-train-events-role}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 3 ? * MON *)}"

exec "$SCRIPT_DIR/schedule_sagemaker_training.sh"
