#!/usr/bin/env bash
#
# Weekly SageMaker training schedule for Apex Permian anomaly detection.
#
# Usage (from repo root):
#   ./anomaly_detection/deploy/schedule_sagemaker_training_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export TASK_FAMILY="${TASK_FAMILY:-anomaly-detection-apex-sagemaker-train-launch}"
export RULE_NAME="${RULE_NAME:-anomaly-detection-apex-sagemaker-train-weekly}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-detection-apex-sagemaker-train}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-anomaly-detection-apex-sagemaker-train-events-role}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(30 2 ? * MON *)}"

exec "$SCRIPT_DIR/schedule_sagemaker_training.sh"
