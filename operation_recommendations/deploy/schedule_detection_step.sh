#!/usr/bin/env bash
#
# Weekly operation recommendations detection for South Texas Energy Partners.
#
# Usage:
#   ./operation_recommendations/deploy/schedule_detection_step.sh
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
export TASK_FAMILY="${TASK_FAMILY:-operations-detection-step}"
export RULE_NAME="${RULE_NAME:-operations-detection-step-weekly}"
export LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection-step}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-operations-step-detection-events-role}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(30 1 ? * MON *)}"
export OPS_DETECTION_DAYS="${OPS_DETECTION_DAYS:-7}"
export OPS_DETECTION_FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-0}"

exec "$SCRIPT_DIR/schedule_detection.sh"
