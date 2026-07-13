#!/usr/bin/env bash
#
# Schedule South Texas Energy Partners anomaly detection (separate ECS task).
#
# Usage (from repo root):
#   ./anomaly_detection/deploy/schedule_detection_step.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

_SAVED_ORGANIZATION_ID="${ORGANIZATION_ID:-}"
_SAVED_TASK_FAMILY="${TASK_FAMILY:-}"
_SAVED_RULE_NAME="${RULE_NAME:-}"
_SAVED_LOG_GROUP="${LOG_GROUP:-}"
_SAVED_EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-}"
_SAVED_SCHEDULE_CRON="${SCHEDULE_CRON:-}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export ORGANIZATION_ID="${_SAVED_ORGANIZATION_ID:-${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}}"
export TASK_FAMILY="${_SAVED_TASK_FAMILY:-anomaly-detection-step}"
export RULE_NAME="${_SAVED_RULE_NAME:-anomaly-detection-step-daily}"
export LOG_GROUP="${_SAVED_LOG_GROUP:-/ecs/anomaly-detection-step}"
export EVENTBRIDGE_ROLE_NAME="${_SAVED_EVENTBRIDGE_ROLE_NAME:-ecs-anomaly-detection-step-events-role}"
export SCHEDULE_CRON="${_SAVED_SCHEDULE_CRON:-cron(0 1 * * ? *)}"

exec "$SCRIPT_DIR/schedule_detection.sh"
