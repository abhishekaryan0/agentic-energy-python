#!/usr/bin/env bash
#
# One-off STEP detection bootstrap: score last N rows, write checkpoint, then daily
# schedule continues incrementally.
#
# Usage:
#   ./anomaly_detection/deploy/run_detection_ecs_step.sh
#   ANOMALY_BOOTSTRAP_ROWS=100 ./anomaly_detection/deploy/run_detection_ecs_step.sh
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
export ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-anomaly-detection-step}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-detection-step}"
export ANOMALY_DETECTION_MODE="${ANOMALY_DETECTION_MODE:-incremental}"
export ANOMALY_BOOTSTRAP_ROWS="${ANOMALY_BOOTSTRAP_ROWS:-${ANOMALY_LAST_ROWS:-100}}"
export ANOMALY_LAST_ROWS="${ANOMALY_LAST_ROWS:-$ANOMALY_BOOTSTRAP_ROWS}"
export ANOMALY_ENABLE_LLM_SUGGESTIONS="${ANOMALY_ENABLE_LLM_SUGGESTIONS:-1}"
export ANOMALY_RESET_CHECKPOINT="${ANOMALY_RESET_CHECKPOINT:-0}"

exec "$SCRIPT_DIR/run_detection_ecs.sh"
