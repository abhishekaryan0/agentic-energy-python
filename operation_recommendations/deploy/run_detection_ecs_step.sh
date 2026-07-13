#!/usr/bin/env bash
#
# One-off STEP operations recommendations detection (real S3 write).
#
# Usage:
#   ./operation_recommendations/deploy/run_detection_ecs_step.sh
#   OPS_DETECTION_DAYS=30 OPS_DETECTION_FORCE_WINDOW=1 ./operation_recommendations/deploy/run_detection_ecs_step.sh
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
export ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-operations-detection-step}"
export LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection-step}"

exec "$SCRIPT_DIR/run_detection_ecs.sh"
