#!/usr/bin/env bash
#
# One-off STEP anomaly review bootstrap (real S3 write + checkpoint).
#
# Usage:
#   ./anomaly_review/deploy/run_detection_ecs_step.sh
#   ANOMALY_REVIEW_INITIAL_DAYS=30 ./anomaly_review/deploy/run_detection_ecs_step.sh
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
export ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-anomaly-review-detection-step}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-review-detection-step}"
export ANOMALY_REVIEW_INITIAL_DAYS="${ANOMALY_REVIEW_INITIAL_DAYS:-30}"
export ANOMALY_REVIEW_FETCH_DAYS="${ANOMALY_REVIEW_FETCH_DAYS:-180}"
# Bootstrap always reprocesses the window (ignore inherited deploy-shell env).
export ANOMALY_REVIEW_FORCE_WINDOW=1

exec "$SCRIPT_DIR/run_detection_ecs.sh"
