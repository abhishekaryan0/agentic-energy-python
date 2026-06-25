#!/usr/bin/env bash
#
# One-off Apex Permian anomaly review: last 7 days (real write), then weekly
# schedule continues incrementally from checkpoint (FORCE_WINDOW=0).
#
# Usage:
#   ./anomaly_review/deploy/run_detection_ecs_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-anomaly-review-detection-apex-permian}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-review-detection-apex-permian}"
export ANOMALY_REVIEW_FORCE_WINDOW="${ANOMALY_REVIEW_FORCE_WINDOW:-1}"
export ANOMALY_REVIEW_INITIAL_DAYS="${ANOMALY_REVIEW_INITIAL_DAYS:-7}"
export ANOMALY_REVIEW_FETCH_DAYS="${ANOMALY_REVIEW_FETCH_DAYS:-180}"

exec "$SCRIPT_DIR/run_detection_ecs.sh"
