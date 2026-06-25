#!/usr/bin/env bash
#
# One-off Apex Permian operations recommendations: score last 7 days (real write),
# then weekly schedule continues incrementally from checkpoint (FORCE_WINDOW=0).
#
# Usage:
#   ./operation_recommendations/deploy/run_detection_ecs_apex.sh
#   OPS_DETECTION_DAYS=7 ./operation_recommendations/deploy/run_detection_ecs_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-operations-detection-apex-permian}"
export LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection-apex-permian}"
export OPS_DETECTION_DAYS="${OPS_DETECTION_DAYS:-7}"
export OPS_DETECTION_FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-1}"
export OPS_RESET_CHECKPOINT="${OPS_RESET_CHECKPOINT:-0}"

exec "$SCRIPT_DIR/run_detection_ecs.sh"
