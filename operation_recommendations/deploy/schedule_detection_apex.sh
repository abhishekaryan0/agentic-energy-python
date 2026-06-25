#!/usr/bin/env bash
#
# Weekly operation recommendations detection for Apex Permian.
#
# Usage:
#   ./operation_recommendations/deploy/schedule_detection_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export TASK_FAMILY="${TASK_FAMILY:-operations-detection-apex-permian}"
export RULE_NAME="${RULE_NAME:-operations-detection-apex-permian-weekly}"
export LOG_GROUP="${LOG_GROUP:-/ecs/operations-detection-apex-permian}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-operations-apex-detection-events-role}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 3 ? * MON *)}"
export OPS_DETECTION_DAYS="${OPS_DETECTION_DAYS:-7}"
export OPS_DETECTION_FORCE_WINDOW="${OPS_DETECTION_FORCE_WINDOW:-0}"

exec "$SCRIPT_DIR/schedule_detection.sh"
