#!/usr/bin/env bash
#
# Weekly anomaly review detection for Apex Permian.
#
# Usage:
#   ./anomaly_review/deploy/schedule_detection_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export TASK_FAMILY="${TASK_FAMILY:-anomaly-review-detection-apex-permian}"
export RULE_NAME="${RULE_NAME:-anomaly-review-detection-apex-permian-weekly}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-review-detection-apex-permian}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-anomaly-review-apex-detection-events-role}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(15 3 ? * MON *)}"
export ANOMALY_REVIEW_FORCE_WINDOW="${ANOMALY_REVIEW_FORCE_WINDOW:-0}"

exec "$SCRIPT_DIR/schedule_detection.sh"
