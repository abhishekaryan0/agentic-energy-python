#!/usr/bin/env bash
#
# Schedule Apex Permian anomaly detection (separate ECS task from Red Buff).
#
# Usage (from repo root):
#   ./anomaly_detection/deploy/schedule_detection_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export TASK_FAMILY="${TASK_FAMILY:-anomaly-detection-apex-permian}"
export RULE_NAME="${RULE_NAME:-anomaly-detection-apex-permian-daily}"
export LOG_GROUP="${LOG_GROUP:-/ecs/anomaly-detection-apex-permian}"
export EVENTBRIDGE_ROLE_NAME="${EVENTBRIDGE_ROLE_NAME:-ecs-anomaly-detection-apex-events-role}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(45 0 * * ? *)}"
export ANOMALY_MODEL_TRAINING_JOB="${APEX_ANOMALY_MODEL_TRAINING_JOB:-anomaly-train-apex-20260618-1224}"

# Stricter detection for Apex Permian (fewer false positives than Red Buff defaults).
export ANOMALY_THRESHOLD="${APEX_ANOMALY_THRESHOLD:-0.85}"
export ANOMALY_ML_ONLY_THRESHOLD="${APEX_ANOMALY_ML_ONLY_THRESHOLD:-0.92}"
export ANOMALY_RULE_DEVIATION_PCT="${APEX_ANOMALY_RULE_DEVIATION_PCT:-45}"
export ANOMALY_VIOLATION_MIN_BEYOND_PCT="${APEX_ANOMALY_VIOLATION_MIN_BEYOND_PCT:-15}"
export ANOMALY_MIN_SCORE_WITH_VIOLATIONS="${APEX_ANOMALY_MIN_SCORE_WITH_VIOLATIONS:-0.55}"

exec "$SCRIPT_DIR/schedule_detection.sh"
