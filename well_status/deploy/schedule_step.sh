#!/usr/bin/env bash
#
# Schedule South Texas Energy Partners well status via EventBridge -> Lambda (daily).
#
# Usage (from repo root):
#   ./well_status/deploy/schedule_step.sh
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export ORGANIZATION_ID="${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}"
export LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-well-status-step}"
export LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-well-status-step-lambda-role}"
export RULE_NAME="${RULE_NAME:-well-status-step-daily}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(45 2 * * ? *)}"

exec "$SCRIPT_DIR/schedule.sh"
