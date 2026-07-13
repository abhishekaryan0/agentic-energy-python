#!/usr/bin/env bash
#
# Schedule Apex Permian well status via EventBridge -> Lambda (daily).
#
# Usage (from repo root):
#   ./well_status/deploy/schedule_apex.sh
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

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-well-status-apex-permian}"
export LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-well-status-apex-permian-lambda-role}"
export RULE_NAME="${RULE_NAME:-well-status-apex-permian-daily}"
export SCHEDULE_CRON="${SCHEDULE_CRON:-cron(15 2 * * ? *)}"

exec "$SCRIPT_DIR/schedule.sh"
