#!/usr/bin/env bash
#
# One-off S3 -> PostgreSQL sync for South Texas Energy Partners only.
#
# Usage:
#   ./s3_to_postgres/deploy/run_sync_step.sh
#   ./s3_to_postgres/deploy/run_sync_step.sh backfill daily_production
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

STEP_ORG="${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}"
LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-s3-to-postgres-sync}"
SYNC_MODE="${1:-incremental}"
TABLE="${2:-all}"
OUT="/tmp/s3_to_pg_step_${TABLE}.json"

PAYLOAD=$(cat <<EOF
{"sync_mode":"${SYNC_MODE}","table":"${TABLE}","organization_id":"${STEP_ORG}"}
EOF
)

echo "Invoking ${LAMBDA_FUNCTION_NAME} org=${STEP_ORG} mode=${SYNC_MODE} table=${TABLE}"
aws lambda invoke \
  --function-name "$LAMBDA_FUNCTION_NAME" \
  --region "$LAMBDA_REGION" \
  --cli-binary-format raw-in-base64-out \
  --payload "$PAYLOAD" \
  "$OUT" >/dev/null

echo "Response written to $OUT"
python3 -m json.tool "$OUT" 2>/dev/null || cat "$OUT"
