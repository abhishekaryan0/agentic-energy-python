#!/usr/bin/env bash
#
# Schedule S3 -> PostgreSQL sync via EventBridge -> Lambda.
#
# Default: every 24 hours at 01:00 UTC (after anomaly detection at 00:30 UTC).
# SCHEDULE_CRON="rate(24 hours)"  # alternative: every 24h from rule creation
#
# Usage (from repo root):
#   export S3_BUCKET=...
#   export ORGANIZATION_ID=...
#   export DATABASE_URL=...                    # or DATABASE_URL_SECRET_ARN
#   export LAMBDA_SUBNETS=subnet-aaa,subnet-bbb
#   export LAMBDA_SECURITY_GROUPS=sg-xxx
#   ./s3_to_postgres/deploy/schedule.sh
#
# Optional:
#   SCHEDULE_CRON="cron(0 1 * * ? *)"
#   LAMBDA_REGION=us-east-1
#   LAMBDA_FUNCTION_NAME=s3-to-postgres-sync
#   RULE_NAME=s3-to-postgres-anomaly-suggestions-daily
#   SKIP_LAMBDA_DEPLOY=1
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

_SAVED_DATABASE_URL="${DATABASE_URL:-}"
_SAVED_DATABASE_URL_SECRET_ARN="${DATABASE_URL_SECRET_ARN:-}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -n "$_SAVED_DATABASE_URL" ]]; then
  export DATABASE_URL="$_SAVED_DATABASE_URL"
fi
if [[ -n "$_SAVED_DATABASE_URL_SECRET_ARN" ]]; then
  export DATABASE_URL_SECRET_ARN="$_SAVED_DATABASE_URL_SECRET_ARN"
fi

LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-s3-to-postgres-sync}"
RULE_NAME="${RULE_NAME:-s3-to-postgres-sync-daily}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 1 * * ? *)}"
ORGANIZATION_ID="${ORGANIZATION_ID:?Set ORGANIZATION_ID}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

if [[ "${SKIP_LAMBDA_DEPLOY:-0}" != "1" ]]; then
  "$SCRIPT_DIR/deploy.sh"
fi

LAMBDA_ARN="arn:aws:lambda:${LAMBDA_REGION}:${ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}"

echo ""
echo "=== EventBridge schedule ==="
aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$SCHEDULE_CRON" \
  --state ENABLED \
  --region "$LAMBDA_REGION" \
  --description "Every 24h: sync detection parquet and daily production to PostgreSQL"

STATEMENT_ID="AllowEventBridgeInvoke-${RULE_NAME}"
aws lambda remove-permission \
  --function-name "$LAMBDA_FUNCTION_NAME" \
  --statement-id "$STATEMENT_ID" \
  --region "$LAMBDA_REGION" 2>/dev/null || true

aws lambda add-permission \
  --function-name "$LAMBDA_FUNCTION_NAME" \
  --statement-id "$STATEMENT_ID" \
  --action "lambda:InvokeFunction" \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${LAMBDA_REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
  --region "$LAMBDA_REGION" >/dev/null

TARGET_JSON="$(cat <<EOF
[
  {
    "Id": "S3ToPostgresSync",
    "Arn": "${LAMBDA_ARN}",
    "Input": "{\"sync_mode\":\"incremental\",\"table\":\"all\"}"
  }
]
EOF
)"

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$LAMBDA_REGION" >/dev/null

echo "Schedule: $SCHEDULE_CRON"
echo "Rule    : $RULE_NAME"
echo "Target  : $LAMBDA_FUNCTION_NAME"
echo ""
echo "Manual backfill:"
echo "  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $LAMBDA_REGION \\"
echo "    --cli-binary-format raw-in-base64-out \\"
echo "    --payload '{\"sync_mode\":\"backfill\",\"table\":\"anomaly_suggestions\"}' /tmp/s3_to_pg_out.json"
