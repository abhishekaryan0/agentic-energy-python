#!/usr/bin/env bash
#
# Schedule well status update via EventBridge -> Lambda (daily).
#
# Usage (from repo root):
#   ./well_status/deploy/schedule.sh
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

_SAVED_ORGANIZATION_ID="${ORGANIZATION_ID:-}"
_SAVED_LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-}"
_SAVED_LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-}"
_SAVED_RULE_NAME="${RULE_NAME:-}"
_SAVED_SCHEDULE_CRON="${SCHEDULE_CRON:-}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -n "$_SAVED_ORGANIZATION_ID" ]]; then
  export ORGANIZATION_ID="$_SAVED_ORGANIZATION_ID"
fi
if [[ -n "$_SAVED_LAMBDA_FUNCTION_NAME" ]]; then
  export LAMBDA_FUNCTION_NAME="$_SAVED_LAMBDA_FUNCTION_NAME"
fi
if [[ -n "$_SAVED_LAMBDA_ROLE_NAME" ]]; then
  export LAMBDA_ROLE_NAME="$_SAVED_LAMBDA_ROLE_NAME"
fi
if [[ -n "$_SAVED_RULE_NAME" ]]; then
  export RULE_NAME="$_SAVED_RULE_NAME"
fi
if [[ -n "$_SAVED_SCHEDULE_CRON" ]]; then
  export SCHEDULE_CRON="$_SAVED_SCHEDULE_CRON"
fi

LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-well-status}"
RULE_NAME="${RULE_NAME:-well-status-daily}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(0 2 * * ? *)}"
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
  --description "Daily well status update (Lambda): S3 sensor -> PostgreSQL"

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
    "Id": "WellStatusLambda",
    "Arn": "${LAMBDA_ARN}",
    "Input": "{}"
  }
]
EOF
)"

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$LAMBDA_REGION" >/dev/null
aws events remove-targets --rule "$RULE_NAME" --ids WellStatusTask --region "$LAMBDA_REGION" 2>/dev/null || true

echo "Schedule: $SCHEDULE_CRON (rule: $RULE_NAME) -> Lambda $LAMBDA_FUNCTION_NAME"
echo "Logs: /aws/lambda/$LAMBDA_FUNCTION_NAME"
