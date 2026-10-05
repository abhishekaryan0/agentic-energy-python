#!/usr/bin/env bash
#
# Schedule STEP V2 pipeline via EventBridge -> Lambda (daily incremental).
#
# Usage (from repo root):
#   ./SOUTH_TEXAS_ENERGY_PARTNERS/v2/deploy/schedule.sh
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME_V2:-south-texas-energy-partners-v2}"
RULE_NAME="${RULE_NAME_V2:-south-texas-step-v2-daily}"
SCHEDULE_CRON="${SCHEDULE_CRON_V2:-cron(45 0 * * ? *)}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

if [[ "${SKIP_LAMBDA_DEPLOY:-0}" != "1" ]]; then
  "$SCRIPT_DIR/deploy.sh"
fi

LAMBDA_ARN="arn:aws:lambda:${LAMBDA_REGION}:${ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}"

echo ""
echo "=== EventBridge schedule (V2) ==="
aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$SCHEDULE_CRON" \
  --state ENABLED \
  --region "$LAMBDA_REGION" \
  --description "Daily STEP V2 synthetic data generation (Lambda): Snowflake *_V2"

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
    "Id": "SouthTexasStepV2Lambda",
    "Arn": "${LAMBDA_ARN}",
    "Input": "{}"
  }
]
EOF
)"

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$LAMBDA_REGION" >/dev/null

echo "Schedule: $SCHEDULE_CRON (rule: $RULE_NAME) -> Lambda $LAMBDA_FUNCTION_NAME"
echo "Logs: /aws/lambda/$LAMBDA_FUNCTION_NAME"
echo ""
echo "First-time full load (all wells):"
echo "  ./SOUTH_TEXAS_ENERGY_PARTNERS/v2/deploy/run_full_backfill.sh"
