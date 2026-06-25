#!/usr/bin/env bash
#
# Schedule Snowflake -> S3 transformation export via EventBridge -> Lambda.
# EventBridge rule transformations-daily -> Lambda (daily incremental export).
#
# Usage (from repo root):
#   export S3_BUCKET=...
#   export ORGANIZATION_IDS=...   # or RED_BUFF_/APEX_PERMIAN_ORGANIZATION_ID in .env
#   # Snowflake vars in .env
#   ./transformations/deploy/schedule.sh
#
# Optional:
#   SCHEDULE_CRON="cron(15 0 * * ? *)"      # default: daily at 00:15 UTC
#   LAMBDA_REGION=us-east-1
#   LAMBDA_FUNCTION_NAME=transformations
#   RULE_NAME=transformations-daily
#   SKIP_LAMBDA_DEPLOY=1
#   SKIP_ECR_BUILD=1
#
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-transformations}"
RULE_NAME="${RULE_NAME:-transformations-daily}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(15 0 * * ? *)}"
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
  --description "Snowflake to S3 parquet incremental export (Lambda) once daily (UTC)"

for LEGACY_RULE in transformations-every-2h transformations-every-10m transformations-every-20m; do
  if aws events describe-rule --name "$LEGACY_RULE" --region "$LAMBDA_REGION" &>/dev/null; then
    aws events disable-rule --name "$LEGACY_RULE" --region "$LAMBDA_REGION"
    echo "Disabled legacy rule: $LEGACY_RULE"
  fi
done

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
    "Id": "TransformationsLambda",
    "Arn": "${LAMBDA_ARN}",
    "Input": "{\"mode\":\"incremental\"}"
  }
]
EOF
)"

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$LAMBDA_REGION" >/dev/null

echo "Schedule: $SCHEDULE_CRON (rule: $RULE_NAME) -> Lambda $LAMBDA_FUNCTION_NAME"
echo ""
echo "Done. Transformations export runs on schedule: $SCHEDULE_CRON"
echo "One-time full backfill (local or manual invoke):"
echo "  python3 -m transformations.run --mode full"
echo "  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $LAMBDA_REGION \\"
echo "    --cli-binary-format raw-in-base64-out --payload '{\"mode\":\"full\"}' /tmp/transformations_out.json"
echo "Logs: CloudWatch /aws/lambda/$LAMBDA_FUNCTION_NAME"
