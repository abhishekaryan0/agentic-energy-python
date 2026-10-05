#!/usr/bin/env bash
#
# Schedule V2 Snowflake → S3 export under {ORG}/v2/ (does not touch v1 paths).
# Default orgs: Red Buff + STEP.
#
# Usage (from repo root):
#   ./transformations/deploy/schedule_v2.sh
#
# Optional:
#   SCHEDULE_CRON="cron(50 0 * * ? *)"   # default 00:50 UTC (after v1 00:45)
#   RULE_NAME=transformations-v2-daily
#   SKIP_LAMBDA_DEPLOY=1
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
RULE_NAME="${RULE_NAME:-transformations-v2-daily}"
SCHEDULE_CRON="${SCHEDULE_CRON:-cron(50 0 * * ? *)}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

RED_ORG="${RED_BUFF_ORGANIZATION_ID:-3e0b5b18-bec1-488a-82c1-a47881c69058}"
STEP_ORG="${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}"

if [[ "${SKIP_LAMBDA_DEPLOY:-0}" != "1" ]]; then
  "$SCRIPT_DIR/deploy.sh"
fi

LAMBDA_ARN="arn:aws:lambda:${LAMBDA_REGION}:${ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}"

echo ""
echo "=== EventBridge V2 schedule (writes under {org}/v2/ only) ==="
aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$SCHEDULE_CRON" \
  --state ENABLED \
  --region "$LAMBDA_REGION" \
  --description "Snowflake *_V2 to S3 under {org}/v2/sensor_data (Red Buff + STEP)"

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
    "Id": "TransformationsV2Lambda",
    "Arn": "${LAMBDA_ARN}",
    "Input": "{\"mode\":\"incremental\",\"version\":\"v2\",\"organization_ids\":[\"${RED_ORG}\",\"${STEP_ORG}\"]}"
  }
]
EOF
)"

aws events put-targets --rule "$RULE_NAME" --targets "$TARGET_JSON" --region "$LAMBDA_REGION" >/dev/null

echo "Schedule: $SCHEDULE_CRON (rule: $RULE_NAME) -> Lambda $LAMBDA_FUNCTION_NAME"
echo "S3 layout (v1 unchanged):"
echo "  s3://\$S3_BUCKET/{org}/sensor_data/          # v1"
echo "  s3://\$S3_BUCKET/{org}/v2/sensor_data/       # v2 (new)"
echo ""
echo "Full V2 backfill:"
echo "  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $LAMBDA_REGION \\"
echo "    --cli-binary-format raw-in-base64-out \\"
echo "    --payload '{\"mode\":\"full\",\"version\":\"v2\"}' /tmp/transformations_v2_out.json"
echo "  # synth per org:"
echo "  aws lambda invoke ... --payload '{\"mode\":\"full\",\"version\":\"v2\",\"synth\":true,\"organization_id\":\"${RED_ORG}\"}' ..."
