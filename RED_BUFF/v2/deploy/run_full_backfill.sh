#!/usr/bin/env bash
#
# Full RED_BUFF V2 backfill into Snowflake via Lambda (batched wells).
# Lambda max timeout is 15m — process WELL_LIMIT wells per invoke.
#
# Usage (from repo root):
#   ./RED_BUFF/v2/deploy/run_full_backfill.sh
#   RESET=0 ./RED_BUFF/v2/deploy/run_full_backfill.sh   # resume without wipe
#   WELL_LIMIT=5 ./RED_BUFF/v2/deploy/run_full_backfill.sh
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
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME_RED_BUFF_V2:-red-buff-v2}"
WELL_LIMIT="${WELL_LIMIT:-10}"
RESET="${RESET:-1}"
TOTAL_WELLS="${TOTAL_WELLS:-100}"
OUT_DIR="${OUT_DIR:-/tmp/red_buff_v2_backfill}"
mkdir -p "$OUT_DIR"

if [[ "${SKIP_LAMBDA_DEPLOY:-0}" != "1" ]]; then
  "$SCRIPT_DIR/deploy.sh"
fi

echo "Full V2 backfill via Lambda: $LAMBDA_FUNCTION_NAME"
echo "  well_limit=$WELL_LIMIT  reset=$RESET  total=$TOTAL_WELLS"

offset=0
first=1
while [[ "$offset" -lt "$TOTAL_WELLS" ]]; do
  payload_file="$(mktemp)"
  if [[ "$first" -eq 1 && "$RESET" == "1" ]]; then
    python3 - <<PY >"$payload_file"
import json
print(json.dumps({
  "reset": True,
  "force_full_backfill": True,
  "well_offset": $offset,
  "well_limit": $WELL_LIMIT,
}))
PY
  else
    python3 - <<PY >"$payload_file"
import json
print(json.dumps({
  "force_full_backfill": True,
  "well_offset": $offset,
  "well_limit": $WELL_LIMIT,
}))
PY
  fi
  first=0

  out_file="$OUT_DIR/batch_${offset}.json"
  echo ""
  echo "=== Invoke offset=$offset limit=$WELL_LIMIT ==="
  aws lambda invoke \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --region "$LAMBDA_REGION" \
    --cli-binary-format raw-in-base64-out \
    --cli-read-timeout 960 \
    --payload "file://${payload_file}" \
    "$out_file" >/dev/null
  rm -f "$payload_file"

  python3 - <<PY
import json, sys
path = "$out_file"
raw = json.load(open(path))
body = raw.get("body", raw)
if isinstance(body, str):
    body = json.loads(body)
print(json.dumps(body, indent=2))
status = body.get("status") or raw.get("statusCode")
if status not in ("success", "skipped", 200):
    print("Batch failed:", body, file=sys.stderr)
    sys.exit(1)
nxt = body.get("next_well_offset")
open("$OUT_DIR/next_offset.txt", "w").write("" if nxt is None else str(nxt))
PY

  next_offset="$(cat "$OUT_DIR/next_offset.txt")"
  if [[ -z "$next_offset" ]]; then
    echo ""
    echo "Full V2 backfill complete."
    exit 0
  fi
  offset="$next_offset"
done

echo "Full V2 backfill complete (processed through offset loop)."
