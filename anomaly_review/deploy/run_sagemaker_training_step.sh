#!/usr/bin/env bash
#
# Launch SageMaker anomaly review training for South Texas Energy Partners.
#
# Usage:
#   ./anomaly_review/deploy/run_sagemaker_training_step.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export ORGANIZATION_ID="${STEP_ORGANIZATION_ID:-d270f89a-aada-4ff2-adb0-68b3e8087565}"
if [[ "${TRAINING_JOB_NAME:-}" != review-train* ]]; then
  export TRAINING_JOB_NAME="review-train-step-$(date +%Y%m%d-%H%M)"
fi

"$SCRIPT_DIR/run_sagemaker_training.sh"

echo ""
echo "Model will appear under s3://\${S3_BUCKET}/\${ORGANIZATION_ID}/model/${TRAINING_JOB_NAME}/output/model.tar.gz"
echo "Detection auto-picks the latest review-train-* tarball for the org (no .env job name needed)."
