#!/usr/bin/env bash
#
# Launch SageMaker anomaly review training for Apex Permian.
#
# Usage:
#   ./anomaly_review/deploy/run_sagemaker_training_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
if [[ -z "${TRAINING_JOB_NAME:-}" ]]; then
  export TRAINING_JOB_NAME="review-train-apex-$(date +%Y%m%d-%H%M)"
fi

exec "$SCRIPT_DIR/run_sagemaker_training.sh"
