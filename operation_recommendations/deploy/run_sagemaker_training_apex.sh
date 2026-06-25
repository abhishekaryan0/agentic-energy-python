#!/usr/bin/env bash
#
# Launch SageMaker operations training for Apex Permian.
#
# Usage:
#   ./operation_recommendations/deploy/run_sagemaker_training_apex.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORGANIZATION_ID="${APEX_PERMIAN_ORGANIZATION_ID:-147a92b9-f90a-473d-b443-e6e668c4fbb7}"
export TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-operations-train-apex-$(date +%Y%m%d-%H%M)}"

exec "$SCRIPT_DIR/run_sagemaker_training.sh"
