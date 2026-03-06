#!/usr/bin/env bash
# Check SageMaker endpoint status (one per line, no concatenation).
# Usage: ./deploy/check_endpoints.sh [region]
set -e
REGION="${1:-us-east-1}"
for name in anomaly-detection-endpoint anomaly-operations-endpoint anomaly-review-endpoint; do
  status=$(aws sagemaker describe-endpoint --endpoint-name "$name" --region "$REGION" --query 'EndpointStatus' --output text 2>/dev/null || echo "not found")
  printf "%s: %s\n" "$name" "$status"
done
