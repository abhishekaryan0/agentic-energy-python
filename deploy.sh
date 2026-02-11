#!/usr/bin/env bash
set -euo pipefail

# Simple deploy script for ECR -> ECS workflow
# Usage: REGION=us-east-1 ACCOUNT_ID=200937443240 ./deploy.sh

REGION=${REGION:-us-east-1}
ACCOUNT_ID=${ACCOUNT_ID:-200937443240}
REPO_NAME=${REPO_NAME:-at-risk-assets}
IMAGE_TAG=${IMAGE_TAG:-latest}
CLUSTER=${CLUSTER:-at-risk-assets9}
SERVICE=${SERVICE:-at-risk-assets-service-r8iju4u2}
TASK_DEF_FILE=${TASK_DEF_FILE:-deploy/ecs/task-definition.json}
LOG_GROUP=${LOG_GROUP:-/ecs/at-risk-assets}

echo "1) Authenticate to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "2) Build Docker image..."
docker build -f deploy/Dockerfile -t "$REPO_NAME:$IMAGE_TAG" .

echo "3) Tag image for ECR..."
docker tag "$REPO_NAME:$IMAGE_TAG" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

echo "4) Push to ECR..."
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

echo "5) Register new task definition revision..."
aws ecs register-task-definition --cli-input-json file://"$TASK_DEF_FILE" --region "$REGION"

echo "6) Force ECS service redeploy..."
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --force-new-deployment --region "$REGION"

echo "7) Tail deployment logs (CTRL+C to exit)..."
aws logs tail "$LOG_GROUP" --region "$REGION" --follow

echo "Done."
