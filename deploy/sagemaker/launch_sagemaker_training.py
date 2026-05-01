"""
Launch SageMaker training jobs for ML models.

Designed to run in a scheduled ECS Fargate task (EventBridge -> ECS RunTask).

Run a single model by setting:
  MODEL_KEY=anomaly_detection|anomaly_review|operations|financial

If MODEL_KEY is unset/empty, training jobs for all models are created.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import boto3


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required env var: {name}")
    return str(v)


def _opt_env(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return str(v)


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M", time.gmtime())


@dataclass(frozen=True)
class TrainingSpec:
    key: str
    job_prefix: str
    ecr_repo_env: str
    ecr_repo_default: str
    instance_type_env: str
    instance_type_default: str = "ml.m5.xlarge"


SPECS: list[TrainingSpec] = [
    TrainingSpec(
        key="anomaly_detection",
        job_prefix="anomaly-train",
        ecr_repo_env="ECR_REPO_TRAIN",
        ecr_repo_default="anomaly-detection-sagemaker-train",
        instance_type_env="INSTANCE_TYPE_ANOMALY_DETECTION",
    ),
    TrainingSpec(
        key="anomaly_review",
        job_prefix="review-train",
        ecr_repo_env="ECR_REPO_TRAIN_REVIEW",
        ecr_repo_default="anomaly-review-sagemaker-train",
        instance_type_env="INSTANCE_TYPE_ANOMALY_REVIEW",
    ),
    TrainingSpec(
        key="operations",
        job_prefix="operations-train",
        ecr_repo_env="ECR_REPO_TRAIN_OPERATIONS",
        ecr_repo_default="operations-sagemaker-train",
        instance_type_env="INSTANCE_TYPE_OPERATIONS",
    ),
    TrainingSpec(
        key="financial",
        job_prefix="financial-train",
        ecr_repo_env="ECR_REPO_TRAIN_FINANCIAL",
        ecr_repo_default="financial-sagemaker-train",
        instance_type_env="INSTANCE_TYPE_FINANCIAL",
    ),
]


def _ecr_uri(account_id: str, region: str, repo: str, tag: str) -> str:
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"


def _channel(name: str, s3_uri_prefix: str) -> dict:
    return {
        "ChannelName": name,
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": s3_uri_prefix if s3_uri_prefix.endswith("/") else (s3_uri_prefix + "/"),
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


def _create_training_job(
    sm,
    *,
    job_name: str,
    role_arn: str,
    image_uri: str,
    instance_type: str,
    model_output_s3: str,
    input_channels: list[dict],
    hyperparameters: dict[str, str],
) -> None:
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={"TrainingImage": image_uri, "TrainingInputMode": "File"},
        InputDataConfig=input_channels,
        OutputDataConfig={"S3OutputPath": model_output_s3 if model_output_s3.endswith("/") else (model_output_s3 + "/")},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={k: str(v) for k, v in hyperparameters.items()},
    )


def main() -> None:
    region = _env("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    account_id = _env("AWS_ACCOUNT_ID")
    s3_bucket = _env("S3_BUCKET")
    org_id = _env("ORGANIZATION_ID")
    image_tag = _opt_env("IMAGE_TAG", "latest")

    sagemaker_role_arn = _env("SAGEMAKER_ROLE_ARN")

    per_well_limit = _opt_env("PER_WELL_LIMIT", "5000")

    sensor_data_s3 = f"s3://{s3_bucket}/{org_id}/sensor_data"
    financial_data_s3 = f"s3://{s3_bucket}/{org_id}/financial_data"
    model_output_s3 = f"s3://{s3_bucket}/{org_id}/model"

    sm = boto3.client("sagemaker", region_name=region)

    model_key = _opt_env("MODEL_KEY", "").strip()
    if model_key:
        specs = [s for s in SPECS if s.key == model_key]
        if not specs:
            valid = ", ".join(s.key for s in SPECS)
            raise ValueError(f"Invalid MODEL_KEY={model_key!r}. Valid: {valid}")
    else:
        specs = SPECS

    created: list[tuple[str, str]] = []
    ts = _ts()

    for spec in specs:
        repo = _opt_env(spec.ecr_repo_env, spec.ecr_repo_default)
        image_uri = _ecr_uri(account_id, region, repo, image_tag)
        instance_type = _opt_env(spec.instance_type_env, spec.instance_type_default)
        job_name = f"{spec.job_prefix}-{ts}"

        channels = [_channel("sensor_data", sensor_data_s3)]
        if spec.key == "financial":
            channels.append(_channel("financial_data", financial_data_s3))

        hyper = {"ORGANIZATION_ID": org_id}
        if spec.key == "anomaly_detection":
            hyper["PER_WELL_LIMIT"] = per_well_limit

        _create_training_job(
            sm,
            job_name=job_name,
            role_arn=sagemaker_role_arn,
            image_uri=image_uri,
            instance_type=instance_type,
            model_output_s3=model_output_s3,
            input_channels=channels,
            hyperparameters=hyper,
        )
        created.append((spec.key, job_name))

    lines = ["Started SageMaker training jobs:"]
    for key, job_name in created:
        lines.append(f"- {key}: {job_name}")
    lines.append(f"Output prefix: {model_output_s3}/<job_name>/output/model.tar.gz")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

