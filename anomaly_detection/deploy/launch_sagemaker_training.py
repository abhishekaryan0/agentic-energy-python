"""Start a SageMaker training job for anomaly_detection (scheduled ECS launcher)."""

from __future__ import annotations

import os
import time

import boto3


def main() -> None:
    region = os.environ["AWS_REGION"]
    account_id = os.environ["AWS_ACCOUNT_ID"]
    bucket = os.environ["S3_BUCKET"]
    org = os.environ["ORGANIZATION_ID"]
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    repo = os.environ.get("ECR_REPO_TRAIN", "anomaly-detection-sagemaker-train")
    tag = os.environ.get("IMAGE_TAG", "latest")
    instance = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge")
    per_well = os.environ.get("PER_WELL_LIMIT", "5000")
    partition_days = os.environ.get("ANOMALY_SENSOR_PARTITION_DAYS", "250")

    image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
    ts = time.strftime("%Y%m%d-%H%M", time.gmtime())
    job_name = os.environ.get("TRAINING_JOB_NAME", f"anomaly-train-{ts}")
    sensor_s3 = f"s3://{bucket}/{org}/sensor_data/well_sensor_readings/"
    model_out = f"s3://{bucket}/{org}/model/"

    sm = boto3.client("sagemaker", region_name=region)
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        InputDataConfig=[
            {
                "ChannelName": "sensor_data",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": sensor_s3,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={"S3OutputPath": model_out},
        ResourceConfig={"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={
            "ORGANIZATION_ID": org,
            "PER_WELL_LIMIT": per_well,
            "ANOMALY_SENSOR_PARTITION_DAYS": partition_days,
        },
    )
    print(f"Started SageMaker training job: {job_name}", flush=True)
    print(f"Output: {model_out}{job_name}/output/model.tar.gz", flush=True)


if __name__ == "__main__":
    main()
