"""
Example Lambda handler: call SageMaker anomaly endpoint and return the result.

Use this as the Lambda function code (or adapt). Ensure the Lambda execution role
has sagemaker:InvokeEndpoint on your endpoint.

Environment variables (set in Lambda config):
  SAGEMAKER_ENDPOINT_NAME - name of the anomaly detection endpoint
"""

import json
import os


def lambda_handler(event, context):
    """
    event can be:
      - Direct invoke: { "readings": {...}, "lift_type": "ESP", "well_id": "...", "historical_data": [] }
      - API Gateway: event["body"] is the JSON string of the above
    """
    import boto3

    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint_name:
        return {"statusCode": 500, "body": json.dumps({"error": "SAGEMAKER_ENDPOINT_NAME not set"})}

    # Parse body (API Gateway sends body as string)
    if "body" in event and isinstance(event["body"], str):
        try:
            body = json.loads(event["body"])
        except json.JSONDecodeError as e:
            return {"statusCode": 400, "body": json.dumps({"error": f"Invalid JSON: {e}"})}
    else:
        body = event

    readings = body.get("readings")
    lift_type = body.get("lift_type")
    if not readings or not lift_type:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "readings and lift_type are required"}),
        }

    payload = {
        "readings": readings,
        "lift_type": lift_type,
        "well_id": body.get("well_id"),
        "historical_data": body.get("historical_data", []),
    }

    try:
        client = boto3.client("sagemaker-runtime")
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        return {"statusCode": 200, "body": json.dumps(result)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
