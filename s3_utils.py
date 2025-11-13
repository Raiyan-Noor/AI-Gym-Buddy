import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


load_dotenv()


class S3ClientWrapper:
    """Thin wrapper around boto3 for upload/presign helpers."""

    def __init__(self) -> None:
        bucket = os.environ.get("AWS_S3_BUCKET")
        if not bucket:
            raise RuntimeError("Set the AWS_S3_BUCKET environment variable (see .env).")
        region = os.environ.get("AWS_REGION", "us-east-1")
        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))

    def upload_file(self, local_path: str, key: str) -> None:
        try:
            self.client.upload_file(local_path, self.bucket, key)
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to upload {local_path} to s3://{self.bucket}/{key}: {exc}") from exc

    def presign_url(self, key: str, expires_in: int = 3600) -> str:
        try:
            return self.client.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_in
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to presign URL for {key}: {exc}") from exc
