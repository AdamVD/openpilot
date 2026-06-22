#!/usr/bin/env python3
"""One-time AWS setup for the rlog bucket. DEV-MACHINE ONLY (uses boto3).

Creates a private, versioned bucket with a cost-capping lifecycle rule and a dedicated
PutObject-only IAM user scoped to <bucket>/<prefix>/*. The resulting access key can only
write to our prefix: a stolen device key can run up storage cost but cannot read, list,
or delete our data (and versioning means it can't even truly overwrite history).

Credentials come from the standard AWS env vars / profile (boto3 picks them up); these
must be your admin creds, NOT the device key this script mints.

Usage:
  PYTHONPATH=/home/adam/comma/.awsdeps \\
    python3 sunnypilot/s3_uploader/aws_setup.py --bucket my-rlogs --region us-east-1 [--prefix rlogs] [--glacier-days 30] [--expire-days 365]
"""
import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError

STATE_PATH = "/home/adam/comma/.s3_uploader_state.json"


def ensure_bucket(s3, bucket: str, region: str) -> None:
  try:
    s3.head_bucket(Bucket=bucket)
    print(f"bucket {bucket} already exists")
  except ClientError:
    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
      kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    print(f"created bucket {bucket}")

  s3.put_public_access_block(
    Bucket=bucket,
    PublicAccessBlockConfiguration={
      "BlockPublicAcls": True, "IgnorePublicAcls": True,
      "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
    },
  )
  s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
  print("public access blocked + versioning enabled")


def ensure_lifecycle(s3, bucket: str, prefix: str, glacier_days: int, expire_days: int) -> None:
  rule: dict = {
    "ID": "rlog-cost-control",
    "Filter": {"Prefix": prefix + "/"},
    "Status": "Enabled",
  }
  if glacier_days > 0:
    rule["Transitions"] = [{"Days": glacier_days, "StorageClass": "GLACIER"}]
  if expire_days > 0:
    rule["Expiration"] = {"Days": expire_days}
    rule["NoncurrentVersionExpiration"] = {"NoncurrentDays": max(1, expire_days)}
  s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": [rule]})
  print(f"lifecycle: glacier@{glacier_days}d expire@{expire_days}d")


def ensure_user(bucket: str, prefix: str) -> tuple[str, str]:
  iam = boto3.client("iam")
  user = f"rlog-uploader-{bucket}"
  try:
    iam.create_user(UserName=user)
    print(f"created IAM user {user}")
  except iam.exceptions.EntityAlreadyExistsException:
    print(f"IAM user {user} already exists")

  policy = {
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*",
    }],
  }
  iam.put_user_policy(UserName=user, PolicyName="put-only", PolicyDocument=json.dumps(policy))
  print("attached PutObject-only policy")

  key = iam.create_access_key(UserName=user)["AccessKey"]
  return key["AccessKeyId"], key["SecretAccessKey"]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--bucket", required=True)
  ap.add_argument("--region", required=True)
  ap.add_argument("--prefix", default="rlogs")
  ap.add_argument("--glacier-days", type=int, default=30)
  ap.add_argument("--expire-days", type=int, default=365)
  args = ap.parse_args()
  prefix = args.prefix.strip("/")

  # Virtual-hosted PUTs use https://<bucket>.s3.<region>.amazonaws.com; the S3 wildcard cert
  # matches only one label, so a dotted bucket name fails TLS validation on every upload.
  if "." in args.bucket:
    sys.exit("bucket name must not contain dots (breaks TLS for virtual-hosted PUTs)")

  s3 = boto3.client("s3", region_name=args.region)
  ensure_bucket(s3, args.bucket, args.region)
  ensure_lifecycle(s3, args.bucket, prefix, args.glacier_days, args.expire_days)
  access_key, secret_key = ensure_user(args.bucket, prefix)

  device_cfg = {
    "enabled": True,
    "bucket": args.bucket,
    "region": args.region,
    "prefix": prefix,
    "access_key": access_key,
    "secret_key": secret_key,
  }
  with open(STATE_PATH, "w") as f:
    json.dump({"bucket": args.bucket, "region": args.region, "prefix": prefix,
               "access_key": access_key}, f, indent=2)

  print("\n=== write this to the device at /data/s3_uploader.json (chmod 600) ===\n")
  print(json.dumps(device_cfg, indent=2))
  print("\n(the secret key is shown only once; rerun mints a new key)", file=sys.stderr)


if __name__ == "__main__":
  main()
