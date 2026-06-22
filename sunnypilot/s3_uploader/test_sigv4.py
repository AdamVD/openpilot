#!/usr/bin/env python3
"""Off-device check: our hand-rolled SigV4 must byte-match botocore's for the same inputs.

Run with botocore importable, e.g.:
  PYTHONPATH=/home/adam/comma/.awsdeps python3 sunnypilot/s3_uploader/test_sigv4.py
"""
import datetime

from openpilot.sunnypilot.s3_uploader.sigv4 import sign_put


def _botocore_auth(ak, sk, region, bucket, key, now):
  from botocore.auth import S3SigV4Auth
  from botocore.awsrequest import AWSRequest
  from botocore.credentials import Credentials

  url, _ = sign_put(ak, sk, region, bucket, key, now=now)
  auth = S3SigV4Auth(Credentials(ak, sk), "s3", region)
  auth.payload = lambda request: "UNSIGNED-PAYLOAD"  # force unsigned, streamed body

  # Build the request and force botocore to use our frozen timestamp. add_auth()
  # restamps with wall-clock time, so drive the signing primitives directly.
  req = AWSRequest(method="PUT", url=url, headers={"x-amz-content-sha256": "UNSIGNED-PAYLOAD"})
  req.context["timestamp"] = now.strftime("%Y%m%dT%H%M%SZ")
  auth._modify_request_before_signing(req)
  cr = auth.canonical_request(req)
  sts = auth.string_to_sign(req, cr)
  signature = auth.signature(sts, req)
  scope = f"{now:%Y%m%d}/{region}/s3/aws4_request"
  sh = "host;x-amz-content-sha256;x-amz-date"
  return f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={sh}, Signature={signature}"


def main():
  ak, sk = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
  now = datetime.datetime(2026, 6, 21, 12, 0, 0, tzinfo=datetime.UTC)
  cases = [
    ("us-east-1", "my-rlog-bucket", "rlogs/abc123/00000001--a3c20ba54385--0/rlog.zst"),
    ("us-west-2", "my.rlog.bucket", "rlogs/dead-beef/12345678--ffffffffffff--12/rlog.zst"),
    ("eu-central-1", "bucket", "p/d/seg/rlog.zst"),
  ]
  for region, bucket, key in cases:
    _, headers = sign_put(ak, sk, region, bucket, key, now=now)
    mine = headers["Authorization"]
    theirs = _botocore_auth(ak, sk, region, bucket, key, now)
    status = "OK" if mine == theirs else "MISMATCH"
    print(f"[{status}] {region} {bucket} {key}")
    if mine != theirs:
      print("  mine:  ", mine)
      print("  theirs:", theirs)
      raise SystemExit(1)
  print("all SigV4 cases match botocore")


if __name__ == "__main__":
  main()
