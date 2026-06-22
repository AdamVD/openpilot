#!/usr/bin/env python3
"""Minimal, dependency-free AWS Signature Version 4 signer for a single S3 PUT.

We sign on-device with a scoped IAM key so rlogs go device -> S3 directly, never
touching comma's servers. boto3 is far too heavy for the device, so this implements
just the slice of SigV4 we need: a virtual-hosted-style PUT with an unsigned (streamed)
payload.

Reference: https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-authenticating-requests.html
"""
import datetime
import hashlib
import hmac
import urllib.parse

ALGORITHM = "AWS4-HMAC-SHA256"
SERVICE = "s3"
# Lets us stream the file body without buffering/hashing it first.
UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"


def _sha256_hex(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
  return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret_key: str, datestamp: str, region: str) -> bytes:
  k_date = _hmac(("AWS4" + secret_key).encode("utf-8"), datestamp)
  k_region = _hmac(k_date, region)
  k_service = _hmac(k_region, SERVICE)
  return _hmac(k_service, "aws4_request")


def host_for(bucket: str, region: str) -> str:
  return f"{bucket}.s3.{region}.amazonaws.com"


def sign_put(access_key: str, secret_key: str, region: str, bucket: str, key: str,
             now: datetime.datetime | None = None) -> tuple[str, dict[str, str]]:
  """Return (url, headers) for a virtual-hosted-style S3 PUT with an unsigned payload.

  `key` is the object key without a leading slash, e.g. "rlogs/<dongle>/<seg>/rlog.zst".
  """
  if now is None:
    now = datetime.datetime.now(datetime.UTC)
  amz_date = now.strftime("%Y%m%dT%H%M%SZ")
  datestamp = now.strftime("%Y%m%d")

  host = host_for(bucket, region)
  # Encode each path segment per RFC 3986, leaving the path separators intact.
  canonical_uri = "/" + urllib.parse.quote(key, safe="/~")
  url = f"https://{host}{canonical_uri}"

  # Headers we sign, in the order S3 expects them canonicalized (lowercase, sorted).
  signed = {
    "host": host,
    "x-amz-content-sha256": UNSIGNED_PAYLOAD,
    "x-amz-date": amz_date,
  }
  signed_headers = ";".join(sorted(signed))
  canonical_headers = "".join(f"{k}:{signed[k]}\n" for k in sorted(signed))

  canonical_request = "\n".join([
    "PUT",
    canonical_uri,
    "",  # empty canonical query string
    canonical_headers,
    signed_headers,
    UNSIGNED_PAYLOAD,
  ])

  credential_scope = f"{datestamp}/{region}/{SERVICE}/aws4_request"
  string_to_sign = "\n".join([
    ALGORITHM,
    amz_date,
    credential_scope,
    _sha256_hex(canonical_request.encode("utf-8")),
  ])

  signature = hmac.new(_signing_key(secret_key, datestamp, region),
                       string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

  authorization = f"{ALGORITHM} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

  headers = {
    "x-amz-content-sha256": UNSIGNED_PAYLOAD,
    "x-amz-date": amz_date,
    "Authorization": authorization,
  }
  return url, headers
