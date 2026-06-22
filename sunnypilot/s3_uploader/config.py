#!/usr/bin/env python3
"""On-device config + credential loading for the S3 rlog uploader.

We deliberately read from a perms-restricted JSON file on /data rather than the params
DB: params keys live in C++ (common/params_keys.h) and adding one forces a scons rebuild,
which would break the "push Python only" deploy. A file also keeps the secret out of the
params store and out of logs.
"""
import json
import os
from dataclasses import dataclass

from openpilot.common.swaglog import cloudlog

CONFIG_PATH = os.getenv("S3_UPLOADER_CONFIG", "/data/s3_uploader.json")

REQUIRED_KEYS = ("bucket", "region", "access_key", "secret_key")


@dataclass
class S3Config:
  bucket: str
  region: str
  access_key: str
  secret_key: str
  prefix: str = "rlogs"
  enabled: bool = True


def load_config(path: str = CONFIG_PATH) -> S3Config | None:
  """Return an S3Config, or None if the file is missing/disabled/malformed.

  Returning None makes the daemon a harmless no-op so it can ship before creds exist.
  """
  if not os.path.exists(path):
    return None

  try:
    with open(path) as f:
      data = json.load(f)
  except (OSError, ValueError):
    cloudlog.exception("s3_uploader: failed to read config")
    return None

  if not data.get("enabled", True):
    return None

  missing = [k for k in REQUIRED_KEYS if not data.get(k)]
  if missing:
    cloudlog.error(f"s3_uploader: config missing keys: {missing}")
    return None

  return S3Config(
    bucket=data["bucket"],
    region=data["region"],
    access_key=data["access_key"],
    secret_key=data["secret_key"],
    prefix=data.get("prefix", "rlogs").strip("/"),
    enabled=True,
  )
