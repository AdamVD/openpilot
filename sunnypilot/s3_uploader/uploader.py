#!/usr/bin/env python3
"""Auto-upload 100% of rlogs directly to our own S3 bucket.

Runs in parallel with comma's uploader (untouched) and signs each PUT on-device with
SigV4 so logs go device -> S3 directly, never through api.commadotai.com. Scope is
rlog-only, wifi/ethernet-only (cellular is metered). See the plan/memory for the
deleter keep-up caveat: capture is gated by parked-on-wifi time vs rlog bytes.

Adapted from sunnypilot/sunnylink/uploader.py.
"""
import os
import random
import threading
import time
import traceback
from collections.abc import Iterator

import requests

from cereal import log
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.loggerd.xattr_cache import getxattr, setxattr
from openpilot.sunnypilot.s3_uploader import sigv4
from openpilot.sunnypilot.s3_uploader.config import S3Config, load_config

NetworkType = log.DeviceState.NetworkType

# Independent of comma's 'user.upload' and sunnylink's 'user.sunny.upload' so the three
# uploaders never interfere with each other's dedup.
UPLOAD_ATTR_NAME = 'user.s3.upload'
UPLOAD_ATTR_VALUE = b'1'

UPLOAD_NAME = 'rlog.zst'
UPLOAD_TIMEOUT = 60

allow_sleep = bool(os.getenv("UPLOADER_SLEEP", "1"))
force_wifi = os.getenv("FORCEWIFI") is not None
fake_upload = os.getenv("FAKEUPLOAD") is not None


class Uploader:
  def __init__(self, cfg: S3Config, dongle_id: str, root: str):
    self.cfg = cfg
    self.dongle_id = dongle_id
    self.root = root
    self.last_filename = ""

  def list_upload_files(self) -> Iterator[tuple[str, str]]:
    for logdir in listdir_by_creation(self.root):
      path = os.path.join(self.root, logdir)
      try:
        names = os.listdir(path)
      except OSError:
        continue

      # skip routes that are still being written
      if any(name.endswith(".lock") for name in names):
        continue

      if UPLOAD_NAME not in names:
        continue

      fn = os.path.join(path, UPLOAD_NAME)
      try:
        is_uploaded = getxattr(fn, UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE
      except OSError:
        # deleter could have removed it mid-scan
        continue
      if is_uploaded:
        continue

      # S3 object key: prefix/dongle/<route>--<seg>/rlog.zst
      key = "/".join((self.cfg.prefix, self.dongle_id, logdir, UPLOAD_NAME))
      yield key, fn

  def next_file_to_upload(self) -> tuple[str, str] | None:
    # oldest-first so we clear the backlog before the deleter reaches it
    return next(self.list_upload_files(), None)

  def do_upload(self, key: str, fn: str) -> requests.Response:
    url, headers = sigv4.sign_put(self.cfg.access_key, self.cfg.secret_key,
                                  self.cfg.region, self.cfg.bucket, key)
    # rlogs are already zstd-compressed on disk and comfortably fit in memory, so PUT the
    # raw bytes: this guarantees a correct Content-Length and avoids requests falling back
    # to chunked transfer-encoding, which S3 rejects (501) for a plain PUT.
    with open(fn, "rb") as f:
      data = f.read()
    return requests.put(url, data=data, headers=headers, timeout=UPLOAD_TIMEOUT)

  def upload(self, key: str, fn: str, network_type: int) -> bool:
    try:
      sz = os.path.getsize(fn)
    except OSError:
      cloudlog.exception("s3_uploader: getsize failed")
      return False

    cloudlog.event("s3_upload_start", key=key, fn=fn, sz=sz, network_type=network_type)

    if sz == 0:
      success = True  # tag empty files as done
    elif fake_upload:
      success = True
    else:
      start_time = time.monotonic()
      stat = None
      last_exc = None
      try:
        stat = self.do_upload(key, fn)
      except Exception as e:
        last_exc = (e, traceback.format_exc())

      if stat is not None and stat.status_code in (200, 201):
        self.last_filename = fn
        dt = time.monotonic() - start_time
        speed = (sz / 1e6) / dt if dt > 0 else 0
        cloudlog.event("s3_upload_success", key=key, fn=fn, sz=sz, network_type=network_type, speed=speed)
        success = True
      else:
        success = False
        code = stat.status_code if stat is not None else None
        body = stat.content.decode("utf-8", "replace")[:512] if stat is not None else None
        cloudlog.event("s3_upload_failed", code=code, body=body, exc=last_exc, key=key, fn=fn, sz=sz)

    if success:
      try:
        setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
      except OSError:
        cloudlog.event("s3_uploader_setxattr_failed", key=key, fn=fn)

    return success

  def step(self, network_type: int) -> bool | None:
    d = self.next_file_to_upload()
    if d is None:
      return None
    key, fn = d
    return self.upload(key, fn, network_type)


def main(exit_event: threading.Event | None = None) -> None:
  if exit_event is None:
    exit_event = threading.Event()

  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("s3_uploader: failed to set core affinity")

  params = Params()
  cfg = load_config()
  if cfg is None:
    cloudlog.info("s3_uploader: no/disabled config, exiting")
    return

  dongle_id = params.get("DongleId")
  if dongle_id is None:
    cloudlog.info("s3_uploader: missing dongle id, exiting")
    return

  sm = messaging.SubMaster(['deviceState'])
  uploader = Uploader(cfg, dongle_id, Paths.log_root())

  backoff = 0.1
  while not exit_event.is_set():
    sm.update(0)
    offroad = params.get_bool("IsOffroad")
    network_type = sm['deviceState'].networkType if not force_wifi else NetworkType.wifi
    metered = sm['deviceState'].networkMetered and not force_wifi

    # wifi/ethernet only: skip no-connection and metered (incl. comma prime LTE)
    if network_type == NetworkType.none or metered:
      if allow_sleep:
        time.sleep(60 if offroad else 5)
      continue

    success = uploader.step(network_type.raw)
    if success is None:
      backoff = 60 if offroad else 5
    elif success:
      backoff = 0.1
    else:
      backoff = min(backoff * 2, 120)
    if allow_sleep:
      time.sleep(backoff + random.uniform(0, backoff))


if __name__ == "__main__":
  main()
