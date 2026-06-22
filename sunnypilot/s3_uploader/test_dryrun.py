#!/usr/bin/env python3
"""Off-device dry-run: directory scan, oldest-first selection, rlog-only selector,
xattr dedup, and SigV4 URL build. No real S3 (FAKEUPLOAD=1)."""
import os
import tempfile

os.environ["FAKEUPLOAD"] = "1"
os.environ["FORCEWIFI"] = "1"

from openpilot.sunnypilot.s3_uploader.config import S3Config
from openpilot.sunnypilot.s3_uploader import uploader as up


def _seg(root, name, locked=False, rlog=True):
  d = os.path.join(root, name)
  os.makedirs(d)
  if rlog:
    with open(os.path.join(d, "rlog.zst"), "wb") as f:
      f.write(b"\x28\xb5\x2f\xfd" + b"x" * 100)  # fake zstd-ish bytes
  # qlog/qcamera should be ignored by the rlog-only selector
  with open(os.path.join(d, "qlog.zst"), "wb") as f:
    f.write(b"q" * 10)
  if locked:
    open(os.path.join(d, "rlog.lock"), "w").close()


def main():
  with tempfile.TemporaryDirectory() as root:
    _seg(root, "00000001--aaaaaaaaaaaa--0")          # oldest, eligible
    _seg(root, "00000002--bbbbbbbbbbbb--0")          # eligible
    _seg(root, "00000003--cccccccccccc--0", locked=True)   # locked -> skip
    _seg(root, "00000004--dddddddddddd--0", rlog=False)    # no rlog -> skip

    cfg = S3Config(bucket="my-rlogs", region="us-east-1", access_key="AK", secret_key="SK", prefix="rlogs")
    u = up.Uploader(cfg, dongle_id="dead1234", root=root)

    files = list(u.list_upload_files())
    keys = [k for k, _ in files]
    print("eligible:", keys)
    assert all(k.endswith("/rlog.zst") for k in keys), "selector must only return rlog.zst"
    assert len(files) == 2, f"expected 2 eligible (locked+no-rlog skipped), got {len(files)}"
    assert keys[0] == "rlogs/dead1234/00000001--aaaaaaaaaaaa--0/rlog.zst", "must be oldest-first w/ dongle+seg key"

    # SigV4 URL/header build for the chosen object
    key, fn = u.next_file_to_upload()
    from openpilot.sunnypilot.s3_uploader import sigv4
    url, headers = sigv4.sign_put(cfg.access_key, cfg.secret_key, cfg.region, cfg.bucket, key)
    print("url:", url)
    assert url == f"https://my-rlogs.s3.us-east-1.amazonaws.com/{key}"
    assert headers["x-amz-content-sha256"] == "UNSIGNED-PAYLOAD"
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AK/")

    # fake upload marks xattr; second pass skips it (dedup) and advances to next
    assert u.step(network_type=0) is True
    key2, _ = u.next_file_to_upload()
    print("after 1st upload, next:", key2)
    assert key2 == "rlogs/dead1234/00000002--bbbbbbbbbbbb--0/rlog.zst", "first file should be deduped"

    # upload the rest, then nothing left
    assert u.step(network_type=0) is True
    assert u.next_file_to_upload() is None
    print("dry-run OK: rlog-only, oldest-first, lock-skip, dedup, SigV4 all verified")


if __name__ == "__main__":
  main()
