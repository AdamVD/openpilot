# S3 rlog auto-uploader

Auto-uploads **100% of rlogs** straight to our own S3 bucket, in parallel with comma's
uploader (which is left untouched, so Connect keeps working). Each PUT is signed on-device
with AWS SigV4 — logs go **device → S3 directly**, never through `api.commadotai.com`.

- **Scope:** `rlog.zst` only (no qlog, no camera).
- **Network:** wifi/ethernet only. Cellular (incl. the comma prime LTE SIM) is metered and skipped.
- **When:** offroad (parked), oldest-first.

## ⚠ "100%" is gated by keep-up, not code

The device deleter keeps only ~5 GB / 10% free and deletes **oldest-first regardless of
upload status**. Capture = (parked-on-wifi hours × home upstream) vs (rlog bytes generated).
Normal daily driving at home keeps up fine; a multi-day road trip away from wifi will lag and
may lose the tail. Measure before relying on it (step 4).

## Deploy

### 1. AWS setup (dev machine, once)

```bash
cd /home/adam/comma/openpilot
PYTHONPATH=/home/adam/comma/.awsdeps \
  python3 sunnypilot/s3_uploader/aws_setup.py --bucket <name> --region <region>
```

Creates a private, versioned bucket + lifecycle (Glacier@30d, expire@365d) + a dedicated
**PutObject-only** IAM user scoped to `<bucket>/rlogs/*`. Prints the device config JSON
(secret shown once).

### 2. Put creds on the device (once)

SSH to the comma and write the printed JSON to `/data/s3_uploader.json`, then lock it down:

```bash
cat > /data/s3_uploader.json <<'EOF'
{ "enabled": true, "bucket": "...", "region": "...", "prefix": "rlogs",
  "access_key": "...", "secret_key": "..." }
EOF
chmod 600 /data/s3_uploader.json
```

No params/scons rebuild needed — config is a plain file.

### 3. Ship the code

Push this branch; the daemon auto-registers via the `os.path.exists` guard in
`system/manager/process_config.py` and starts offroad after reboot.

### 4. Verify keep-up (on device)

```bash
du -sh /data/media/0/realdata/*/rlog.zst | tail   # typical rlog size
```

Check it clears within a normal parked-at-home window vs your home upstream Mbps.

## Tests (off-device)

```bash
# SigV4 must byte-match botocore:
PYTHONPATH=/home/adam/comma/.awsdeps:/home/adam/comma/openpilot \
  python3 sunnypilot/s3_uploader/test_sigv4.py

# selector / oldest-first / dedup / URL build:
PYTHONPATH=/home/adam/comma/openpilot \
  .venv/bin/python sunnypilot/s3_uploader/test_dryrun.py
```

## Security

The device key can only `s3:PutObject` to `<bucket>/rlogs/*` — no list/get/delete. With bucket
versioning on, a stolen key can run up storage cost but cannot read, erase, or truly overwrite
history. To rotate: rerun `aws_setup.py` (mints a new key) and update `/data/s3_uploader.json`.

## Troubleshooting (on device)

- **`501 Not Implemented`** — chunked transfer-encoding; shouldn't happen (we PUT raw bytes).
- **`403 RequestTimeTooSkewed`** at boot — device clock not yet NTP/GPS-synced; self-heals on
  retry once time is correct. Not a bad key.
- **TLS / cert errors** — bucket name must not contain dots (the wildcard cert matches one
  label); `aws_setup.py` enforces this.
- **First real device→S3 PUT is the actual gate** — signing and selection are unit-verified
  off-device, but the upload itself only runs on the car.

## Files

- `sigv4.py` — dependency-free AWS SigV4 (UNSIGNED-PAYLOAD streaming PUT)
- `config.py` — loads `/data/s3_uploader.json`
- `uploader.py` — the daemon (`main()`); adapted from `sunnypilot/sunnylink/uploader.py`
- `aws_setup.py` — dev-machine bucket/IAM/lifecycle setup (boto3)
