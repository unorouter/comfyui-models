#!/usr/bin/env python3
"""
Prefetch models from Cloudflare R2 to local NVMe before ComfyUI starts.

Flow:
  1. Read model-manifest.json (R2 keys + local dests).
  2. For each model: skip if dest exists with correct size.
  3. Otherwise stream from R2 via comfy-models-r2 Worker over a bearer
     token. The Worker holds the R2 binding; the token gates reads.
  4. Atomic rename into /comfyui/models/<type>/<file>.
  5. Exec /start.sh (worker-comfyui launcher) when done.
"""

import concurrent.futures
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("COMFY_MODELS_ROOT", "/comfyui/models"))
MANIFEST_PATH = Path(os.environ.get("MODEL_MANIFEST", "/model-manifest.json"))
WORKER_URL = os.environ.get("R2_WORKER_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("R2_WORKER_TOKEN", "")
START_SCRIPT = os.environ.get("START_SCRIPT", "/start.sh")
MAX_PARALLEL = int(os.environ.get("PREFETCH_PARALLEL", "4"))
CHUNK_SIZE = 16 * 1024 * 1024
HTTP_TIMEOUT = 300
MAX_RETRIES = 5


def log(msg: str) -> None:
    print(f"[prefetch] {msg}", flush=True)


def build_request(path: str, key: str, method: str = "GET") -> urllib.request.Request:
    if not WORKER_URL or not WORKER_TOKEN:
        raise RuntimeError("R2_WORKER_URL and R2_WORKER_TOKEN must be set")
    url = f"{WORKER_URL}{path}?key={urllib.parse.quote(key, safe='/')}"
    return urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
    )


def head_size(key: str) -> int | None:
    req = build_request("/head", key, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download(key: str, dest: Path, expected_size: int | None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        try:
            req = build_request("/get", key)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, tmp.open("wb") as f:
                shutil.copyfileobj(resp, f, length=CHUNK_SIZE)
            size = tmp.stat().st_size
            if expected_size is not None and size != expected_size:
                raise IOError(f"size mismatch: got {size} expected {expected_size}")
            tmp.rename(dest)
            return size
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            tmp.unlink(missing_ok=True)
            if attempt >= MAX_RETRIES:
                raise
            backoff = min(60, 2 ** attempt)
            log(f"retry {attempt}/{MAX_RETRIES} for {key} after error: {e} (sleep {backoff}s)")
            time.sleep(backoff)


def ensure_model(entry: dict) -> tuple[str, float, int, bool]:
    r2_key = entry["r2_key"]
    dest = MODELS_ROOT / entry["dest"]
    expected = entry.get("size_bytes")

    if dest.exists():
        if expected is None or dest.stat().st_size == expected:
            return r2_key, 0.0, dest.stat().st_size, True
        log(f"size mismatch on cached {dest} ({dest.stat().st_size} != {expected}), refetching")
        dest.unlink()

    started = time.monotonic()
    size = download(r2_key, dest, expected)
    elapsed = time.monotonic() - started
    return r2_key, elapsed, size, False


def prefetch_all(manifest: dict) -> None:
    models = manifest["models"]
    log(f"prefetching {len(models)} models -> {MODELS_ROOT}")
    started = time.monotonic()
    total_bytes = 0
    fetched_count = 0
    failures: list[tuple[str, BaseException]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(ensure_model, m): m for m in models}
        for fut in concurrent.futures.as_completed(futures):
            m = futures[fut]
            try:
                key, elapsed, size, cached = fut.result()
                total_bytes += size
                if cached:
                    log(f"cached {key}")
                else:
                    fetched_count += 1
                    mb = size / (1024 * 1024)
                    rate = mb / elapsed if elapsed else 0
                    log(f"fetched {key} ({mb:.0f} MB in {elapsed:.1f}s, {rate:.1f} MB/s)")
            except Exception as e:
                failures.append((m["r2_key"], e))
                log(f"FAILED {m['r2_key']}: {e}")

    total_gb = total_bytes / (1024 ** 3)
    total_elapsed = time.monotonic() - started
    log(f"done: {fetched_count} fetched, {total_gb:.1f} GB on disk, {total_elapsed:.1f}s wall")

    if failures:
        log(f"{len(failures)} prefetch failures, aborting before ComfyUI starts")
        sys.exit(1)


def main() -> None:
    if not MANIFEST_PATH.exists():
        log(f"manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
    manifest = json.loads(MANIFEST_PATH.read_text())
    prefetch_all(manifest)

    log(f"exec {START_SCRIPT}")
    os.execv(START_SCRIPT, [START_SCRIPT])


if __name__ == "__main__":
    main()
