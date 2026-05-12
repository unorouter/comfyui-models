# comfyui-models

ComfyUI worker image with R2-backed lazy model prefetch. Workers built from this image are ~5-10 GB; they pull models from Cloudflare R2 on first boot, mirror them onto local NVMe, then start ComfyUI. No DC lock-in, no 100 GB image pull, no network volume.

## Architecture

```
HuggingFace / Civitai (origin)
        |
        v
[scripts/mirror-to-r2.sh] mirrors once
        |
        v
Cloudflare R2 (comfy-models bucket, ~100 GB)  <-- canonical store
        |
        | streamed by
        v
[Cloudflare Worker]  comfy-models-r2.<acct>.workers.dev
        |  bearer auth (R2_WORKER_TOKEN)
        v
[RunPod worker container]
  /prefetch.py  ->  /comfyui/models/<type>/<file>
  exec /start.sh  ->  ComfyUI + rp_handler
```

Three pieces:

1. **`scripts/mirror-to-r2.sh`** - one-shot tool that pulls every model in `model-manifest.json` from HF / Civitai / GitHub releases and uploads it to R2 via rclone. Re-run any time to add models.
2. **`worker/`** - Cloudflare Worker (TypeScript) bound to the R2 bucket. Streams objects to authenticated callers (`GET /get?key=...`, `GET /head?key=...`). No presigning.
3. **Slim Docker image** - `0don/worker-comfyui:studio-redesign-base` + `prefetch.py` + `model-manifest.json`. CMD runs `prefetch.py`, which fetches everything in the manifest, then execs `/start.sh` to launch ComfyUI + the RunPod handler.

## Why this replaces baked-in models

Old image: every model layered into a single ~95 GB Docker image, multi-DC by virtue of being self-contained. Cold pull 5-15 min, model changes meant rebuilding + repushing 100 GB.

New image: ~5-10 GB. Cold pull ~30-60 s. Adding/swapping a model is `rclone copy` + a manifest entry, no rebuild.

R2 economics for this workload: `$0.015/GB-month * 100 GB = $1.50/mo` storage, **zero egress** (Cloudflare's killer feature for AI inference). Class B read ops are negligible at our scale.

## One-time setup

### 1) Create the R2 bucket

```bash
wrangler r2 bucket create comfy-models
```

### 2) Mirror models from HF/Civitai to R2

```bash
# Install deps once
pip install -U "huggingface_hub[hf_xet]"
rclone config  # add an R2 remote called "r2" (Storage=s3, provider=Cloudflare)

# Mirror
HF_TOKEN=hf_xxx R2_REMOTE=r2 R2_BUCKET=comfy-models ./scripts/mirror-to-r2.sh
```

This downloads to `/tmp/comfy-models-stage` then uploads to R2. Both halves are resumable.

### 3) Deploy the Cloudflare Worker

```bash
cd worker
bun install              # or npm install
wrangler secret put AUTH_TOKEN     # paste a long random token; save it for step 4
wrangler deploy
```

Note the deployment URL, e.g. `https://comfy-models-r2.<acct>.workers.dev`.

### 4) Configure the RunPod template env

Set on the RunPod template (or endpoint envOverrides):

| Var | Value |
|---|---|
| `R2_WORKER_URL` | `https://comfy-models-r2.<acct>.workers.dev` |
| `R2_WORKER_TOKEN` | same token from step 3 |
| `PREFETCH_PARALLEL` | `4` (default; tune up if Worker / NVMe can take it) |

### 5) Point the endpoint at the slim image

```bash
runpodctl template update <template-id> --image docker.io/0don/comfyui-models:latest
```

Force a worker rollover so they re-pull:

```bash
curl -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"workersMin":0,"workersMax":0}' \
  "https://rest.runpod.io/v1/endpoints/<endpoint-id>"

sleep 90

curl -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"workersMin":0,"workersMax":10}' \
  "https://rest.runpod.io/v1/endpoints/<endpoint-id>"
```

## Manifest

`model-manifest.json` is the single source of truth for which models a worker prefetches. Each entry:

```json
{
  "r2_key": "checkpoints/prefectPonyXL_v40.safetensors",
  "dest":   "checkpoints/prefectPonyXL_v40.safetensors",
  "size_bytes": 6938040682
}
```

- `r2_key` - object key in the R2 bucket.
- `dest` - path under `/comfyui/models/`. ComfyUI scans default dirs (`checkpoints/`, `loras/`, `unet/`, `clip/`, `vae/`, `controlnet/`, `embeddings/`, `upscale_models/`, `diffusion_models/`, `ultralytics/{bbox,segm}/`, `sams/`) so no `extra_model_paths.yaml` patch is needed.
- `size_bytes` - optional. If set, `prefetch.py` skips re-fetch when local size matches and refetches when it doesn't.

### Adding a new model

1. Upload to R2: `rclone copyto local-file r2:comfy-models/<r2_key>`
2. Add an entry to `model-manifest.json`.
3. Update `unorouter/src/lib/db/seeds.ts` so the catalog row matches the new filename.
4. Commit + push. Build is now seconds (no model downloads in Docker).
5. Either let workers re-pull on next cold start, or force-restart workers so they prefetch the new file.

## Image contents

Built on `0don/worker-comfyui:studio-redesign-base`. Custom nodes pre-installed in the base: Impact Pack, Impact Subpack, controlnet_aux, LayerDiffuse, Inspire Pack, smZNodes; ComfyUI-Manager installed as Python package at `/opt/venv/lib/python3.12/site-packages/comfyui_manager`.

No models are baked. They land on `/comfyui/models/<type>/` at runtime via `prefetch.py`.

Models served from R2 (see `model-manifest.json` for the live list):

| Category | Files | Size |
|---|---|---|
| **SDXL checkpoints** | prefectPonyXL_v40, lustifySDXLNSFW_endgame, sd_xl_base_1.0 | ~20 GB |
| **Flux 2 dev** | flux2_dev_fp8mixed (unet), mistral_3_small_flux2_fp8 (clip), flux2-vae | ~51 GB |
| **LoRAs** | Sinfully_Stylish_dramitic_bold_lighting, sinfully_stylish_PONY_0.2, Expressive_H-000001, wlop-000018-pony, jinx | ~1.2 GB |
| **Embeddings** | EasyNegative | 25 KB |
| **ESRGAN upscalers** | RealESRGAN_x4plus, RealESR_AnimeVideoV3, 4x-UltraSharp, 4x_NMKD-Siax_200k | ~195 MB |
| **xinsir SDXL ControlNets** | depth, canny, openpose | ~7 GB |
| **YOLO bbox detectors** | face_yolov8s/m/n/n_v2/v9c, hand_yolov8s/n/v9c (8 files) | ~210 MB |
| **YOLO segm detectors** | person_yolov8n/m/s-seg (3 files) | ~82 MB |
| **SAM** | sam_vit_b_01ec64 | 358 MB |
| **LayerDiffuse SDXL** | layer_xl_transparent_attn | 709 MB |

## Worker prefetch behavior

`prefetch.py` runs as PID 1. Per manifest entry:

1. If `dest` already exists and matches `size_bytes` (when provided), skip.
2. Else `GET {R2_WORKER_URL}/get?key=<r2_key>` with `Authorization: Bearer <token>`.
3. Stream to `<dest>.part`, atomic rename to `<dest>`.
4. Retry up to 5 times with exponential backoff on network failures.

Up to `PREFETCH_PARALLEL` (default 4) downloads run concurrently. Once everything is on disk, `prefetch.py` execs `/start.sh` (inherited from the base image), which launches ComfyUI plus the RunPod handler.

**Failure handling**: any unrecoverable prefetch error aborts before ComfyUI starts, so the worker fails fast rather than serving requests with a missing model.

## Cold-start budget

Approx times in EU regions, R2 colocated, `PREFETCH_PARALLEL=4`:

| Phase | Old (baked) | New (R2 prefetch) |
|---|---|---|
| Docker pull | 5-15 min | 30-60 s |
| Model load | included above | 60-120 s (first boot) |
| ComfyUI init | 20-30 s | 20-30 s |
| **Total cold** | **5-15 min** | **~2-3 min** |

Subsequent boots on the same physical RunPod host hit the Docker layer cache *and* the prefetched models on the worker filesystem stay until the host evicts them.

## Build

CI runs on push to `main` or manual dispatch. Since there are no model downloads in the build, GitHub-hosted runners are fine and the previous self-hosted requirement is gone. Build now takes seconds.

Required secrets:
- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` - publish to Docker Hub
- `GITHUB_TOKEN` - publish to ghcr.io (`packages:write`)

No HF token needed at build time (models live in R2, not in the image).

### Tags published

- `docker.io/0don/comfyui-models:latest`
- `docker.io/0don/comfyui-models:<sha>`
- `ghcr.io/unorouter/comfyui-models:latest`
- `ghcr.io/unorouter/comfyui-models:<sha>`

Docker Hub ~6x faster cold pull than ghcr.io from EU-RO-1.

## Trade-offs

- ✅ Image is ~5-10 GB. Cold pull ~30-60 s.
- ✅ Model swaps are an R2 upload + manifest edit. No CI rebuild.
- ✅ Workers spawn in any RunPod DC (R2 is global, zero egress fees).
- ✅ Build is seconds, runs on GitHub-hosted runners.
- ❌ First cold boot on a fresh host adds ~60-120 s for prefetch (vs 0 s once models were baked).
- ❌ Requires Cloudflare Worker to be reachable from RunPod. Hard outage of Workers = workers can't boot. Mitigations: keep the Worker stupid simple (current impl is one R2 binding + auth check), monitor health endpoint.

## Files

- `Dockerfile` - slim build, copies `prefetch.py` + `model-manifest.json` over the studio-redesign-base image.
- `model-manifest.json` - canonical list of R2 keys to fetch on boot.
- `prefetch.py` - PID 1 inside the container; downloads then execs `/start.sh`.
- `worker/` - Cloudflare Worker (R2 binding + bearer auth).
- `scripts/mirror-to-r2.sh` - HF/Civitai/GitHub -> local -> R2 mirror.
- `.github/workflows/build.yml` - image build/publish.
