# comfyui-models

ComfyUI worker image with all studio-redesign models baked in. Workers built from this image can spawn in any RunPod datacenter — no network volume required.

## Image contents

Built on `0don/worker-comfyui:studio-redesign-base` (runtime + 7 custom_nodes pre-installed: Impact Pack, Impact Subpack, controlnet_aux, LayerDiffuse, Manager, Inspire Pack, smZNodes).

Baked model files (~80 GB total):
- **SDXL checkpoints**: prefectPonyXL_v40, lustifySDXLNSFW_endgame, sd_xl_base_1.0 (~20 GB)
- **Flux 2 dev**: diffusion_models + text_encoder + vae (~52 GB)
- **LoRAs** (5): Sinfully Stylish x2, Expressive H, wlop, Jinx (~1.2 GB)
- **Embeddings** (1): EasyNegative (70 KB)
- **ESRGAN upscalers** (4): RealESRGAN_x4plus, AnimeVideoV3, UltraSharp, NMKD-Siax (~250 MB)
- **xinsir SDXL ControlNets** (3): depth, canny, openpose (~7 GB)
- **YOLO detectors** (2): face_yolov8s, hand_yolov9c (~72 MB)
- **SAM**: sam_vit_b_01ec64 (358 MB) — FaceDetailer mask refinement
- **LayerDiffuse SDXL**: layer_xl_transparent_attn (709 MB)

## Build

CI runs on push to `main` or manual dispatch. Uses self-hosted GitHub Actions runner (the image is too large for GitHub-hosted runners' 14 GB disk).

Required secrets:
- `HF_TOKEN` — bypasses anonymous HuggingFace rate limits on multi-GB downloads.

Required permissions on the GitHub token: `packages:write` (to push to ghcr.io).

## Use on RunPod

```bash
# Point the serverless endpoint at this image
runpodctl template update gcgakg920o --image ghcr.io/unorouter/comfyui-models:latest

# Remove the network volume (no longer needed)
curl -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"networkVolumeIds": [], "dataCenterIds": []}' \
  "https://rest.runpod.io/v1/endpoints/<endpoint-id>"
```

## Trade-off

- ✅ Workers spawn in any RunPod datacenter — no capacity throttling locked to a single DC.
- ✅ No network volume storage cost (saves ~$6/mo per 90 GB volume).
- ❌ Cold-host pulls take 5-15 minutes (vs 30-60s with a warm volume). Subsequent pulls on the same machine hit Docker's layer cache.
- ❌ Model changes require a full rebuild + push (~30-45 min). Network volume + `aws s3 cp` would be seconds.

For high-volume production (>100 jobs/day) where cold-host pulls are rare, this is the better architecture. For low-volume personal use with sporadic traffic, a network volume in a high-capacity DC may be a better fit.

## Adding new models

1. Edit `Dockerfile`, add an `RUN hf download …` line.
2. Update `unorouter/src/lib/db/seeds.ts` with the matching catalog row (filename must match exactly).
3. Commit + push. CI builds the new image. `runpodctl template update` to roll it out.
