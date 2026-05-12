# syntax=docker/dockerfile:1.10
#
# Slim ComfyUI worker. Models live in Cloudflare R2, fetched on first
# worker boot into /comfyui/models, then ComfyUI starts normally.
#
# Why: previously this image baked all ~100 GB of models so RunPod
# workers could spawn in any DC without a network volume. Cold pull was
# 5-15 min and every model update meant rebuilding + repushing 100 GB.
# Switching to R2-backed lazy prefetch drops image size to ~5-10 GB,
# cold pull to ~30-60 s, and model swaps to a single R2 upload.
#
# Flow on container start:
#   /prefetch.py reads /model-manifest.json -> for each entry, calls the
#   comfy-models-r2 Cloudflare Worker over HTTPS with a bearer token.
#   Worker streams the object from a private R2 bucket. prefetch.py
#   writes to a .part file and atomically renames into
#   /comfyui/models/<type>/. When all entries are present (with matching
#   size_bytes if provided), prefetch.py execs /start.sh from the base
#   image, which launches ComfyUI plus the RunPod handler.
#
# Env required at runtime (set on RunPod template):
#   R2_WORKER_URL    https URL of the worker (e.g. https://comfy-models-r2.<acct>.workers.dev)
#   R2_WORKER_TOKEN  matches AUTH_TOKEN secret on the worker
#   PREFETCH_PARALLEL (optional, default 4)
#
# Custom nodes (smZ, LayerDiffuse, Impact Pack, ControlNet aux, Inspire
# Pack) come pre-installed in the studio-redesign-base image and are NOT
# touched here.

FROM 0don/worker-comfyui:studio-redesign-base

COPY prefetch.py /prefetch.py
COPY model-manifest.json /model-manifest.json

# Ensure model dirs exist with permissions before prefetch writes into them.
RUN mkdir -p /comfyui/models/checkpoints \
             /comfyui/models/loras \
             /comfyui/models/embeddings \
             /comfyui/models/upscale_models \
             /comfyui/models/controlnet \
             /comfyui/models/ultralytics/bbox \
             /comfyui/models/ultralytics/segm \
             /comfyui/models/sams \
             /comfyui/models/diffusion_models \
             /comfyui/models/unet \
             /comfyui/models/clip \
             /comfyui/models/vae

# Override base CMD so prefetch runs first; prefetch.py execs /start.sh
# when all models are on disk. Base image CMD was ["/start.sh"].
CMD ["python3", "-u", "/prefetch.py"]
