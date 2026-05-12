# syntax=docker/dockerfile:1.10
#
# Models baked into the worker image so RunPod workers can spawn in any
# datacenter without a network volume. Built on top of the studio-redesign
# runtime image (ComfyUI + Impact Pack + ControlNet aux + LayerDiffuse +
# Inspire Pack + smZNodes pre-installed).
#
# Image size with everything baked: ~95-110 GB. Cold-host pulls take 5-15
# minutes; subsequent pulls hit the Docker layer cache on the same machine.
# Trade-off: long cold start but no DC lock-in -> workers spawn in any
# region with GPU capacity.
#
# Performance: download phases are split into independent stages so
# BuildKit runs them in parallel. Each individual file gets its OWN
# stage (no shell `&` background races inside a stage). Plus hf-xet's
# range_gets saturates per-file bandwidth. Plus COPY --link for fast
# cross-stage assembly.
#
# Edge case (see comfyui-runpod-memory.md): worker-comfyui's
# extra_model_paths.yaml only scans `unet/` and `clip/`, NOT
# `diffusion_models/` and `text_encoders/`. Flux 2 weights MUST land in
# unet/ and clip/ respectively or ComfyUI returns value_not_in_list.
#
# Critical bug fixed 2026-05-12: previously dl-sdxl ran 3 `hf download &`
# in background within ONE stage. Only Pony survived to the final image
# - Endgame + SDXL base were silently lost (hf-xet metadata lock or
# shared --local-dir cache race). Now each checkpoint is its own stage.

# ============================================================
# Shared download base - installs hf-xet once, reused by all
# download stages so they don't each repeat the pip install.
# ============================================================
FROM 0don/worker-comfyui:studio-redesign-base AS dl-base
ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=4
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -U "huggingface_hub[hf_xet]"

# ============================================================
# Independent download stages - BuildKit runs all in parallel.
# One file per stage = no intra-stage race conditions.
# ============================================================

# --- 1) SDXL fine-tunes: one stage per ckpt ---
# Per-stage HF cache id avoids cross-stage lock contention while still
# letting re-runs of the SAME stage reuse already-fetched blobs.
FROM dl-base AS dl-pony
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-pony \
    mkdir -p /out/checkpoints \
    && hf download Romanos575/prefectPonyXL_v4 prefectPonyXL_v40.safetensors \
        --local-dir /out/checkpoints

FROM dl-base AS dl-endgame
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-endgame \
    mkdir -p /out/checkpoints \
    && hf download fandyy24/lustifySDXLNSFW_endgame lustifySDXLNSFW_endgame.safetensors \
        --local-dir /out/checkpoints

FROM dl-base AS dl-sdxl-base
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-sdxl-base \
    mkdir -p /out/checkpoints \
    && hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors \
        --local-dir /out/checkpoints

# --- 2) Flux 2 stack: one stage per file ---
FROM dl-base AS dl-flux2-unet
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    --mount=type=cache,target=/root/.cache/huggingface,id=hf-flux2-unet \
    mkdir -p /out/unet \
    && hf download Comfy-Org/flux2-dev split_files/diffusion_models/flux2_dev_fp8mixed.safetensors --local-dir /tmp/x \
    && mv /tmp/x/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors /out/unet/ \
    && rm -rf /tmp/x

FROM dl-base AS dl-flux2-clip
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    --mount=type=cache,target=/root/.cache/huggingface,id=hf-flux2-clip \
    mkdir -p /out/clip \
    && hf download Comfy-Org/flux2-dev split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors --local-dir /tmp/y \
    && mv /tmp/y/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors /out/clip/ \
    && rm -rf /tmp/y

FROM dl-base AS dl-flux2-vae
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    --mount=type=cache,target=/root/.cache/huggingface,id=hf-flux2-vae \
    mkdir -p /out/vae \
    && hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors --local-dir /tmp/z \
    && mv /tmp/z/split_files/vae/flux2-vae.safetensors /out/vae/ \
    && rm -rf /tmp/z

# --- 3) LoRAs: one stage per file (matches LORA_SEEDS in unorouter seeds.ts) ---
FROM dl-base AS dl-lora-1
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-lora-1 \
    mkdir -p /out/loras \
    && hf download Naznut/Pony_LORAs Sinfully_Stylish_dramitic_bold_lighting.safetensors --local-dir /out/loras

FROM dl-base AS dl-lora-2
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-lora-2 \
    mkdir -p /out/loras \
    && hf download Naznut/Pony_LORAs sinfully_stylish_PONY_0.2.safetensors --local-dir /out/loras

FROM dl-base AS dl-lora-3
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-lora-3 \
    mkdir -p /out/loras \
    && hf download Naznut/Pony_LORAs Expressive_H-000001.safetensors --local-dir /out/loras

FROM dl-base AS dl-lora-4
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-lora-4 \
    mkdir -p /out/loras \
    && hf download SirVeggie/wlop-pony-lora wlop-000018-pony.safetensors --local-dir /out/loras

FROM dl-base AS dl-lora-5
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-lora-5 \
    mkdir -p /out/loras \
    && hf download Naznut/Pony_LORAs jinx.safetensors --local-dir /out/loras

# --- 4) Embeddings ---
FROM dl-base AS dl-embeddings
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-embeddings \
    mkdir -p /out/embeddings \
    && hf download embed/EasyNegative EasyNegative.safetensors --local-dir /out/embeddings/

# --- 5) ESRGAN upscalers: one stage per file ---
# wget -c + cache mount lets restarted/partial fetches resume instead of
# re-downloading from byte 0 on every cache miss.
FROM dl-base AS dl-upscale-1
RUN --mount=type=cache,target=/var/cache/dl,id=dl-upscale-1 \
    mkdir -p /out/upscale_models \
    && wget -qc -P /var/cache/dl https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    && cp /var/cache/dl/RealESRGAN_x4plus.pth /out/upscale_models/RealESRGAN_x4plus.pth

FROM dl-base AS dl-upscale-2
RUN --mount=type=cache,target=/var/cache/dl,id=dl-upscale-2 \
    mkdir -p /out/upscale_models \
    && wget -qc -P /var/cache/dl https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth \
    && cp /var/cache/dl/realesr-animevideov3.pth /out/upscale_models/RealESR_AnimeVideoV3.pth

FROM dl-base AS dl-upscale-3
RUN --mount=type=cache,target=/var/cache/dl,id=dl-upscale-3 \
    mkdir -p /out/upscale_models \
    && wget -qc -P /var/cache/dl https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth \
    && cp /var/cache/dl/4x_NMKD-Siax_200k.pth /out/upscale_models/4x_NMKD-Siax_200k.pth

FROM dl-base AS dl-upscale-4
RUN --mount=type=cache,target=/var/cache/dl,id=dl-upscale-4 \
    mkdir -p /out/upscale_models \
    && wget -qc -P /var/cache/dl https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth \
    && cp /var/cache/dl/4x-UltraSharp.pth /out/upscale_models/4x-UltraSharp.pth

# --- 6) SDXL ControlNets (xinsir): one stage per file ---
FROM dl-base AS dl-cn-depth
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-cn-depth \
    mkdir -p /out/controlnet \
    && hf download xinsir/controlnet-depth-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/a \
    && mv /tmp/a/diffusion_pytorch_model.safetensors /out/controlnet/control-depth-sdxl.safetensors \
    && rm -rf /tmp/a

FROM dl-base AS dl-cn-canny
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-cn-canny \
    mkdir -p /out/controlnet \
    && hf download xinsir/controlnet-canny-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/b \
    && mv /tmp/b/diffusion_pytorch_model.safetensors /out/controlnet/control-canny-sdxl.safetensors \
    && rm -rf /tmp/b

FROM dl-base AS dl-cn-openpose
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-cn-openpose \
    mkdir -p /out/controlnet \
    && hf download xinsir/controlnet-openpose-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/c \
    && mv /tmp/c/diffusion_pytorch_model.safetensors /out/controlnet/control-openpose-sdxl.safetensors \
    && rm -rf /tmp/c

# --- 7) ADetailer YOLO bbox/segm + SAM: one stage per file ---
# Impact Pack scans models/ultralytics/{bbox,segm}/.
FROM dl-base AS dl-yolo-bbox
RUN --mount=type=cache,target=/var/cache/dl,id=dl-yolo-bbox \
    mkdir -p /out/ultralytics/bbox /var/cache/dl \
    && for f in face_yolov8s.pt face_yolov9c.pt face_yolov8m.pt face_yolov8n.pt face_yolov8n_v2.pt \
                hand_yolov8s.pt hand_yolov9c.pt hand_yolov8n.pt; do \
         wget -qc -P /var/cache/dl "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" \
         && cp "/var/cache/dl/$f" "/out/ultralytics/bbox/$f"; \
       done

FROM dl-base AS dl-yolo-segm
RUN --mount=type=cache,target=/var/cache/dl,id=dl-yolo-segm \
    mkdir -p /out/ultralytics/segm /var/cache/dl \
    && for f in person_yolov8n-seg.pt person_yolov8m-seg.pt person_yolov8s-seg.pt; do \
         wget -qc -P /var/cache/dl "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" \
         && cp "/var/cache/dl/$f" "/out/ultralytics/segm/$f"; \
       done

FROM dl-base AS dl-sam
RUN --mount=type=cache,target=/var/cache/dl,id=dl-sam \
    mkdir -p /out/sams \
    && wget -qc -P /var/cache/dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    && cp /var/cache/dl/sam_vit_b_01ec64.pth /out/sams/sam_vit_b_01ec64.pth

# --- 8) LayerDiffuse SDXL ---
FROM dl-base AS dl-layerdiffuse
RUN --mount=type=cache,target=/root/.cache/huggingface,id=hf-layerdiffuse \
    mkdir -p /out/diffusion_models \
    && hf download LayerDiffusion/layerdiffusion-v1 layer_xl_transparent_attn.safetensors \
        --local-dir /out/diffusion_models/

# ============================================================
# Final stage - merges all download stages with COPY --link.
# Multiple COPY into same dest dir works fine with --link.
# ============================================================
FROM 0don/worker-comfyui:studio-redesign-base

# Checkpoints (3 separate stages each writing to /out/checkpoints)
COPY --link --from=dl-pony         /out/checkpoints       /comfyui/models/checkpoints
COPY --link --from=dl-endgame      /out/checkpoints       /comfyui/models/checkpoints
COPY --link --from=dl-sdxl-base    /out/checkpoints       /comfyui/models/checkpoints

# Flux 2 (3 stages, 3 dirs)
COPY --link --from=dl-flux2-unet   /out/unet              /comfyui/models/unet
COPY --link --from=dl-flux2-clip   /out/clip              /comfyui/models/clip
COPY --link --from=dl-flux2-vae    /out/vae               /comfyui/models/vae

# LoRAs (5 separate stages)
COPY --link --from=dl-lora-1       /out/loras             /comfyui/models/loras
COPY --link --from=dl-lora-2       /out/loras             /comfyui/models/loras
COPY --link --from=dl-lora-3       /out/loras             /comfyui/models/loras
COPY --link --from=dl-lora-4       /out/loras             /comfyui/models/loras
COPY --link --from=dl-lora-5       /out/loras             /comfyui/models/loras

# Embeddings
COPY --link --from=dl-embeddings   /out/embeddings        /comfyui/models/embeddings

# Upscalers (4 separate stages)
COPY --link --from=dl-upscale-1    /out/upscale_models    /comfyui/models/upscale_models
COPY --link --from=dl-upscale-2    /out/upscale_models    /comfyui/models/upscale_models
COPY --link --from=dl-upscale-3    /out/upscale_models    /comfyui/models/upscale_models
COPY --link --from=dl-upscale-4    /out/upscale_models    /comfyui/models/upscale_models

# ControlNets (3 separate stages)
COPY --link --from=dl-cn-depth     /out/controlnet        /comfyui/models/controlnet
COPY --link --from=dl-cn-canny     /out/controlnet        /comfyui/models/controlnet
COPY --link --from=dl-cn-openpose  /out/controlnet        /comfyui/models/controlnet

# YOLO + SAM
COPY --link --from=dl-yolo-bbox    /out/ultralytics       /comfyui/models/ultralytics
COPY --link --from=dl-yolo-segm    /out/ultralytics       /comfyui/models/ultralytics
COPY --link --from=dl-sam          /out/sams              /comfyui/models/sams

# LayerDiffuse
COPY --link --from=dl-layerdiffuse /out/diffusion_models  /comfyui/models/diffusion_models

# Verify everything landed. List with counts so log shows file count per dir.
RUN echo "=== Staged models ===" \
    && for d in checkpoints loras embeddings upscale_models controlnet ultralytics/bbox \
                ultralytics/segm sams diffusion_models unet clip vae; do \
         echo "--- /comfyui/models/$d/ ---"; \
         ls -lh "/comfyui/models/$d/" 2>&1 | tail -n +2; \
       done
