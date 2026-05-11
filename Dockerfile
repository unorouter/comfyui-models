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
# BuildKit runs them in parallel. Plus within each stage, individual
# files download concurrently via shell `&`. Plus hf-xet's
# HF_XET_HIGH_PERFORMANCE saturates per-file bandwidth. Plus COPY --link
# for fast cross-stage assembly.
#
# Edge case (see comfyui-runpod-memory.md): worker-comfyui's
# extra_model_paths.yaml only scans `unet/` and `clip/`, NOT
# `diffusion_models/` and `text_encoders/`. Flux 2 weights MUST land in
# unet/ and clip/ respectively or ComfyUI returns value_not_in_list.

# ============================================================
# Shared download base — installs hf-xet once, reused by all
# download stages so they don't each repeat the pip install.
# ============================================================
FROM 0don/worker-comfyui:studio-redesign-base AS dl-base
# Conservative concurrency: 7 parallel stages already saturate the
# runner's NIC. HF_XET_HIGH_PERFORMANCE + range=16 on top crushed don
# server's 31GB RAM. Default range_gets (4) is enough alongside stage
# parallelism.
ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=4
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -U "huggingface_hub[hf_xet]"

# ============================================================
# Independent download stages — BuildKit runs all 7 in parallel
# (each `FROM dl-base AS ...` creates an independent branch).
# Final stage merges them all via COPY --link.
# ============================================================

# --- 1) SDXL fine-tunes (~20 GB) ---
FROM dl-base AS dl-sdxl
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    mkdir -p /out/checkpoints && cd /out/checkpoints \
    && hf download Romanos575/prefectPonyXL_v4 prefectPonyXL_v40.safetensors --local-dir . & \
       hf download fandyy24/lustifySDXLNSFW_endgame lustifySDXLNSFW_endgame.safetensors --local-dir . & \
       hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir . & \
       wait

# --- 2) Flux 2 stack (~52 GB, the bottleneck) ---
FROM dl-base AS dl-flux2
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    mkdir -p /out/unet /out/clip /out/vae \
    && ( hf download Comfy-Org/flux2-dev split_files/diffusion_models/flux2_dev_fp8mixed.safetensors --local-dir /tmp/x \
         && mv /tmp/x/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors /out/unet/ ) & \
       ( hf download Comfy-Org/flux2-dev split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors --local-dir /tmp/y \
         && mv /tmp/y/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors /out/clip/ ) & \
       ( hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors --local-dir /tmp/z \
         && mv /tmp/z/split_files/vae/flux2-vae.safetensors /out/vae/ ) & \
       wait && rm -rf /tmp/x /tmp/y /tmp/z

# --- 3) LoRAs (~1.2 GB) — matches LORA_SEEDS in unorouter seeds.ts ---
FROM dl-base AS dl-loras
RUN mkdir -p /out/loras && cd /out/loras \
    && hf download Naznut/Pony_LORAs Sinfully_Stylish_dramitic_bold_lighting.safetensors --local-dir . & \
       hf download Naznut/Pony_LORAs sinfully_stylish_PONY_0.2.safetensors --local-dir . & \
       hf download Naznut/Pony_LORAs Expressive_H-000001.safetensors --local-dir . & \
       hf download SirVeggie/wlop-pony-lora wlop-000018-pony.safetensors --local-dir . & \
       hf download Naznut/Pony_LORAs jinx.safetensors --local-dir . & \
       wait

# --- 4) Embeddings ---
FROM dl-base AS dl-embeddings
RUN mkdir -p /out/embeddings \
    && hf download embed/EasyNegative EasyNegative.safetensors --local-dir /out/embeddings/

# --- 5) ESRGAN upscalers (~250 MB) ---
FROM dl-base AS dl-upscalers
RUN mkdir -p /out/upscale_models && cd /out/upscale_models \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O RealESRGAN_x4plus.pth & \
       wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -O RealESR_AnimeVideoV3.pth & \
       wget -q https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth -O 4x_NMKD-Siax_200k.pth & \
       wget -q https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth -O 4x-UltraSharp.pth & \
       wait

# --- 6) SDXL ControlNets (xinsir, ~7 GB) ---
FROM dl-base AS dl-controlnets
RUN mkdir -p /out/controlnet \
    && ( hf download xinsir/controlnet-depth-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/a \
         && mv /tmp/a/diffusion_pytorch_model.safetensors /out/controlnet/control-depth-sdxl.safetensors ) & \
       ( hf download xinsir/controlnet-canny-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/b \
         && mv /tmp/b/diffusion_pytorch_model.safetensors /out/controlnet/control-canny-sdxl.safetensors ) & \
       ( hf download xinsir/controlnet-openpose-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/c \
         && mv /tmp/c/diffusion_pytorch_model.safetensors /out/controlnet/control-openpose-sdxl.safetensors ) & \
       wait && rm -rf /tmp/a /tmp/b /tmp/c

# --- 7) ADetailer (~800 MB) — full set matches UI dropdown ---
# Impact Pack scans models/ultralytics/{bbox,segm}/. mediapipe_face_*
# are not files, they're internal Impact Pack identifiers.
FROM dl-base AS dl-adetailer
RUN mkdir -p /out/ultralytics/bbox /out/ultralytics/segm /out/sams \
    && cd /out/ultralytics/bbox \
    && for f in face_yolov8s.pt face_yolov9c.pt face_yolov8m.pt face_yolov8n.pt face_yolov8n_v2.pt \
                hand_yolov8s.pt hand_yolov9c.pt hand_yolov8n.pt; do \
         wget -q "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" -O "$f" & \
       done; wait \
    && cd /out/ultralytics/segm \
    && for f in person_yolov8n-seg.pt person_yolov8m-seg.pt person_yolov8s-seg.pt; do \
         wget -q "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" -O "$f" & \
       done; wait \
    && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -O /out/sams/sam_vit_b_01ec64.pth

# --- 8) LayerDiffuse SDXL (~700 MB) ---
FROM dl-base AS dl-layerdiffuse
RUN mkdir -p /out/diffusion_models \
    && hf download LayerDiffusion/layerdiffusion-v1 layer_xl_transparent_attn.safetensors \
        --local-dir /out/diffusion_models/

# ============================================================
# Final stage — merges all download stages with COPY --link.
# --link enables BuildKit's MergeOp: each COPY produces an
# independent layer that can be reused/rebased without
# invalidating subsequent layers.
# ============================================================
FROM 0don/worker-comfyui:studio-redesign-base
COPY --link --from=dl-sdxl         /out/checkpoints       /comfyui/models/checkpoints
COPY --link --from=dl-flux2        /out/unet              /comfyui/models/unet
COPY --link --from=dl-flux2        /out/clip              /comfyui/models/clip
COPY --link --from=dl-flux2        /out/vae               /comfyui/models/vae
COPY --link --from=dl-loras        /out/loras             /comfyui/models/loras
COPY --link --from=dl-embeddings   /out/embeddings        /comfyui/models/embeddings
COPY --link --from=dl-upscalers    /out/upscale_models    /comfyui/models/upscale_models
COPY --link --from=dl-controlnets  /out/controlnet        /comfyui/models/controlnet
COPY --link --from=dl-adetailer    /out/ultralytics       /comfyui/models/ultralytics
COPY --link --from=dl-adetailer    /out/sams              /comfyui/models/sams
COPY --link --from=dl-layerdiffuse /out/diffusion_models  /comfyui/models/diffusion_models

# Verify everything landed where ComfyUI's extra_model_paths.yaml expects.
RUN ls -lh /comfyui/models/checkpoints/ /comfyui/models/loras/ /comfyui/models/embeddings/ \
    /comfyui/models/upscale_models/ /comfyui/models/controlnet/ /comfyui/models/ultralytics/bbox/ \
    /comfyui/models/ultralytics/segm/ /comfyui/models/sams/ /comfyui/models/diffusion_models/ \
    /comfyui/models/unet/ /comfyui/models/clip/ /comfyui/models/vae/
