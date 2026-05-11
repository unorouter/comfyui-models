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
# Edge case (see comfyui-runpod-memory.md): worker-comfyui's
# extra_model_paths.yaml only scans `unet/` and `clip/`, NOT
# `diffusion_models/` and `text_encoders/`. Flux 2 weights MUST land in
# unet/ and clip/ respectively or ComfyUI returns value_not_in_list.

FROM 0don/worker-comfyui:studio-redesign-base

# hf-xet is the 2026 fast path; HF_HUB_ENABLE_HF_TRANSFER is deprecated.
# - HF_XET_HIGH_PERFORMANCE=1 enables saturation mode (was off by default)
# - HF_XET_NUM_CONCURRENT_RANGE_GETS=16 parallel byte-range GETs per file
ENV HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=16
RUN pip install --no-cache-dir -U "huggingface_hub[hf_xet]"

# All HF downloads run in parallel within each layer via `&` + `wait`.
# Each `hf download` saturates its own connection; running N in parallel
# multiplies aggregate throughput up to the runner's NIC limit.

# ---- SDXL fine-tunes (~20 GB, 3 files parallel) ----
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    hf download Romanos575/prefectPonyXL_v4 prefectPonyXL_v40.safetensors \
        --local-dir /comfyui/models/checkpoints/ & \
    hf download fandyy24/lustifySDXLNSFW_endgame lustifySDXLNSFW_endgame.safetensors \
        --local-dir /comfyui/models/checkpoints/ & \
    hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors \
        --local-dir /comfyui/models/checkpoints/ & \
    wait

# ---- Flux 2 dev bundle (~52 GB, 3 files parallel) ----
# Files live under split_files/<type>/ in the source repo. Move to the
# scan paths worker-comfyui's extra_model_paths.yaml looks at.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    ( hf download Comfy-Org/flux2-dev split_files/diffusion_models/flux2_dev_fp8mixed.safetensors \
        --local-dir /tmp/hf-flux2-unet \
      && mv /tmp/hf-flux2-unet/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors /comfyui/models/unet/ ) & \
    ( hf download Comfy-Org/flux2-dev split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors \
        --local-dir /tmp/hf-flux2-clip \
      && mv /tmp/hf-flux2-clip/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors /comfyui/models/clip/ ) & \
    ( hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors \
        --local-dir /tmp/hf-flux2-vae \
      && mv /tmp/hf-flux2-vae/split_files/vae/flux2-vae.safetensors /comfyui/models/vae/ ) & \
    wait && rm -rf /tmp/hf-flux2-*

# ---- LoRAs (~1.2 GB, 5 files parallel) ----
# Matches LORA_SEEDS in unorouter src/lib/db/seeds.ts.
RUN mkdir -p /comfyui/models/loras && cd /comfyui/models/loras \
    && hf download Naznut/Pony_LORAs Sinfully_Stylish_dramitic_bold_lighting.safetensors --local-dir . & \
       hf download Naznut/Pony_LORAs sinfully_stylish_PONY_0.2.safetensors --local-dir . & \
       hf download Naznut/Pony_LORAs Expressive_H-000001.safetensors --local-dir . & \
       hf download SirVeggie/wlop-pony-lora wlop-000018-pony.safetensors --local-dir . & \
       hf download Naznut/Pony_LORAs jinx.safetensors --local-dir . & \
       wait

# ---- Embeddings (~70 KB) ----
RUN mkdir -p /comfyui/models/embeddings \
    && hf download embed/EasyNegative EasyNegative.safetensors --local-dir /comfyui/models/embeddings/

# ---- ESRGAN upscalers (~250 MB, 4 wgets parallel) ----
RUN mkdir -p /comfyui/models/upscale_models && cd /comfyui/models/upscale_models \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O RealESRGAN_x4plus.pth & \
       wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -O RealESR_AnimeVideoV3.pth & \
       wget -q https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth -O 4x_NMKD-Siax_200k.pth & \
       wget -q https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth -O 4x-UltraSharp.pth & \
       wait

# ---- SDXL ControlNets (xinsir, ~7 GB, 3 files parallel) ----
RUN mkdir -p /comfyui/models/controlnet \
    && ( hf download xinsir/controlnet-depth-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/cn-depth \
         && mv /tmp/cn-depth/diffusion_pytorch_model.safetensors /comfyui/models/controlnet/control-depth-sdxl.safetensors ) & \
       ( hf download xinsir/controlnet-canny-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/cn-canny \
         && mv /tmp/cn-canny/diffusion_pytorch_model.safetensors /comfyui/models/controlnet/control-canny-sdxl.safetensors ) & \
       ( hf download xinsir/controlnet-openpose-sdxl-1.0 diffusion_pytorch_model.safetensors --local-dir /tmp/cn-openpose \
         && mv /tmp/cn-openpose/diffusion_pytorch_model.safetensors /comfyui/models/controlnet/control-openpose-sdxl.safetensors ) & \
       wait && rm -rf /tmp/cn-*

# ---- ADetailer dependencies (YOLO + SAM, ~800 MB, 12 wgets parallel) ----
# Full set matching the UI's YOLO_MODELS dropdown in
# unorouter/src/components/pages/sidebar/generate/fields/adetailer-section.tsx.
# mediapipe_face_* are NOT files — Impact Pack maps those names to its
# bundled mediapipe library internally.
RUN mkdir -p /comfyui/models/ultralytics/bbox /comfyui/models/ultralytics/segm /comfyui/models/sams \
    && cd /comfyui/models/ultralytics/bbox \
    && for f in face_yolov8s.pt face_yolov9c.pt face_yolov8m.pt face_yolov8n.pt face_yolov8n_v2.pt \
                hand_yolov8s.pt hand_yolov9c.pt hand_yolov8n.pt; do \
         wget -q "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" -O "$f" & \
       done; wait \
    && cd /comfyui/models/ultralytics/segm \
    && for f in person_yolov8n-seg.pt person_yolov8m-seg.pt person_yolov8s-seg.pt; do \
         wget -q "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" -O "$f" & \
       done; wait \
    && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -O /comfyui/models/sams/sam_vit_b_01ec64.pth

# ---- LayerDiffuse SDXL (~700 MB) ----
RUN mkdir -p /comfyui/models/diffusion_models \
    && hf download LayerDiffusion/layerdiffusion-v1 layer_xl_transparent_attn.safetensors \
        --local-dir /comfyui/models/diffusion_models/

# Verify everything landed where ComfyUI's extra_model_paths.yaml expects.
RUN ls -lh /comfyui/models/checkpoints/ /comfyui/models/loras/ /comfyui/models/embeddings/ \
    /comfyui/models/upscale_models/ /comfyui/models/controlnet/ /comfyui/models/ultralytics/bbox/ \
    /comfyui/models/ultralytics/segm/ /comfyui/models/sams/ /comfyui/models/diffusion_models/ \
    /comfyui/models/unet/ /comfyui/models/clip/ /comfyui/models/vae/
