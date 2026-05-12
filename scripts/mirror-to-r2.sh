#!/usr/bin/env bash
#
# One-time (or rerunnable) mirror of HF/Civitai model set to Cloudflare R2.
#
# Reads model-manifest.json, downloads each file from HuggingFace (and a
# few GitHub release URLs for ESRGAN/SAM), then uploads to the R2 bucket
# with matching keys. rclone handles parallelism and resume; hf-xet
# accelerates the HF downloads.
#
# Pre-reqs:
#   - rclone configured with an R2 remote named "r2":
#       rclone config
#       (Storage = s3, provider = Cloudflare, endpoint = https://<ACCT>.r2.cloudflarestorage.com)
#   - huggingface_hub installed (pip install -U "huggingface_hub[hf_xet]")
#   - HF_TOKEN env var set (gated repos: Flux 2)
#
# Usage:
#   R2_REMOTE=r2 R2_BUCKET=comfy-models HF_TOKEN=hf_xxx ./scripts/mirror-to-r2.sh
#
# Idempotent: rclone copy skips files already present with matching size.

set -euo pipefail

R2_REMOTE=${R2_REMOTE:-r2}
R2_BUCKET=${R2_BUCKET:-comfy-models}
STAGING=${STAGING:-/tmp/comfy-models-stage}
PARALLEL=${PARALLEL:-4}

mkdir -p "$STAGING"

hf_grab() {
  local repo=$1 file=$2 dest_dir=$3
  mkdir -p "$STAGING/$dest_dir"
  if [ -f "$STAGING/$dest_dir/$(basename "$file")" ]; then
    echo "[skip] $repo/$file"
    return
  fi
  echo "[hf]   $repo/$file"
  hf download "$repo" "$file" --local-dir "$STAGING/$dest_dir"
}

hf_grab_subpath() {
  # For repos that nest files under split_files/<type>/<file>; we flatten
  # to dest_dir/<basename(file)>.
  local repo=$1 file=$2 dest_dir=$3
  local base
  base=$(basename "$file")
  if [ -f "$STAGING/$dest_dir/$base" ]; then
    echo "[skip] $repo/$file"
    return
  fi
  echo "[hf]   $repo/$file"
  local tmp
  tmp=$(mktemp -d)
  hf download "$repo" "$file" --local-dir "$tmp"
  mkdir -p "$STAGING/$dest_dir"
  mv "$tmp/$file" "$STAGING/$dest_dir/$base"
  rm -rf "$tmp"
}

wget_grab() {
  local url=$1 dest_path=$2
  if [ -f "$STAGING/$dest_path" ]; then
    echo "[skip] $url"
    return
  fi
  echo "[wget] $url"
  mkdir -p "$(dirname "$STAGING/$dest_path")"
  wget -qc -O "$STAGING/$dest_path" "$url"
}

# --- SDXL checkpoints ---
hf_grab "Romanos575/prefectPonyXL_v4"               "prefectPonyXL_v40.safetensors"        "checkpoints"
hf_grab "fandyy24/lustifySDXLNSFW_endgame"          "lustifySDXLNSFW_endgame.safetensors"  "checkpoints"
hf_grab "stabilityai/stable-diffusion-xl-base-1.0"  "sd_xl_base_1.0.safetensors"           "checkpoints"

# --- Flux 2 (gated, needs HF_TOKEN) ---
hf_grab_subpath "Comfy-Org/flux2-dev" "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors" "unet"
hf_grab_subpath "Comfy-Org/flux2-dev" "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" "clip"
hf_grab_subpath "Comfy-Org/flux2-dev" "split_files/vae/flux2-vae.safetensors" "vae"

# --- LoRAs ---
hf_grab "Naznut/Pony_LORAs"        "Sinfully_Stylish_dramitic_bold_lighting.safetensors" "loras"
hf_grab "Naznut/Pony_LORAs"        "sinfully_stylish_PONY_0.2.safetensors"               "loras"
hf_grab "Naznut/Pony_LORAs"        "Expressive_H-000001.safetensors"                     "loras"
hf_grab "SirVeggie/wlop-pony-lora" "wlop-000018-pony.safetensors"                        "loras"
hf_grab "Naznut/Pony_LORAs"        "jinx.safetensors"                                    "loras"

# --- Embeddings ---
hf_grab "embed/EasyNegative" "EasyNegative.safetensors" "embeddings"

# --- ESRGAN upscalers ---
wget_grab "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"        "upscale_models/RealESRGAN_x4plus.pth"
wget_grab "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"  "upscale_models/RealESR_AnimeVideoV3.pth"
wget_grab "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth"               "upscale_models/4x_NMKD-Siax_200k.pth"
wget_grab "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"                   "upscale_models/4x-UltraSharp.pth"

# --- SDXL ControlNets (xinsir) ---
hf_grab_subpath "xinsir/controlnet-depth-sdxl-1.0"    "diffusion_pytorch_model.safetensors" "controlnet/_depth_tmp"
hf_grab_subpath "xinsir/controlnet-canny-sdxl-1.0"    "diffusion_pytorch_model.safetensors" "controlnet/_canny_tmp"
hf_grab_subpath "xinsir/controlnet-openpose-sdxl-1.0" "diffusion_pytorch_model.safetensors" "controlnet/_openpose_tmp"
mv -n "$STAGING/controlnet/_depth_tmp/diffusion_pytorch_model.safetensors"    "$STAGING/controlnet/control-depth-sdxl.safetensors"    || true
mv -n "$STAGING/controlnet/_canny_tmp/diffusion_pytorch_model.safetensors"    "$STAGING/controlnet/control-canny-sdxl.safetensors"    || true
mv -n "$STAGING/controlnet/_openpose_tmp/diffusion_pytorch_model.safetensors" "$STAGING/controlnet/control-openpose-sdxl.safetensors" || true
rmdir "$STAGING/controlnet/_depth_tmp" "$STAGING/controlnet/_canny_tmp" "$STAGING/controlnet/_openpose_tmp" 2>/dev/null || true

# --- YOLO bbox/segm + SAM ---
for f in face_yolov8s.pt face_yolov9c.pt face_yolov8m.pt face_yolov8n.pt face_yolov8n_v2.pt \
         hand_yolov8s.pt hand_yolov9c.pt hand_yolov8n.pt; do
  wget_grab "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" "ultralytics/bbox/$f"
done
for f in person_yolov8n-seg.pt person_yolov8m-seg.pt person_yolov8s-seg.pt; do
  wget_grab "https://huggingface.co/Bingsu/adetailer/resolve/main/$f" "ultralytics/segm/$f"
done
wget_grab "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" "sams/sam_vit_b_01ec64.pth"

# --- LayerDiffuse ---
hf_grab "LayerDiffusion/layerdiffusion-v1" "layer_xl_transparent_attn.safetensors" "diffusion_models"

# --- Push everything to R2 ---
echo
echo "=== Uploading $STAGING -> $R2_REMOTE:$R2_BUCKET ==="
rclone copy "$STAGING" "$R2_REMOTE:$R2_BUCKET" \
  --transfers "$PARALLEL" \
  --checkers $((PARALLEL * 2)) \
  --s3-upload-concurrency 4 \
  --s3-chunk-size 64M \
  --progress

echo
echo "Done. R2 bucket $R2_BUCKET now mirrors $STAGING."
echo "Verify keys match model-manifest.json:"
echo "  rclone ls $R2_REMOTE:$R2_BUCKET | sort"
