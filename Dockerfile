# fakinflu — worker RunPod Hunyuan3D-2 (texture Paint v2-0-Turbo + forme HD DiT v2-0-Turbo)
# Image "devel" : necessaire pour compiler custom_rasterizer / differentiable_renderer (CUDA).
ARG CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HY3DGEN_MODELS=/models \
    U2NET_HOME=/models/u2net \
    # pas de GPU au build : on cible explicitement A10/A40/L4/4090/A100/H100
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git ca-certificates \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# 1) Torch (meme version que le worker promptgen)
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2) Hunyuan3D-2 — commit epingle pour des builds reproductibles
ARG HY3D_REF=f8db63096c8282cb27354314d896feba5ba6ff8a
RUN git clone --filter=blob:none https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/Hunyuan3D-2 \
 && cd /opt/Hunyuan3D-2 && git checkout ${HY3D_REF} && rm -rf .git assets

# Deps sans gradio/fastapi (inutiles ici) — d'ou --no-deps sur hy3dgen
RUN pip3 install \
      "diffusers==0.32.2" "transformers==4.48.3" "huggingface_hub==0.27.1" accelerate \
      einops omegaconf opencv-python-headless "numpy<2" tqdm \
      trimesh pymeshlab pygltflib xatlas ninja pybind11 \
      rembg onnxruntime runpod \
 && pip3 install --no-deps -e /opt/Hunyuan3D-2

# 3) Extensions CUDA de la texture
RUN cd /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python3 setup.py install \
 && cd /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && python3 setup.py install

# 4) Poids BAKES (seulement les sous-dossiers utiles, safetensors uniquement : ~17 Go
#    au lieu de ~60 Go pour tout le repo) -> pas de telechargement au cold start.
RUN python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "tencent/Hunyuan3D-2",
    local_dir="/models/tencent/Hunyuan3D-2",
    allow_patterns=[
        "hunyuan3d-dit-v2-0-turbo/config.yaml",
        "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors",
        "hunyuan3d-vae-v2-0-turbo/config.yaml",
        "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors",
        "hunyuan3d-delight-v2-0/*",
        "hunyuan3d-paint-v2-0-turbo/*",
    ],
    ignore_patterns=["*.ckpt", "hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin"],
)
EOF

# 5) Modele rembg (u2net) bake aussi
RUN python3 -c "from rembg import new_session; new_session()"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
