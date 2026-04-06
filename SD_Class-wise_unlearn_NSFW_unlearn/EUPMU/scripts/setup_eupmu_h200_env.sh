#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-eupmu-h200}"

source /opt/conda/etc/profile.d/conda.sh

conda create -y -n "${ENV_NAME}" python=3.10 pip
conda activate "${ENV_NAME}"

# H200-compatible PyTorch wheels.
python -m pip install --upgrade pip
python -m pip install --force-reinstall \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Repo/runtime dependencies used by the Imagenette SD class-wise path.
python -m pip install \
  numpy==1.24.4 \
  scipy==1.9.1 \
  pandas==2.0.3 \
  matplotlib==3.7.5 \
  imageio==2.9.0 \
  imageio-ffmpeg==0.4.2 \
  omegaconf==2.1.1 \
  einops==0.3.0 \
  opencv-python-headless==4.7.0.72 \
  albumentations==0.4.3 \
  diffusers==0.14.0 \
  transformers==4.26.1 \
  huggingface_hub==0.14.1 \
  datasets==2.14.7 \
  pytorch-lightning==1.4.2 \
  torchmetrics==0.6.0 \
  torch-fidelity==0.3.0 \
  kornia==0.6.0 \
  test-tube==0.7.5 \
  wandb==0.17.9 \
  invisible-watermark==0.2.0 \
  pudb==2019.2

python -m pip install -e ./src/taming-transformers

python - <<'PY'
import torch
import cv2
from datasets import load_dataset
from diffusers import LMSDiscreteScheduler
from imwatermark import WatermarkEncoder
from pytorch_lightning import seed_everything
from torchmetrics.image.fid import FID

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2, device="cuda:0")
    print("cuda_tensor_ok", tuple(x.shape), x.device)
print("cv2", cv2.__version__)
ds = load_dataset("frgfm/imagenette", "160px", split="train[:1]")
print("imagenette_ok", len(ds))
print("env_ok")
PY
