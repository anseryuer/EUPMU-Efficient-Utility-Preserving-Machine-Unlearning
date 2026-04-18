#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-eupmu}"

source /home/tianbai/miniconda3/etc/profile.d/conda.sh

conda create -y -n "${ENV_NAME}" python=3.10 pip
conda activate "${ENV_NAME}"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/eupmu_cuda_paths.sh" <<'EOF'
#!/usr/bin/env bash
export _EUPMU_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH-}"
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/eupmu_cuda_paths.sh" <<'EOF'
#!/usr/bin/env bash
if [ -n "${_EUPMU_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="${_EUPMU_OLD_LD_LIBRARY_PATH}"
  unset _EUPMU_OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOF

# H200-compatible PyTorch wheels.
python -m pip install --upgrade pip
python -m pip install --force-reinstall \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Keep the bundled CUDA user-space libraries aligned with the cu121 torch build.
# A newer nvJitLink wheel can be pulled in later and break torch import at runtime.
python -m pip install --force-reinstall \
  nvidia-cuda-runtime-cu12==12.1.105 \
  nvidia-cuda-nvrtc-cu12==12.1.105 \
  nvidia-cuda-cupti-cu12==12.1.105 \
  nvidia-cublas-cu12==12.1.3.1 \
  nvidia-cufft-cu12==11.0.2.54 \
  nvidia-curand-cu12==10.3.2.106 \
  nvidia-cusolver-cu12==11.4.5.107 \
  nvidia-cusparse-cu12==12.1.0.106 \
  nvidia-nccl-cu12==2.21.5 \
  nvidia-nvjitlink-cu12==12.1.105 \
  nvidia-nvtx-cu12==12.1.105 \
  nvidia-cudnn-cu12==9.1.0.70

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
  albumentations>=0.4.3 \
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

python -m pip install taming-transformers-rom1504
python -m pip install git+https://github.com/openai/CLIP.git
