#!/usr/bin/env bash
set -euo pipefail

# Paper-style Table 1 reproduction for Stable Diffusion class-wise forgetting
# using EUPMU and OMD-TCH.
#
# Paper SD settings from Appendix D:
# - optimizer: Adam
# - epochs: 5
# - lr: 1e-5
# - alpha: 0.01
# - batch_size: 8
# - DDIM steps: 100
# - guidance scale: 7.5
#
# OMD-TCH is not a paper baseline in 2510.22124v2, so its method-specific parameters
# use the repo defaults unless overridden via environment variables.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
METHODS="${METHODS:-eu omd_tch}"
CLASS_LIST="${CLASS_LIST:-0 1 2 3 4 5 6 7 8 9}"
EVAL_SAVE_ROOT="${EVAL_SAVE_ROOT:-evaluation_folder/paper_table1_eupmu_omd_tch}"
PROMPTS_PATH="${PROMPTS_PATH:-prompts/imagenette.csv}"

RUN_SAVE_REAL_IMAGES="${RUN_SAVE_REAL_IMAGES:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_FID="${RUN_FID:-1}"
RUN_CLASSIFY="${RUN_CLASSIFY:-1}"

TRAIN_ALPHA="${TRAIN_ALPHA:-0.01}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
DDIM_STEPS="${DDIM_STEPS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

METHODS="$METHODS" \
CLASS_LIST="$CLASS_LIST" \
EVAL_SAVE_ROOT="$EVAL_SAVE_ROOT" \
PROMPTS_PATH="$PROMPTS_PATH" \
RUN_SAVE_REAL_IMAGES="$RUN_SAVE_REAL_IMAGES" \
RUN_TRAIN="$RUN_TRAIN" \
RUN_GENERATE="$RUN_GENERATE" \
RUN_FID="$RUN_FID" \
RUN_CLASSIFY="$RUN_CLASSIFY" \
TRAIN_ALPHA="$TRAIN_ALPHA" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
TRAIN_EPOCHS="$TRAIN_EPOCHS" \
TRAIN_LR="$TRAIN_LR" \
GUIDANCE_SCALE="$GUIDANCE_SCALE" \
DDIM_STEPS="$DDIM_STEPS" \
IMAGE_SIZE="$IMAGE_SIZE" \
FID_SCRIPT="eval-scripts/compute-fid-per-class.py" \
bash run_table1_sd.sh

"$PYTHON_BIN" eval-scripts/collect-table1-results.py \
  --evaluation_root "$EVAL_SAVE_ROOT" \
  --prompts_path "$PROMPTS_PATH" \
  --methods "$METHODS" \
  --class_ids "$CLASS_LIST" \
  --output_csv "$EVAL_SAVE_ROOT/table1_eupmu_omd_tch.csv" \
  --output_md "$EVAL_SAVE_ROOT/table1_eupmu_omd_tch.md"
