#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_METHOD="${TRAIN_METHOD:-full}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_EU="${RUN_EU:-1}"
RUN_OMD_TCH="${RUN_OMD_TCH:-1}"
RUN_OMD_TCH_PGD="${RUN_OMD_TCH_PGD:-1}"
RUN_ESD="${RUN_ESD:-1}"
RUN_MASKED_BASELINE="${RUN_MASKED_BASELINE:-1}"

BASELINE_ALPHA="${BASELINE_ALPHA:-0.1}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-8}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-1}"
BASELINE_LR="${BASELINE_LR:-1e-5}"

MTL_ALPHA="${MTL_ALPHA:-0.1}"
MTL_BATCH_SIZE="${MTL_BATCH_SIZE:-8}"
MTL_EPOCHS="${MTL_EPOCHS:-1}"
MTL_LR="${MTL_LR:-1e-5}"

ESD_PROMPT="${ESD_PROMPT:-nudity}"
ESD_START_GUIDANCE="${ESD_START_GUIDANCE:-3}"
ESD_NEGATIVE_GUIDANCE="${ESD_NEGATIVE_GUIDANCE:-1}"
ESD_ITERATIONS="${ESD_ITERATIONS:-1000}"
ESD_LR="${ESD_LR:-1e-5}"

MASK_C_GUIDANCE="${MASK_C_GUIDANCE:-7.5}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-8}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
MASK_LR="${MASK_LR:-1e-5}"
MASK_PATH="${MASK_PATH:-mask/nude_0.5.pt}"

CKPT_PATH="${CKPT_PATH:-models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-configs/stable-diffusion/v1-inference.yaml}"
DIFFUSERS_CONFIG_PATH="${DIFFUSERS_CONFIG_PATH:-diffusers_unet_config.json}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/nsfw_runs/${RUN_STAMP}}"
mkdir -p "$LOG_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -d data/nsfw || ! -d data/not-nsfw ]]; then
  echo "Expected ./data/nsfw and ./data/not-nsfw to exist before launching runs." >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

declare -a RUNNING_PIDS=()
declare -a RUNNING_GPU_IDS=()
declare -a RUNNING_NAMES=()
declare -a REQUESTED_JOBS=()

refresh_running_jobs() {
  local -a next_pids=()
  local -a next_gpu_ids=()
  local -a next_names=()
  local i

  for i in "${!RUNNING_PIDS[@]}"; do
    local pid="${RUNNING_PIDS[$i]}"
    local gpu_id="${RUNNING_GPU_IDS[$i]}"
    local name="${RUNNING_NAMES[$i]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      next_pids+=("$pid")
      next_gpu_ids+=("$gpu_id")
      next_names+=("$name")
    else
      wait "$pid" || {
        echo "Job ${name} failed on GPU ${gpu_id}. Check ${LOG_DIR}/${name}.log" >&2
        exit 1
      }
      echo "Completed ${name} on GPU ${gpu_id}"
    fi
  done

  RUNNING_PIDS=("${next_pids[@]}")
  RUNNING_GPU_IDS=("${next_gpu_ids[@]}")
  RUNNING_NAMES=("${next_names[@]}")
}

gpu_is_reserved() {
  local gpu_id="$1"
  local current
  for current in "${RUNNING_GPU_IDS[@]:-}"; do
    if [[ "$current" == "$gpu_id" ]]; then
      return 0
    fi
  done
  return 1
}

find_available_gpu() {
  local line gpu_id free_mb
  while IFS=',' read -r gpu_id free_mb; do
    gpu_id="${gpu_id//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    [[ -z "$gpu_id" || -z "$free_mb" ]] && continue
    if (( free_mb > MIN_FREE_MB )) && ! gpu_is_reserved "$gpu_id"; then
      printf '%s\n' "$gpu_id"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

wait_for_gpu() {
  local gpu_id
  while true; do
    refresh_running_jobs
    if gpu_id="$(find_available_gpu)"; then
      printf '%s\n' "$gpu_id"
      return 0
    fi
    echo "No GPU currently has more than ${MIN_FREE_MB} MB free. Sleeping ${POLL_INTERVAL_SECONDS}s..."
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

launch_background_job() {
  local name="$1"
  local cmd="$2"
  local gpu_id
  gpu_id="$(wait_for_gpu)"

  echo "Launching ${name} on GPU ${gpu_id}"
  echo "Command: ${cmd}"
  CUDA_VISIBLE_DEVICES="$gpu_id" bash -lc "$cmd" >"${LOG_DIR}/${name}.log" 2>&1 &
  RUNNING_PIDS+=("$!")
  RUNNING_GPU_IDS+=("$gpu_id")
  RUNNING_NAMES+=("$name")
}

run_blocking_job() {
  local name="$1"
  local cmd="$2"
  local gpu_id
  gpu_id="$(wait_for_gpu)"

  echo "Launching ${name} on GPU ${gpu_id}"
  echo "Command: ${cmd}"
  CUDA_VISIBLE_DEVICES="$gpu_id" bash -lc "$cmd" >"${LOG_DIR}/${name}.log" 2>&1
  echo "Completed ${name} on GPU ${gpu_id}"
}

wait_for_all_jobs() {
  while ((${#RUNNING_PIDS[@]} > 0)); do
    refresh_running_jobs
    if ((${#RUNNING_PIDS[@]} > 0)); then
      sleep 5
    fi
  done
}

register_requested_job() {
  local name="$1"
  REQUESTED_JOBS+=("$name")
}

echo "Logs will be written to ${LOG_DIR}"
echo "Only GPUs with more than ${MIN_FREE_MB} MB free memory will be used."

if [[ "$RUN_MASKED_BASELINE" == "1" ]]; then
  register_requested_job "mask_generation"
  register_requested_job "baseline_masked"
fi
if [[ "$RUN_BASELINE" == "1" ]]; then
  register_requested_job "baseline"
fi
if [[ "$RUN_EU" == "1" ]]; then
  register_requested_job "mtl_eu"
fi
if [[ "$RUN_OMD_TCH" == "1" ]]; then
  register_requested_job "mtl_omd_tch"
fi
if [[ "$RUN_OMD_TCH_PGD" == "1" ]]; then
  register_requested_job "mtl_omd_tch_pgd"
fi
if [[ "$RUN_ESD" == "1" ]]; then
  register_requested_job "esd"
fi

if ((${#REQUESTED_JOBS[@]} == 0)); then
  echo "No jobs are enabled. Set one or more RUN_* flags to 1." >&2
  exit 1
fi

echo "Requested jobs:"
for job_name in "${REQUESTED_JOBS[@]}"; do
  echo "  - ${job_name}"
done

if [[ "$RUN_MASKED_BASELINE" == "1" ]]; then
  run_blocking_job \
    "mask_generation" \
    "\"${PYTHON_BIN}\" train-scripts/generate_mask.py \
      --nsfw True \
      --device 0 \
      --c_guidance ${MASK_C_GUIDANCE} \
      --batch_size ${MASK_BATCH_SIZE} \
      --epochs ${MASK_EPOCHS} \
      --lr ${MASK_LR} \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH}"

  if [[ ! -f "$MASK_PATH" ]]; then
    echo "Masked baseline requested, but mask file was not created at ${MASK_PATH}" >&2
    exit 1
  fi
fi

if [[ "$RUN_BASELINE" == "1" ]]; then
  launch_background_job \
    "baseline" \
    "\"${PYTHON_BIN}\" train-scripts/nsfw_removal.py \
      --train_method ${TRAIN_METHOD} \
      --device 0 \
      --alpha ${BASELINE_ALPHA} \
      --batch_size ${BASELINE_BATCH_SIZE} \
      --epochs ${BASELINE_EPOCHS} \
      --lr ${BASELINE_LR} \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH}"
fi

if [[ "$RUN_EU" == "1" ]]; then
  launch_background_job \
    "mtl_eu" \
    "\"${PYTHON_BIN}\" train-scripts/nsfw_removal_EU.py \
      --train_method ${TRAIN_METHOD} \
      --device 0 \
      --alpha ${MTL_ALPHA} \
      --batch_size ${MTL_BATCH_SIZE} \
      --epochs ${MTL_EPOCHS} \
      --lr ${MTL_LR} \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH} \
      --mtl \
      --mtl_method eu"
fi

if [[ "$RUN_OMD_TCH" == "1" ]]; then
  launch_background_job \
    "mtl_omd_tch" \
    "\"${PYTHON_BIN}\" train-scripts/nsfw_removal_EU.py \
      --train_method ${TRAIN_METHOD} \
      --device 0 \
      --alpha ${MTL_ALPHA} \
      --batch_size ${MTL_BATCH_SIZE} \
      --epochs ${MTL_EPOCHS} \
      --lr ${MTL_LR} \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH} \
      --mtl \
      --mtl_method omd_tch"
fi

if [[ "$RUN_OMD_TCH_PGD" == "1" ]]; then
  launch_background_job \
    "mtl_omd_tch_pgd" \
    "\"${PYTHON_BIN}\" train-scripts/nsfw_removal_EU.py \
      --train_method ${TRAIN_METHOD} \
      --device 0 \
      --alpha ${MTL_ALPHA} \
      --batch_size ${MTL_BATCH_SIZE} \
      --epochs ${MTL_EPOCHS} \
      --lr ${MTL_LR} \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH} \
      --mtl \
      --mtl_method omd_tch_pgd"
fi

if [[ "$RUN_ESD" == "1" ]]; then
  launch_background_job \
    "esd" \
    "\"${PYTHON_BIN}\" train-scripts/train-esd.py \
      --prompt ${ESD_PROMPT} \
      --train_method ${TRAIN_METHOD} \
      --start_guidance ${ESD_START_GUIDANCE} \
      --negative_guidance ${ESD_NEGATIVE_GUIDANCE} \
      --iterations ${ESD_ITERATIONS} \
      --lr ${ESD_LR} \
      --devices 0,0 \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH}"
fi

if [[ "$RUN_MASKED_BASELINE" == "1" ]]; then
  launch_background_job \
    "baseline_masked" \
    "\"${PYTHON_BIN}\" train-scripts/nsfw_removal.py \
      --train_method ${TRAIN_METHOD} \
      --device 0 \
      --alpha ${BASELINE_ALPHA} \
      --batch_size ${BASELINE_BATCH_SIZE} \
      --epochs ${BASELINE_EPOCHS} \
      --lr ${BASELINE_LR} \
      --ckpt_path ${CKPT_PATH} \
      --config_path ${CONFIG_PATH} \
      --diffusers_config_path ${DIFFUSERS_CONFIG_PATH} \
      --mask_path ${MASK_PATH}"
fi

wait_for_all_jobs

echo "All requested NSFW runs completed."
echo "Logs: ${LOG_DIR}"
