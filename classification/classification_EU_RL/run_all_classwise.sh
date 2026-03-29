#!/usr/bin/env bash
set -euo pipefail

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
CLASS_TO_REPLACE="${CLASS_TO_REPLACE:-0}"
GPU="${GPU:-0}"
SAVE_DIR="${SAVE_DIR:-output}"
MASK="${MASK:-pretrained_models/resnet18/cifar10/model_SA_best.pth.tar}"
SALUN_MASK_PATH="${SALUN_MASK_PATH:-saliency_maps/resnet18/cifar10/forget_10.0%/with_0.5.pt}"
RUN_SALUN_MASK_GEN="${RUN_SALUN_MASK_GEN:-0}"
RUN_SALUN="${RUN_SALUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_ENTITY_TAG="${WANDB_ENTITY_TAG:-None}"
FORGET_TAG="${FORGET_TAG:-forget_10.0%}"
RUN_OMD_ABLATION_GRID="${RUN_OMD_ABLATION_GRID:-0}"
OMD_ABLATION_ETAS="${OMD_ABLATION_ETAS:-0.01 0.03 0.1}"

run() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
  "$@"
}

run_if_needed() {
  local result_path="$1"
  shift
  if [[ -f "$result_path" ]]; then
    echo
    echo "[skip] Found existing result: $result_path"
    return 0
  fi
  run "$@"
}

run_if_needed_any() {
  local result_path="$1"
  local alias_path="$2"
  shift 2
  if [[ -f "$result_path" ]]; then
    echo
    echo "[skip] Found existing result: $result_path"
    return 0
  fi
  if [[ -n "$alias_path" && -f "$alias_path" ]]; then
    echo
    echo "[skip] Found equivalent existing result: $alias_path"
    return 0
  fi
  run "$@"
}

method_result_path() {
  local method="$1"
  printf '%s/%s/%s/%s/%s/%s/evaluation_result.json' \
    "$SAVE_DIR" "$ARCH" "$DATASET" "$FORGET_TAG" "$method" "$WANDB_ENTITY_TAG"
}

mtl_result_path() {
  local mtl_method="$1"
  printf '%s/%s/%s/%s/RL/%s/%s/evaluation_result.json' \
    "$SAVE_DIR" "$ARCH" "$DATASET" "$FORGET_TAG" "$mtl_method" "$WANDB_ENTITY_TAG"
}

plain_rl_result_path() {
  printf '%s/%s/%s/%s/RL/%s/evaluation_result.json' \
    "$SAVE_DIR" "$ARCH" "$DATASET" "$FORGET_TAG" "$WANDB_ENTITY_TAG"
}

BASE_FORGET_ARGS=(
  --arch "$ARCH"
  --dataset "$DATASET"
  --class_to_replace "$CLASS_TO_REPLACE"
  --mask "$MASK"
  --save_dir "$SAVE_DIR"
  --gpu "$GPU"
)

BASE_RANDOM_ARGS=(
  --arch "$ARCH"
  --dataset "$DATASET"
  --class_to_replace "$CLASS_TO_REPLACE"
  --mask "$MASK"
  --save_dir "$SAVE_DIR"
  --gpu "$GPU"
)

run_if_needed "$(method_result_path retrain)" \
  "$PYTHON_BIN" -u main_forget.py \
  "${BASE_FORGET_ARGS[@]}" \
  --unlearn retrain \
  --unlearn_epochs 160 \
  --unlearn_lr 0.1

run_if_needed "$(method_result_path FT)" \
  "$PYTHON_BIN" -u main_forget.py \
  "${BASE_FORGET_ARGS[@]}" \
  --unlearn FT \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-2

run_if_needed "$(method_result_path GA)" \
  "$PYTHON_BIN" -u main_forget.py \
  "${BASE_FORGET_ARGS[@]}" \
  --unlearn GA \
  --unlearn_epochs 5 \
  --unlearn_lr 3e-4

run_if_needed "$(method_result_path wfisher)" \
  "$PYTHON_BIN" -u main_forget.py \
  "${BASE_FORGET_ARGS[@]}" \
  --unlearn wfisher \
  --unlearn_epochs 5 \
  --alpha 2

run_if_needed "$(method_result_path FT_prune)" \
  "$PYTHON_BIN" -u main_forget.py \
  "${BASE_FORGET_ARGS[@]}" \
  --unlearn FT_prune \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-2 \
  --alpha 1e-4

run_if_needed "$(plain_rl_result_path)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3

run_if_needed "$(mtl_result_path famo)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method famo

run_if_needed "$(mtl_result_path igs)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method igs

run_if_needed "$(mtl_result_path eu)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method eu \
  --eu_w_lr 1 \
  --eu_error 0.01

run_if_needed "$(mtl_result_path eu_fast)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 2e-3 \
  --mtl \
  --mtl_method eu_fast \
  --eu_w_lr 1 \
  --eu_error 0.01

run_if_needed "$(mtl_result_path gdr_gma)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 10 \
  --unlearn_lr 0.1 \
  --mtl \
  --mtl_method gdr_gma

run_if_needed "$(mtl_result_path chebyshev)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method chebyshev \
  --cheby_retain_weight 1.0 \
  --cheby_forget_weight 1.0 \
  --cheby_retain_ref 0.0 \
  --cheby_forget_ref 0.0 \
  --cheby_rho 1e-3

run_if_needed "$(mtl_result_path omd_tch)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method omd_tch \
  --omd_tch_retain_weight 1.0 \
  --omd_tch_forget_weight 1.0 \
  --omd_tch_retain_ref 0.0 \
  --omd_tch_forget_ref 0.0 \
  --omd_tch_eta 0.1 \
  --omd_tch_rho 0.0

run_if_needed "$(mtl_result_path afl)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method afl \
  --omd_tch_retain_weight 1.0 \
  --omd_tch_forget_weight 1.0 \
  --omd_tch_retain_ref 0.0 \
  --omd_tch_forget_ref 0.0 \
  --omd_tch_eta 0.1 \
  --omd_tch_rho 0.0

run_if_needed_any "$(mtl_result_path afleg)" "$(mtl_result_path omd_tch)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method afleg \
  --omd_tch_retain_weight 1.0 \
  --omd_tch_forget_weight 1.0 \
  --omd_tch_retain_ref 0.0 \
  --omd_tch_forget_ref 0.0 \
  --omd_tch_eta 0.1 \
  --omd_tch_rho 0.0

run_if_needed "$(mtl_result_path ada_afleg)" \
  "$PYTHON_BIN" -u main_random.py \
  "${BASE_RANDOM_ARGS[@]}" \
  --unlearn RL \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-3 \
  --mtl \
  --mtl_method ada_afleg \
  --omd_tch_retain_weight 1.0 \
  --omd_tch_forget_weight 1.0 \
  --omd_tch_retain_ref 0.0 \
  --omd_tch_forget_ref 0.0 \
  --omd_tch_eta 0.1 \
  --omd_tch_rho 0.0


if [[ "$RUN_OMD_ABLATION_GRID" == "1" ]]; then
  for ETA in $OMD_ABLATION_ETAS; do
    ETA_TAG="eta_${ETA//./p}"

    run_if_needed "${SAVE_DIR}/${ARCH}/${DATASET}/${FORGET_TAG}/RL/omd_tch_eg/afleg_${ETA_TAG}/evaluation_result.json"       "$PYTHON_BIN" -u main_random.py       "${BASE_RANDOM_ARGS[@]}"       --save_dir "$SAVE_DIR"       --unlearn RL       --unlearn_epochs 5       --unlearn_lr 1e-3       --mtl       --mtl_method omd_tch_eg       --wandb_entity "afleg_${ETA_TAG}"       --omd_tch_retain_weight 1.0       --omd_tch_forget_weight 1.0       --omd_tch_retain_ref 0.0       --omd_tch_forget_ref 0.0       --omd_tch_eta "$ETA"       --omd_tch_rho 0.0

    run_if_needed "${SAVE_DIR}/${ARCH}/${DATASET}/${FORGET_TAG}/RL/omd_tch_pgd/afl_${ETA_TAG}/evaluation_result.json"       "$PYTHON_BIN" -u main_random.py       "${BASE_RANDOM_ARGS[@]}"       --save_dir "$SAVE_DIR"       --unlearn RL       --unlearn_epochs 5       --unlearn_lr 1e-3       --mtl       --mtl_method omd_tch_pgd       --wandb_entity "afl_${ETA_TAG}"       --omd_tch_retain_weight 1.0       --omd_tch_forget_weight 1.0       --omd_tch_retain_ref 0.0       --omd_tch_forget_ref 0.0       --omd_tch_eta "$ETA"       --omd_tch_rho 0.0

    run_if_needed "${SAVE_DIR}/${ARCH}/${DATASET}/${FORGET_TAG}/RL/ada_omd_tch_eg/ada_afleg_${ETA_TAG}/evaluation_result.json"       "$PYTHON_BIN" -u main_random.py       "${BASE_RANDOM_ARGS[@]}"       --save_dir "$SAVE_DIR"       --unlearn RL       --unlearn_epochs 5       --unlearn_lr 1e-3       --mtl       --mtl_method ada_omd_tch_eg       --wandb_entity "ada_afleg_${ETA_TAG}"       --omd_tch_retain_weight 1.0       --omd_tch_forget_weight 1.0       --omd_tch_retain_ref 0.0       --omd_tch_forget_ref 0.0       --omd_tch_eta "$ETA"       --omd_tch_rho 0.0
  done
fi

if [[ "$RUN_SALUN_MASK_GEN" == "1" ]]; then
  run "$PYTHON_BIN" generate_mask.py \
    --arch "$ARCH" \
    --dataset "$DATASET" \
    --class_to_replace "$CLASS_TO_REPLACE" \
    --mask "$MASK" \
    --save_dir saliency_maps/
fi

if [[ "$RUN_SALUN" == "1" ]]; then
  echo
  echo "[warn] SalUn-style runs share the plain RL save path in this repo, so skip detection is not reliable."

  run "$PYTHON_BIN" -u main_random.py \
    "${BASE_RANDOM_ARGS[@]}" \
    --unlearn RL \
    --unlearn_epochs 10 \
    --unlearn_lr 0.01 \
    --path "$SALUN_MASK_PATH"

  run "$PYTHON_BIN" -u main_random.py \
    "${BASE_RANDOM_ARGS[@]}" \
    --unlearn RL_proximal \
    --unlearn_epochs 10 \
    --unlearn_lr 0.01 \
    --path "$SALUN_MASK_PATH" \
    --mask_ratio 0.5
fi
