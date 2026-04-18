#!/usr/bin/env bash
set -euo pipefail

TRAIN_ENV="${TRAIN_ENV:-eupmu-h200}"
FID_ENV="${FID_ENV:-fid-eval}"
# 1e-05 10.0 0.3 0.01 3
METHOD="full"
EPOCHS=3
LR="1e-05"
W_LR="10.0"
ERR="0.3"
ALPHA="0.01"
PROMPTS="prompts/imagenette.csv"
SAVE_ROOT="eval_output"

# Keep this aligned with validate_tench_sweep.sh: num_samples=10 here means
# 100 images per prompt because generate-images.py loops ten times internally.
NUM_SAMPLES=10

# eval_output/compvis-cl-class_1-method_full-epoch_3-lr_1e-05-mtl_eu-w_lr_10.0-err_0.3-alpha_0.01

source /opt/conda/etc/profile.d/conda.sh

for CLASS in {0..9}; do
    MODEL_NAME="compvis-cl-class_${CLASS}-method_${METHOD}-epoch_${EPOCHS}-lr_${LR}-mtl_eu-w_lr_${W_LR}-err_${ERR}"
    if [[ "${ALPHA}" != "1.0" ]]; then
        MODEL_NAME="${MODEL_NAME}-alpha_${ALPHA}"
    fi

    SAVE_PATH="${SAVE_ROOT}/${MODEL_NAME}"
    mkdir -p "${SAVE_PATH}"

    echo "================================================="
    echo "Running class ${CLASS} with ${MODEL_NAME}"
    echo "Training env: ${TRAIN_ENV} | FID env: ${FID_ENV}"
    echo "================================================="

    echo "[1/4] Training..."
    conda run -n "${TRAIN_ENV}" python train-scripts/random_label_eu.py \
        --class_to_forget "${CLASS}" \
        --train_method "${METHOD}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --alpha "${ALPHA}" \
        --mtl \
        --mtl_method eu \
        --w_lr "${W_LR}" \
        --error "${ERR}" \
        --wandb

    echo "[2/4] Generating images..."
    conda run -n "${TRAIN_ENV}" python eval-scripts/generate-images.py \
        --model_name "${MODEL_NAME}" \
        --prompts_path "${PROMPTS}" \
        --save_path "${SAVE_ROOT}" \
        --num_samples "${NUM_SAMPLES}"

    echo "[3/4] Evaluating UA..."
    conda run -n "${TRAIN_ENV}" python eval-scripts/imageclassify.py \
        --folder_path "${SAVE_PATH}" \
        --prompts_path "${PROMPTS}"

    echo "[4/4] Evaluating FID..."
    conda run -n "${FID_ENV}" python eval-scripts/compute-fid.py \
        --folder_path "${SAVE_PATH}" \
        --class_to_forget "${CLASS}"

    echo "Finished class ${CLASS}"
    echo "-------------------------------------------------"
done

echo "All class runs completed (classes 1-9)."
