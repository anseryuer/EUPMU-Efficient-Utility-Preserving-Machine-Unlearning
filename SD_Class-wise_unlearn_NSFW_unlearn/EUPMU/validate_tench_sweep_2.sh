#!/bin/bash
# Validate in three stages for class 0 (Tench)

# Base parameters
CLASS=5
METHOD="full"
DEFAULT_EPOCHS=3
PROMPTS="prompts/imagenette.csv"

FID_ENV="${FID_ENV:-fid-eval}"
TRAIN_ENV="${TRAIN_ENV:-eupmu-h200}"

# Array of configurations to test: "LR W_LR ERR ALPHA [EPOCHS]"
    # "1e-05 10.0 1.0 0.01 5" 0 1.4626
    # "1e-05 10.0 0.5 0.01 5" 0 1.08
    # "1e-05 10.0 0.5 0.01 3" 0 1.1823
    # "1e-05 10.0 0.3 0.01 5" 0 1.4063
    # "1e-05 10.0 0.3 0.01 3" 0 0.9737
    # "1e-05 10.0 0.8 0.01 5" 0 1.3675
    # "1e-05 10.0 0.4 0.01 5" 0 1.1546
    # "1e-05 10.0 0.6 0.01 5" 0 1.2165
    # "1e-05 10.0 0.7 0.01 5" 0 1.2642
    # class 5, 3 epochs:
    # "1e-05 10.0 0.3 0.01 3" 1.6413
    # "1e-05 5.0 0.15 0.01 3" 1.577
    # "1e-05 3.0 0.1 0.01 3" 2.0406
    # "1e-05 10.0 0.2 0.01 3" 1.5034
    # "8e-06 10.0 0.2 0.01 3" 1.7159
    # "8e-06 10.0 0.15 0.01 3" 1.4152
    # "1e-05 10.0 0.15 0.01 3" 2.5724
    # "1e-05 10.0 0.12 0.01 3" 1.5720
CONFIGS=(
    "5e-06 10.0 0.15 0.01 3"
    "5e-06 10.0 0.12 0.01 3"
)

for CONFIG in "${CONFIGS[@]}"; do
    read -r LR W_LR ERR ALPHA EPOCHS <<< "$CONFIG"
    EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"

    MODEL_NAME="compvis-cl-class_${CLASS}-method_${METHOD}-epoch_${EPOCHS}-lr_${LR}-mtl_eu-w_lr_${W_LR}-err_${ERR}"
    if [[ "$ALPHA" != "1.0" ]]; then
        MODEL_NAME="${MODEL_NAME}-alpha_${ALPHA}"
    fi

    echo "================================================="
    echo "Running config: LR=$LR, W_LR=$W_LR, ERR=$ERR, ALPHA=$ALPHA, EPOCHS=$EPOCHS"
    echo "================================================="
    # Train for the configured number of epochs
    echo "[1/3] Training..."
    python train-scripts/random_label_eu.py \
        --class_to_forget "$CLASS" \
        --train_method "$METHOD" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --alpha "$ALPHA" \
        --mtl \
        --mtl_method eu \
        --w_lr "$W_LR" \
        --error "$ERR" \
        --wandb

    # Image generation

    echo "[2/3] Generating images..."

    SAVE_PATH="eval_output/${MODEL_NAME}"
    mkdir -p "$SAVE_PATH"

    conda run -n "${TRAIN_ENV}" python eval-scripts/generate-images.py \
        --model_name "$MODEL_NAME" \
        --prompts_path "$PROMPTS" \
        --save_path "eval_output" \
        --num_samples 10

    # Eval the 2 metrics

    echo "[3/3] Evaluating UA, FID"

    conda run -n "${TRAIN_ENV}" python eval-scripts/imageclassify.py \
        --folder_path "$SAVE_PATH" \
        --prompts_path "$PROMPTS"

    # echo "[3/3] Evaluating FID..."
    conda run -n "${FID_ENV}" python eval-scripts/compute-fid.py \
        --folder_path "${SAVE_PATH}" \
        --class_to_forget "${CLASS}"

    # echo "Done!"

    echo "================================================="
    echo "Finished config: LR=$LR, W_LR=$W_LR, ERR=$ERR, ALPHA=$ALPHA, EPOCHS=$EPOCHS"
    echo "================================================="
    echo ""
done
