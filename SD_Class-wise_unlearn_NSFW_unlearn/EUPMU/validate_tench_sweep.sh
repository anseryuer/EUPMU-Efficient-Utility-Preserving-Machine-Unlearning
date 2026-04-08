#!/bin/bash
# Validate in three stages for class 0 (Tench)

# Base parameters
CLASS=0
METHOD="full"
EPOCHS=5
PROMPTS="prompts/imagenette.csv"

# Array of configurations to test: "LR W_LR ERR"
CONFIGS=(
    "1e-05 10.0 1"
    "1e-05 10.0 0.1"
    "1e-06 10.0 1.0"
    "1e-06 5.0 0.1"
    "1e-06 1.0 0.05"
)

for CONFIG in "${CONFIGS[@]}"; do
    read -r LR W_LR ERR <<< "$CONFIG"

    MODEL_NAME="compvis-cl-class_${CLASS}-method_${METHOD}-epoch_${EPOCHS}-lr_${LR}-mtl_eu-w_lr_${W_LR}-err_${ERR}"

    echo "================================================="
    echo "Running config: LR=$LR, W_LR=$W_LR, ERR=$ERR"
    echo "================================================="
    # Train for 5 epochs
    echo "[1/3] Training..."
    python train-scripts/random_label_eu.py \
        --class_to_forget "$CLASS" \
        --train_method "$METHOD" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --mtl \
        --mtl_method eu \
        --w_lr "$W_LR" \
        --error "$ERR" \
        --wandb

    # Image generation

    echo "[2/3] Generating images..."

    SAVE_PATH="eval_output/${MODEL_NAME}"
    mkdir -p "$SAVE_PATH"

    python eval-scripts/generate-images.py \
        --model_name "$MODEL_NAME" \
        --prompts_path "$PROMPTS" \
        --save_path "eval_output" \
        --num_samples 10

    # Eval the 2 metrics

    echo "[3/3] Evaluating UA, FID"

    python eval-scripts/imageclassify.py \
        --folder_path "$SAVE_PATH" \
        --prompts_path "$PROMPTS"

    echo "[3/3] Evaluating FID..."
    python eval-scripts/compute-fid.py \
        --folder_path "$SAVE_PATH" \
        --class_to_forget "$CLASS"

    echo "Done!"

    echo "================================================="
    echo "Finished config: LR=$LR, W_LR=$W_LR, ERR=$ERR"
    echo "================================================="
    echo ""
done