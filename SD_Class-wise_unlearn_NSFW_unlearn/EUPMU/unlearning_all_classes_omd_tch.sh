#!/bin/bash

for class in {0..9}
do
    echo "Running OMD-TCH (EG) for class_to_forget = $class"
    python train-scripts/random_label_eu.py \
        --train_method full \
        --alpha 0.01 \
        --lr 1e-5 \
        --epochs 5 \
        --batch_size 8 \
        --class_to_forget "$class" \
        --device '0' \
        --mtl \
        --mtl_method "omd_tch"
    echo "Finished class $class"
    echo "-------------------"
    sleep 3
done

echo "All OMD-TCH runs completed."
