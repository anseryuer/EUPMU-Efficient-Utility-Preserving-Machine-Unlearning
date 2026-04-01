#!/bin/bash

for class in {0..9}
do
    echo "Running for class_to_forget = $class"
    python train-scripts/random_label_eu.py --train_method full --alpha 0.01 --lr 1e-5 --epochs 5 --class_to_forget $class --device '0' --mtl --mtl_method "eu" --batch_size 8
    echo "Finished class $class"
    echo "-------------------"
    sleep 3
done

echo "All runs completed."