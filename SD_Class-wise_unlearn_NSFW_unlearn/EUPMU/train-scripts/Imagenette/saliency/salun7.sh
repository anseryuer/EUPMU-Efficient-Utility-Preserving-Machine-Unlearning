#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1

source activate ldm

python train-scripts/random_label.py \
        --train_method full \
        --alpha 0.5 \
        --lr 1e-5 \
        --epochs 5  \
        --class_to_forget 7 \
        --mask_path 'mask/7/with_0.5.pt' \
        --device '0' \
        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
        --batch_size 6

