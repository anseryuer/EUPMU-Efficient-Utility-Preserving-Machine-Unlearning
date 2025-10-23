#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1


source activate ldm



python train-scripts/generate_mask_tinyimagenet.py \
        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
        --classes 0 \
        --device '0' \
        --dataset tinyimagenet \
        --data "../data/tiny-imagenet-200"


