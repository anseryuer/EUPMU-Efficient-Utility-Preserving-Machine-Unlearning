#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1


source activate ldm



python train-scripts/generate_mask.py \
        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
        --classes 9 \
        --device '0'


