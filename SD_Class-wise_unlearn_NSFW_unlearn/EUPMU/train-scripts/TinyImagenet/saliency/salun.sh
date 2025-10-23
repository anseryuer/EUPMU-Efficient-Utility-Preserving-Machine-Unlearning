#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1

source activate ldm

#python train-scripts/generate_mask_tinyimagenet.py \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --classes 0 \
#        --device '0' \
#        --dataset tinyimagenet \
#        --data "../data/tiny-imagenet-200"

python train-scripts/random_label_tinyimagenet.py \
        --train_method full \
        --alpha 0.5 \
        --lr 1e-5 \
        --epochs 5  \
        --class_to_forget 0 \
        --mask_path 'mask/0/with_0.5.pt' \
        --device '0' \
        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
        --batch_size 8 \
        --dataset tinyimagenet \
        --data "../data/tiny-imagenet-200"

