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
        --class_to_forget 0 \
        --mask_path 'mask/0/with_0.5.pt' \
        --device '0' \
        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
        --batch_size 8

#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 1 \
#        --mask_path 'mask/1/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
#
#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 2 \
#        --mask_path 'mask/2/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
#
#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 3 \
#        --mask_path 'mask/3/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
#
#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 4 \
#        --mask_path 'mask/4/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
#
#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 5 \
#        --mask_path 'mask/5/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
#
#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 6 \
#        --mask_path 'mask/6/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
#
#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 7 \
#        --mask_path 'mask/7/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8

#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 8 \
#        --mask_path 'mask/8/with_0.5.pt' \
#        --device '1' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8

#python train-scripts/random_label.py \
#        --train_method full \
#        --alpha 0.5 \
#        --lr 1e-5 \
#        --epochs 5  \
#        --class_to_forget 9 \
#        --mask_path 'mask/9/with_0.5.pt' \
#        --device '0' \
#        --ckpt_path './ldm/sd-v1-4-full-ema.ckpt' \
#        --batch_size 8
