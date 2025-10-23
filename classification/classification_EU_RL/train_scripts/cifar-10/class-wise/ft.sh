#!/usr/bin/env bash
python -u main_forget.py \
  --arch resnet18 \
  --dataset cifar10 \
  --unlearn FT \
  --unlearn_epochs 5 \
  --unlearn_lr 1e-2 \
  --class_to_replace 0 \
  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar \
  --save_dir output \
  --gpu 0
