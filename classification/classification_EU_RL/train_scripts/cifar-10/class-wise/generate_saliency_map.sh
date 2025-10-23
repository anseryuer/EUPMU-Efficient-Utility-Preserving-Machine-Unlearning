#!/usr/bin/env bash
python generate_mask.py \
  --arch resnet18 \
  --dataset cifar10 \
  --class_to_replace 0 \
  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar \
  --save_dir saliency_maps/
