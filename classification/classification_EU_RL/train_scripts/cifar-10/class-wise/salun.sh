#!/usr/bin/env bash
python -u main_random.py \
  --unlearn RL \
  --unlearn_epochs 10 \
  --unlearn_lr 0.01 \
  --class_to_replace 0 \
  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar \
  --save_dir output \
  --path saliency_maps/resnet18/cifar10/forget_10.0%/with_0.5.pt
