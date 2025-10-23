#!/usr/bin/env bash
python -u main_random.py \
  --save_dir output \
  --unlearn RL \
  --class_to_replace 0 \
  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar \
  --unlearn_lr 0.1 \
  --unlearn_epochs 10 \
  --mtl_method gdr_gma
