#!/usr/bin/env bash

# Retrain baseline on TinyImagenet forgetting 50% of training data (50000 samples)
python -u main_forget.py \
  --arch resnet18 \
  --dataset TinyImagenet \
  --data "../data/tiny-imagenet-200" \
  --unlearn retrain \
  --num_indexes_to_replace 50000 \
  --unlearn_epochs 182 \
  --unlearn_lr 0.1 \
  --mask pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar \
  --save_dir output/TinyImagenet/retrain_50 \
  "$@"
