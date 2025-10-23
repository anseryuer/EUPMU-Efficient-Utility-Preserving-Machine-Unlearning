#!/usr/bin/env bash
python -u main_random.py \
	--arch resnet18 \
	--dataset cifar10 \
	--unlearn RL \
	--unlearn_epochs 5 \
	--unlearn_lr 5e-3 \
	--num_indexes_to_replace 13500 \
	--mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar \
	--save_dir output \
	--gpu 0 \
	--mtl \
	--mtl_method eu \
	--eu_w_lr 1 \
	--eu_error 0.01