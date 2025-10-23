#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 7:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1



source activate ldm


python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-mask-class_9-method_full-alpha_0.5-epoch_5-lr_1e-05" \
        --device 'cuda:0'