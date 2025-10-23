#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1


#source activate ldm



python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_0-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_1-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_2-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_3-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_4-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_5-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_6-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_7-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_8-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'
python eval-scripts/generate-images.py \
        --prompts_path 'prompts/imagenette.csv' \
        --save_path 'evaluation_folder/' \
        --model_name "compvis-cl-class_9-method_full-alpha_0.01-epoch_5-lr_1e-05" \
        --device 'cuda:0'

#python eval-scripts/generate-images.py \
#        --prompts_path 'prompts/tinyimagenet.csv' \
#        --save_path 'evaluation_folder/' \
#        --model_name "tinyimagenet_gpscagrad_times100-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#        --device 'cuda:0'