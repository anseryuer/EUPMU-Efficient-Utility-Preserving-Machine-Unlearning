#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 7:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1





#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05/"\

#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_1-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_2-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_3-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_4-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_5-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_6-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_7-method_full-alpha_0.5-epoch_5-lr_1e-05/"\

#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_8-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#
#python eval-scripts/imageclassify.py \
#          --prompts_path 'prompts/imagenette.csv' \
#          --device "cuda:0" \
#          --folder_path "evaluation_folder/compvis-cl-mask-class_9-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
#





