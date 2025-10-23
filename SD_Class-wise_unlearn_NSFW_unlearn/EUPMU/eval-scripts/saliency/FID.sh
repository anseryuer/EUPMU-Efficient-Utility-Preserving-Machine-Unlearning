#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 7:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1

source activate ldm




python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 0

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_1-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 1

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_2-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 2


python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_3-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 3

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_4-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 4

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_5-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 5

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_6-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 6

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_7-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 7

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_8-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 8

python eval-scripts/compute-fid.py \
        --folder_path "evaluation_folder/compvis-cl-mask-class_9-method_full-alpha_0.5-epoch_5-lr_1e-05/"\
        --class_to_forget 9
#
