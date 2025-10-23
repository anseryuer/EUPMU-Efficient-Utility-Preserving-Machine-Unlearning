#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 7:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1

#source activate ldm


python eval-scripts/compute-fid.py \
        --class_to_forget 0 \
        --folder_path "evaluation_folder/compvis-cl-class_0-method_full-alpha_0.01-epoch_5-lr_1e-05" \



python eval-scripts/compute-fid.py \
        --class_to_forget 1 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_1-method_full-alpha_0.5-epoch_5-lr_1e-06" \




python eval-scripts/compute-fid.py \
        --class_to_forget 2 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_2-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 3 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_3-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 4 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_4-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 5 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_5-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 6 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_6-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 7 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_7-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 8 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_8-method_full-alpha_0.5-epoch_5-lr_1e-06" \


python eval-scripts/compute-fid.py \
        --class_to_forget 9 \
        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_9-method_full-alpha_0.5-epoch_5-lr_1e-06" \




#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/gpscagrad_times200-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/gpscagrad_times150-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_0.0001" \
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-06" \
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/gpscagrad_times50-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/gpscagrad_times100-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05" \






#
#
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 0 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_0-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 1 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_1-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 2 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_2-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 3 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_3-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 4 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_4-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 5 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_5-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 6 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_6-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 7 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_7-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 8 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_8-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
#
#python eval-scripts/compute-fid.py \
#        --class_to_forget 9 \
#        --folder_path "evaluation_folder/puregps_times100-compvis-cl-class_9-method_full-alpha_0.5-epoch_5-lr_1e-05" \
#
