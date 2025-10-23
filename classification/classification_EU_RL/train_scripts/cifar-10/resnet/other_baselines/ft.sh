
origin_model_path=pretrained_models/0model_SA_best.pth.tar
save_dir=output
gpu_id=0


source activate mu

python -u main_forget.py \
        --arch resnet18 \
        --dataset cifar10 \
        --save_dir ${save_dir} \
        --mask ${origin_model_path} \
        --unlearn FT \
        --num_indexes_to_replace 4500 \
        --unlearn_lr 0.01 \
        --unlearn_epochs 10 \
        --gpu ${gpu_id}