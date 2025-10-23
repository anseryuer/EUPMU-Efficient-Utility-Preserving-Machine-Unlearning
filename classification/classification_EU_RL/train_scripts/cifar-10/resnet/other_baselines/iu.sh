
origin_model_path=pretrained_models/resnet18/cifar10/model_SA_best.pth.tar
save_dir=output
alpha=0.2
gpu_id=0



source activate mu

python -u main_forget.py \
        --arch resnet18 \
        --dataset cifar10 \
        --save_dir ${save_dir} \
        --mask ${origin_model_path} \
        --unlearn wfisher \
        --num_indexes_to_replace 4500 \
        --unlearn_epochs 5 \
        --unlearn_lr 0.01 \
        --alpha ${alpha} \
        --gpu ${gpu_id}