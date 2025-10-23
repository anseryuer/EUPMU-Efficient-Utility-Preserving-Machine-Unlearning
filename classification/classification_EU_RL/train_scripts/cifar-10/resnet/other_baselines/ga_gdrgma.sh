
origin_model_path=save_models/model_SA_best.pth.tar
save_dir=output
gpu_id=0


python -u main_forget.py \
        --arch resnet18 \
        --dataset cifar10 \
        --save_dir ${save_dir} \
        --mask ${origin_model_path} \
        --unlearn GA_gdr_gma \
        --num_indexes_to_replace 4500 \
        --unlearn_lr 0.0001 \
        --unlearn_epochs 5 \
        --gpu ${gpu_id} \
        --mtl \
        --mtl_method gdr_gma
