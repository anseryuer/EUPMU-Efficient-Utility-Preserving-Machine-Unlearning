

arch=resnet18
dataset=cifar10
origin_model_path=pretrained_models/${arch}/${dataset}/model_SA_best.pth.tar
save_dir=output
gpu_id=0


source activate mu

python -u main_forget.py \
        --arch ${arch} \
        --dataset ${dataset} \
        --save_dir ${save_dir} \
        --mask ${origin_model_path} \
        --unlearn boundary_expanding \
        --num_indexes_to_replace 4500 \
        --unlearn_lr 4e-5 \
        --unlearn_epochs 10 \
        --gpu ${gpu_id}

python -u main_forget.py \
        --arch resnet18 \
        --dataset cifar10 \
        --save_dir output \
        --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar \
        --unlearn boundary_expanding \
        --num_indexes_to_replace 13500 \
        --unlearn_lr 1e-4 \
        --unlearn_epochs 10

