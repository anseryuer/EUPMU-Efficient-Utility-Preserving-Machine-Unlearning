

arch=resnet18
dataset=cifar100
gpu_id=0


# source activate mu

python main_train.py \
        --arch ${arch} \
        --dataset ${dataset} \
        --lr 0.1 \
        --epochs 182 \
        --save_dir pretrained_models/${arch}/${dataset}/ \
        --gpu ${gpu_id}