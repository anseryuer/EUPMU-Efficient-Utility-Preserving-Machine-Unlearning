
save_dir=output
arch=resnet18
dataset=cifar10
origin_model_path=pretrained_models/${arch}/${dataset}/model_SA_best.pth.tar
gpu_id=0

python -u main_forget.py \
        --arch ${arch} \
        --dataset ${dataset} \
        --save_dir ${save_dir} \
        --mask ${origin_model_path} \
        --unlearn retrain \
        --num_indexes_to_replace 4500 \
        --unlearn_epochs 160 \
        --unlearn_lr 0.1 \
        --gpu ${gpu_id}