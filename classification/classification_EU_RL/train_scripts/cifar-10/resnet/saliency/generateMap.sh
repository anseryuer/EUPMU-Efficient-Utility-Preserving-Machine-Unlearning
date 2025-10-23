
arch=resnet18
dataset=cifar10
saliency_map_path=salientmap
origin_model_path=pretrained_models/${arch}/${dataset}/model_SA_best.pth.tar
gpu_id=1





source activate mu
python generate_mask.py \
        --arch ${arch} \
        --dataset ${dataset} \
        --save_dir ${saliency_map_path} \
        --mask ${origin_model_path} \
        --unlearn GA \
        --num_indexes_to_replace 4500 \
        --unlearn_epochs 1 \
        --gpu ${gpu_id}