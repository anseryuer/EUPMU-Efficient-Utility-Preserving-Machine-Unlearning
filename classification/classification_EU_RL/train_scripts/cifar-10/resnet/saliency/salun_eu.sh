
arch=resnet18
dataset=cifar10
origin_model_path=pretrained_models/${arch}/${dataset}/model_SA_best.pth.tar
save_dir=output
gpu_id=0
project_name=Unlearn_${arch}_${dataset}_MTL_RL_salient



source activate mu

for unlearn_lr in 6e-3 6e-4 6e-5 1e-3 1e-4 1e-5; do
        entity_name=eu_lr${unlearn_lr}
        python -u main_random.py \
               --arch ${arch} \
               --dataset ${dataset} \
               --unlearn RL \
               --unlearn_epochs 10 \
               --unlearn_lr ${unlearn_lr}  \
               --num_indexes_to_replace 4500 \
               --mask ${origin_model_path} \
               --save_dir ${save_dir} \
               --gpu ${gpu_id} \
               --wandb_project ${project_name} \
               --wandb_entity ${entity_name} \
               --mtl \
               --mtl_method eu
    done
done