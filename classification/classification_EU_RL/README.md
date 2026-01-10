# EUPMU for Classification
This is the code base for EUPMU for Clasification. The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.

## Build the environment

```bash
# create conda environment
conda create -n clsfc python=3.9 -y
conda activate clsfc
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install google-genai
```
## Training from scratch

In train_scripts there are scripts to train models from scratch, as well as some other training scripts.

## Program files
- main_random.py: the main file for EUPMU, GDR-GMA, and Salun with MTL methods implemented inside weighted_methods/weight_methods.py.
- main_forget.py: the main file for other baselines, including FT, GA, IU, l1-sparse, and retrain.

You can find the recommended hyperparameters for each unlearning method below or under train_scripts/ folder. You might need to adjust some hyperparameters for different architectures/datasets.

## Unlearning for Class-wise Data Forgetting

Here you can modify the --class_to_replace class index to whichever class you wish to unlearn.

For class-wise unlearning, we adjust the parameter blind to retrain results. Our parameter tuning target is to achieve complete forgetting while maintaining high model utility.

### Retrain

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn retrain --class_to_replace 0 --unlearn_epochs 160 --unlearn_lr 0.1
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn retrain --unlearn_epochs "160" --unlearn_lr 0.1 --class_to_replace 0  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0"
``` 

### EUPMU

As different tasks and datasets may require different hyperparameters, we provide some recommended hyperparameters for EUPMU as for ResNet18 on CIFAR10 unlearning. With adjusting the unlearn_lr, and eu_error you can explore the pareto front of the trade-off between unlearning effectiveness and model utility.

```
python -u main_random.py --arch resnet18 --dataset cifar10 --unlearn RL --unlearn_epochs "5" --unlearn_lr 5e-3 --class_to_replace 0  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --mtl --mtl_method eu 
--eu_w_lr 1 --eu_error 0.05 --wandb_project cls --wandb_entity clswise_unl
```

### FT

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn FT --class_to_replace 0 --unlearn_lr 0.01 --unlearn_epochs 5
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn FT --unlearn_epochs "5" --unlearn_lr 1e-2 --class_to_replace 0  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0"
```


### GA

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn GA --class_to_replace 0 --unlearn_lr 0.0001 --unlearn_epochs 5
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn GA --unlearn_epochs "5" --unlearn_lr 3e-4 --class_to_replace 0  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0"
```

### IU

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn wfisher --class_to_replace 0 --alpha ${alpha}
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn wfisher --unlearn_epochs "5" --class_to_replace 0  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --alpha 2
```

### l1-sparse

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn FT_prune --class_to_replace 0 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn FT_prune --unlearn_epochs "5" --unlearn_lr 1e-2 --class_to_replace 0  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --alpha 1e-4
```

### SalUn

Generate saliency maps first and put them in saliency_maps/resnet18/cifar10/saliency_map_class_0.pt
```
python generate_mask.py --arch resnet18 --dataset cifar10 --class_to_replace 0 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir saliency_maps/
```

Then run SalUn with the following command, you can change the --path to the path of your saliency map.

```
# Salun
python -u main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.1 --class_to_replace 0 --mask ${origin_model_path} --save_dir ${save_dir} --path ${saliency_map_path}

# Soft-thresholding Salun
python -u main_random.py --unlearn RL_proximal --unlearn_epochs 10 --unlearn_lr 0.1 --class_to_replace 0 --mask ${origin_model_path} --save_dir ${save_dir} --path ${saliency_map_path}
```

```
python -u main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.01 --class_to_replace 0 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --path saliency_maps/resnet18/cifar10/forget_10.0%/with_0.5.pt

# Soft-thresholding Salun
python -u main_random.py --unlearn RL_proximal --unlearn_epochs 10 --unlearn_lr 0.01 --class_to_replace 0 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --path saliency_maps/resnet18/cifar10/forget_10.0%/with_0.5.pt --mask_ratio 0.5
```

### GDR-GMA

```
python -u main_random.py --save_dir ${save_dir} --unlearn RL --class_to_replace 0 --unlearn_lr 0.1 --unlearn_epochs 10 --mtl_method gdr_gma
```

```
python -u main_random.py --save_dir output --unlearn RL --class_to_replace 0 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --unlearn_lr 0.1 --unlearn_epochs 10 --mtl_method gdr_gma
```

## Unlearning for Random Data Forgetting

Here you can change the number after --num_indexes_to_replace to change the percentage of random selected data to forget.

For Random Data Forgetting unlearning, we adjust the parameter blind to retrain results. Our parameter tuning target is to achieve lowest (abs(original model train accuracy - finetuned model train accuracy) + abs(original model validation accuracy - finetuned model validation accuracy) + abs(finetuned model unlearning accuracy - original model validation accuracy)). We choose this because it is safe to assume that in random data forgetting, the unlearning accuracy should be close to the original validation/test accuracy as both are random samples from the same distribution.

### EUPMU

```
python -u main_random.py --save_dir ${save_dir} --unlearn RL --num_indexes_to_replace 4500 --unlearn_lr 0.1 --unlearn_epochs 10 --mtl_method eu
```
We recommend to use the following hyperparameters for EUPMU as for ResNet18 on CIFAR10 unlearning 4500 random samples:
```
python -u main_random.py --arch resnet18 --dataset cifar10 --unlearn RL --unlearn_epochs "5" --unlearn_lr 3e-3 --num_indexes_to_replace 4500  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --mtl --mtl_method eu --eu_w_lr 1 --eu_error 0.01 --wandb_project rnd --wandb_entity rnd_c10_10
```
For 30% data forgetting (13500 samples):
```
python -u main_random.py --arch resnet18 --dataset cifar10 --unlearn RL --unlearn_epochs "5" --unlearn_lr 1e-2 --num_indexes_to_replace 13500  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --mtl --mtl_method eu --eu_w_lr 1 --eu_error 0.01 --wandb_project rnd --wandb_entity rnd_c10_30
```

For 50% data forgetting (22500 samples):
```
python -u main_random.py --arch resnet18 --dataset cifar10 --unlearn RL --unlearn_epochs "5" --unlearn_lr 3e-2 --num_indexes_to_replace 22500  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --mtl --mtl_method eu --eu_w_lr 1 --eu_error 0.05 --weight_init 0.3 --wandb_project rnd --wandb_entity rnd_c10_50
```

For Cifar100 unlearning 10% random samples:
```
python -u main_random.py --arch resnet18 --dataset cifar100 --unlearn RL --unlearn_epochs "5" --unlearn_lr ???
```

### EUPMU with fast approximation

```
python -u main_random.py --arch resnet18 --dataset cifar10 --unlearn RL --unlearn_epochs "5" --unlearn_lr ???
```

### Retrain

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1
```

```
python -u main_forget.py --arch resnet18 --dataset cifar100 --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1 --mask pretrained_models/resnet18/cifar100/model_SA_best.pth.tar --save_dir output
```

### FT

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn FT --unlearn_epochs "5" --unlearn_lr 2e-2 --num_indexes_to_replace 4500 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0"
```

### GA

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5
```
```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn GA --unlearn_epochs "5" --unlearn_lr 0.0015 --num_indexes_to_replace 4500 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0"
```

### IU

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn wfisher --num_indexes_to_replace 4500 --alpha ${alpha}
```

```
python -u main_forget.py --arch resnet18 --dataset cifar10 --unlearn wfisher --unlearn_epochs "5" --num_indexes_to_replace 4500 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir output --gpu "0" --alpha 2
```


### l1-sparse

```
python -u main_forget.py --save_dir ${save_dir} --mask ${origin_model_path} --unlearn FT_prune --num_indexes_to_replace 4500 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10
```
### SalUn
```
# Salun
python -u main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.1 --num_indexes_to_replace 4500 --mask ${origin_model_path} --save_dir ${save_dir} --path ${saliency_map_path}

# Soft-thresholding Salun
python -u main_random.py --unlearn RL_proximal --unlearn_epochs 10 --unlearn_lr 0.1 --num_indexes_to_replace 4500 --mask ${origin_model_path} --save_dir ${save_dir} --path ${saliency_map_path}
```

# Known Issues

This branch: 
The current implementation of MTL methods does not use optimizer.zero_grad() to clear gradients, which may lead to an accumulation of gradients over iterations. This could potentially affect the training dynamics and convergence of the model. Fixing this issue is on our to-do list for future updates. However, despite this limitation, the MTL methods have still demonstrated effective performance in our experiments and such minor bug does not affect the main results of our paper.