# CIFAR-10 (50k train): 10% 4500, 30% 13500, 50% 22500
python generate_mask.py --arch resnet18 --dataset cifar10 --num_indexes_to_replace 4500  --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir saliency_maps/
python generate_mask.py --arch resnet18 --dataset cifar10 --num_indexes_to_replace 13500 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir saliency_maps/
python generate_mask.py --arch resnet18 --dataset cifar10 --num_indexes_to_replace 22500 --mask pretrained_models/resnet18/cifar10/model_SA_best.pth.tar --save_dir saliency_maps/

# CIFAR-100 (50k train): 10% 4500, 30% 13500, 50% 22500
python generate_mask.py --arch resnet18 --dataset cifar100 --num_indexes_to_replace 4500  --mask pretrained_models/resnet18/cifar100/model_SA_best.pth.tar --save_dir saliency_maps/
python generate_mask.py --arch resnet18 --dataset cifar100 --num_indexes_to_replace 13500 --mask pretrained_models/resnet18/cifar100/model_SA_best.pth.tar --save_dir saliency_maps/
python generate_mask.py --arch resnet18 --dataset cifar100 --num_indexes_to_replace 22500 --mask pretrained_models/resnet18/cifar100/model_SA_best.pth.tar --save_dir saliency_maps/

# TinyImageNet (100k train): 10% 10000, 30% 30000, 50% 50000
python generate_mask.py --arch resnet18 --dataset TinyImagenet --num_indexes_to_replace 10000 --mask pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar --save_dir saliency_maps/ --data ../data/tiny-imagenet-200
python generate_mask.py --arch resnet18 --dataset TinyImagenet --num_indexes_to_replace 30000 --mask pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar --save_dir saliency_maps/ --data ../data/tiny-imagenet-200
python generate_mask.py --arch resnet18 --dataset TinyImagenet --num_indexes_to_replace 50000 --mask pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar --save_dir saliency_maps/ --data ../data/tiny-imagenet-200