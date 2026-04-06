from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from datasets import load_dataset


import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
sys.path.append(parent_dir)

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import InterpolationMode
import datasets

import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from tqdm import tqdm



INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.dataset.features["label"].names)
        }
        # self.file_to_class = {
        #     str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))
        # }

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label


class NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


class NOT_NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/not-nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image
    
class ForgetDS(Dataset):
    def __init__(self, data_dir, transform=None, prompt_list = None):
        self.dataset = load_dataset(data_dir)["train"]
        self.transform = transform
        self.prompt_list = prompt_list
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)
        if self.prompt_list is not None:
            return image, self.prompt_list[idx]
        return image


class RetainDS(Dataset):
    def __init__(self, data_dir, transform=None, prompt_list = None):
        self.dataset = load_dataset(data_dir)["train"]
        self.transform = transform
        self.prompt_list = prompt_list
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)
        if self.prompt_list is not None:
            return image, self.prompt_list[idx]
        return image

def setup_model(config, ckpt, device):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location=device)
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def setup_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", class_to_forget, transform)
    # train_set = Imagenette('train', transform)

    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_ga_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    labels = train_set.dataset["label"]
    filtered_indices = [i for i, label in enumerate(labels) if label == class_to_forget]
    filtered_data = torch.utils.data.Subset(train_set, filtered_indices)

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)
    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    labels = train_set.dataset["label"]
    filtered_indices = [i for i, label in enumerate(labels) if label != class_to_forget]
    filtered_data = torch.utils.data.Subset(train_set, filtered_indices)
    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)
    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    labels = train_set.dataset["label"]
    filtered_indices = [i for i, label in enumerate(labels) if label == class_to_forget]
    filtered_data = torch.utils.data.Subset(train_set, filtered_indices)
    train_dl = DataLoader(filtered_data, batch_size=batch_size)
    return train_dl, descriptions


def setup_forget_nsfw_data(batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = NSFW(transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_set = NOT_NSFW(transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl

def setup_forget_concept_unlearn_data(batch_size, image_size, ret_dir, fgt_dir, ret_prompt_list=None, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = ForgetDS(data_dir=fgt_dir,transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    if ret_prompt_list is None:
        remain_set = RetainDS(data_dir=ret_dir, transform=transform)
    else:
        remain_set = RetainDS(data_dir=ret_dir, transform=transform, prompt_list=ret_prompt_list)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)





class TinyImageNetDataset(Dataset):
    def __init__(self, data_path, image_folder_set, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform


        self.class2className={}
        with open(f'{data_path}/words.txt') as words:
            for line in words:
                line = line.strip('\n')
                class_id, class_name = line.split("\t")
                self.class2className[class_id]=class_name

            id2class={v:k for k,v in image_folder_set.class_to_idx.items()}
        self.id2className={}


        for sample in tqdm(image_folder_set.imgs[start:end]):
            class_name = self.class2className[id2class[sample[1]]]
            self.id2className[sample[1]]=class_name
            self.targets.append(sample[1])
            # img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            # if norm_trans is not None:
            #     img = norm_trans(img)
            self.imgs.append(sample[0])
        # self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]
        #
        # img = Image.open(self.imgs[idx]).convert("RGB")
        #
        # if self.transform is not None:
        #     return self.transform(img), self.targets[idx]
        # else:
        #     return img, self.targets[idx]



class TinyImageNetDataset_(Dataset):
    def __init__(self, datsets, transform):
        self.data=datsets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img = Image.open(self.data[idx][0]).convert("RGB")

        if self.transform is not None:
            return self.transform(img), self.data[idx][1]
        else:
            return img, self.data[idx][1]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False, image_size=512, interpolation="bicubic"):
        interpolation = INTERPOLATIONS[interpolation]
        self.args = args

        self.tr_train =  get_transform(interpolation, image_size)


        self.train_path = os.path.join(args.data, "train/")
        self.val_path = os.path.join(args.data, "val/")
        self.test_path = os.path.join(args.data, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]


            paths = glob.glob(os.path.join(args.data, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        class_to_forget: int = None,

    ):

        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(data_dir, train_set)

        train_set.targets = np.array(train_set.targets)

        descriptions = [f"an image of a {label}" for label in train_set.id2className.values()]

        #
        #
        #
        #
        #
        # import pandas as pd
        #
        # # 定义表头和数据
        # columns = ['case_number', 'prompt', 'evaluation_seed', 'class', 'classidx']
        # datas=[]
        # for indx, (classid, class_name) in enumerate(train_set.id2className.items()):
        #     datas.append([classid,
        #                   f"Image of {class_name}",
        #                   np.random.randint(1, 10000),
        #                   class_name,
        #                   classid])
        #
        # # 创建 DataFrame
        # df = pd.DataFrame(datas, columns=columns)
        #
        # # 将 DataFrame 保存为 CSV 文件
        # csv_file_path = 'prompts/tinyimagenet.csv'
        # df.to_csv(csv_file_path, index=False)


        filtered_data_forget = [data for data in train_set if data[1] == int(class_to_forget)]
        filtered_data_forget = TinyImageNetDataset_(filtered_data_forget, train_set.transform)
        train_dl_forget = DataLoader(filtered_data_forget, batch_size=batch_size, num_workers=4)

        filtered_data_retian = [data for data in train_set if data[1] != int(class_to_forget)]
        filtered_data_retian = TinyImageNetDataset_(filtered_data_retian, train_set.transform)
        train_dl_retain = DataLoader(filtered_data_retian, batch_size=batch_size, shuffle=True, num_workers=4)


        print(
            f"Traing retian loader: {len(train_dl_retain.dataset)} images"
        )
        print(
            f"Traing forget loader: {len(train_dl_forget.dataset)} images"
        )
        return train_dl_retain, train_dl_forget, descriptions




