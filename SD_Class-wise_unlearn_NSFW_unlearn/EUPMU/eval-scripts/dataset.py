import os
import numpy as np
import torchvision.transforms as torch_transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize((size, size), interpolation=interpolation),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset.features["label"].names)}

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


class Fake_Imagenette(Dataset):
    def __init__(self, data_dir, class_to_forget, transform=None):
        self.data_dir = data_dir
        self.transform = transform


        # Get all image files in the data folder
        # self.image_files =os.listdir(data_dir)
        self.image_files = [
            f
            for f in os.listdir(data_dir)
            if (f.endswith(".png") and not f.startswith(str(class_to_forget)))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Parse class index from the filename
        filename = self.image_files[idx]
        # print(filename)
        class_idx = int(filename.split("/")[-1].split("_")[0])

        # Load the image
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path)

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, class_idx


def setup_fid_data(class_to_forget, path, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    fake_set = Fake_Imagenette(path, class_to_forget, transform=transform)
    fake_set = [data[0] for data in fake_set]

    real_set = Fake_Imagenette(
        f"./imagenette_without_label_{class_to_forget}", class_to_forget, transform=transform
    )
    real_set = [data[0] for data in real_set]

    return real_set, fake_set

class Fake_Imagenette_One_Class(Dataset):
    def __init__(self, data_dir, class_idx, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get all image files in the data folder corresponding to the class_idx
        self.image_files = [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".png") and f.startswith(str(class_idx) + "_")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Parse class index from the filename
        filename = self.image_files[idx]

        # Load the image
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path)

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image


def setup_fid_data_per_class(class_idx, path, image_size, fgt_class_idx, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    # Load fake images for the specified class
    fake_set = Fake_Imagenette_One_Class(path, class_idx, transform=transform)
    fake_set = [data for data in fake_set]

    # Load real images for the specified class
    real_set = Fake_Imagenette_One_Class(
        f"./imagenette_without_label_{fgt_class_idx}", class_idx, transform=transform
    )
    real_set = [data for data in real_set]

    return real_set, fake_set


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
    def __init__(self, data_path, image_folder_set, norm_trans=None, start=0, end=-1):
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
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, data_dir, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(data_dir, "train/")
        self.val_path = os.path.join(data_dir, "val/")
        self.test_path = os.path.join(data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]


            paths = glob.glob(os.path.join(data_dir, "val/images/*"))
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

        self.train_set = ImageFolder(self.train_path, transform=self.tr_train)
        self.train_set = TinyImageNetDataset(data_dir, self.train_set, self.norm_layer)

        self.train_set.targets = np.array(self.train_set.targets)




