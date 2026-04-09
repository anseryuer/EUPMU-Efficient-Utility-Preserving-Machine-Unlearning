# https://huggingface.co/docs/diffusers/conceptual/evaluation
import argparse

import torch
from dataset import setup_fid_data
from torchmetrics.image.fid import FrechetInceptionDistance

import torchmetrics
print(f"Current torchmetrics version: {torchmetrics.__version__}")
print("!!!PLEASE make sure torchmetrics version is 1.0.0 or newer to avoid an old FID metric problem that breaks the whole thing!!!")

def compute_fid(class_to_forget, path, image_size):

    fid = FrechetInceptionDistance(feature=64)

    real_set, fake_set = setup_fid_data(class_to_forget, path, image_size)

    real_images = torch.stack(real_set).to(torch.uint8).cpu()
    fake_images = torch.stack(fake_set).to(torch.uint8).cpu()
    print("real_images.shape:", real_images.shape)
    print("fake_images.shape:", fake_images.shape)

    fid.update(real_images, real=True)  # doctest: +SKIP
    fid.update(fake_images, real=False)  # doctest: +SKIP

    print("++++++++++++++++++++++++++++++FID RESULE:+++++++++++++++++++++++++++++++")
    print(path)
    print(fid.compute())  # doctest: +SKIP
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--folder_path", help="path of images", type=str, required=True)
    parser.add_argument(
        "--class_to_forget", help="class_to_forget", type=int, required=True
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    args = parser.parse_args()

    path = args.folder_path
    class_to_forget = args.class_to_forget
    image_size = args.image_size
    print("class_to_forget:", class_to_forget)
    compute_fid(class_to_forget, path, image_size)
