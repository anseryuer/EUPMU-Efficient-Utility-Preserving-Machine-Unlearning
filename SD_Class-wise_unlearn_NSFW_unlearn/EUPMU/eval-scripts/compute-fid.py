# https://huggingface.co/docs/diffusers/conceptual/evaluation
import argparse
from pathlib import Path

import torch
from dataset import setup_fid_data
from torchmetrics.image.fid import FID
# from torchmetrics.image.fid import FID


def compute_fid(class_to_forget, path, image_size):

    # fid = FID( feature=64)
    fid = FID(feature=64)

    real_set, fake_set = setup_fid_data(class_to_forget, path, image_size)

    real_images = torch.stack(real_set).to(torch.uint8).cpu()
    fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

    fid.update(real_images, real=True)  # doctest: +SKIP
    fid.update(fake_images, real=False)  # doctest: +SKIP

    print("++++++++++++++++++++++++++++++FID RESULE:+++++++++++++++++++++++++++++++")
    print(path)
    fid_score = fid.compute()
    print(fid_score)  # doctest: +SKIP
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--folder_path", help="path of images", type=str, required=True)
    parser.add_argument(
        "--class_to_forget", help="class_to_forget", type=int, required=False, default=6
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--save_path",
        help="optional path to save the FID as plain text",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()

    path = args.folder_path
    class_to_forget = args.class_to_forget
    image_size = args.image_size
    print("class_to_forget:", class_to_forget)
    fid_score = compute_fid(class_to_forget, path, image_size)
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(f"{float(fid_score):.10f}\n")
