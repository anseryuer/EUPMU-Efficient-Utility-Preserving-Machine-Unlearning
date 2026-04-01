import argparse
from pathlib import Path

import torch
from dataset import setup_fid_data_per_class
from torchmetrics.image.fid import FID


def compute_fid_per_class(class_to_forget, path, image_size):
    # FID instance
    fid = FID(feature=64)

    avg_fid = 0

    # Iterate through each class except the class_to_forget
    for class_idx in range(10):
        if class_idx == class_to_forget:
            continue

        # Get real and fake images for the current class
        real_set, fake_set = setup_fid_data_per_class(class_idx, path, image_size, class_to_forget)

        print(len(real_set), len(fake_set))

        real_images = torch.stack(real_set).to(torch.uint8).cpu()
        fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

        fid.update(real_images, real=True)  # Update with real images for the class
        fid.update(fake_images, real=False)  # Update with fake images for the class

        # Compute and print FID score for the current class
        fid_score = fid.compute()
        print(f"++++++++++++++++++++++++++++++ FID RESULT FOR CLASS {class_idx} +++++++++++++++++++++++++++++++")
        print(f"Class {class_idx}: FID Score = {fid_score}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        avg_fid += fid_score

        # Reset the FID for the next class
        fid.reset()
    average_fid = avg_fid / 9
    print(f"Average FID Score = {average_fid}")
    return average_fid

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
        help="optional path to save the final average FID as plain text",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()

    path = args.folder_path
    class_to_forget = args.class_to_forget
    image_size = args.image_size
    print("class_to_forget:", class_to_forget)
    average_fid = compute_fid_per_class(class_to_forget, path, image_size)
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(f"{float(average_fid):.10f}\n")
