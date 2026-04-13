import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers
from dataset import setup_forget_data, setup_model, setup_remain_data
from diffusers import LMSDiscreteScheduler
from tqdm import tqdm
from weighted_methods.utils import extract_weight_method_parameters_from_args
from weighted_methods.weight_methods import WeightMethods

import random

# EU hyperparameters


def certain_label(
    class_to_forget,
    train_method,
    batch_size,
    epochs,
    lr,
    alpha,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
    w_lr=3.0,
    error=0.0,
    use_wandb=False,
):
    if use_wandb:
        import wandb
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)

    # set model to train
    model.train()
    losses = []

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            parameters.append(param)

    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)

        name = f"compvis-cl-mask-class_{str(class_to_forget)}-method_{train_method}-epoch_{epochs}-lr_{lr}"
    else:
        name = f"compvis-cl-class_{str(class_to_forget)}-method_{train_method}-epoch_{epochs}-lr_{lr}"
    if args.mtl:
        assert args.mtl_method == "eu" # Only implemented for efficient unlearning
        # weight method
        weight_methods_parameters = extract_weight_method_parameters_from_args(args)
        weight_method = WeightMethods(args.mtl_method, n_tasks=2, device=device, w_lr = w_lr, error = error)
        name += f"-mtl_{args.mtl_method}-w_lr_{w_lr}-err_{error}"
    if alpha != 1.0:
        name += f"-alpha_{alpha}"
    # TRAINING CODE
    remain_iter = iter(remain_dl)
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time:
            for i, (images, labels) in enumerate(forget_dl):
                optimizer.zero_grad()

                forget_images, forget_labels = images, labels
                try:
                    remain_images, remain_labels = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_labels = next(remain_iter)
                # print(f"Batch {i}: forget_labels[0]={forget_labels[0]}, remain_labels[0]={remain_labels[0]}")
                # if i == 2: break
                torch.cuda.empty_cache()
                #import pdb
                #pdb.set_trace()
                forget_prompts = [descriptions[label] for label in forget_labels]

                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for label in forget_labels
                ]
                remain_prompts = [descriptions[label] for label in remain_labels]
                #print(forget_prompts, pseudo_prompts, remain_prompts)

                # remain stage
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                remain_loss = model.shared_step(remain_batch)[0]
                scaled_remain_loss = alpha * remain_loss

                # forget stage
                forget_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }

                pseudo_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": pseudo_prompts,
                }

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(
                    0,
                    model.num_timesteps,
                    (forget_input.shape[0],),
                    device=model.device,
                ).long()
                noise = torch.randn_like(forget_input, device=model.device)

                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                forget_loss = criteria(forget_out, pseudo_out)

                # total loss
                torch.cuda.empty_cache()

                #loss = forget_loss + remain_loss
                #loss.backward()
                #print(f"forget_loss: {forget_loss.item() / batch_size}, remain_loss: {remain_loss.item() / batch_size}")
                loss, _ = weight_method.backward(
                    losses=torch.stack([scaled_remain_loss, forget_loss]),
                    shared_parameters=list(model.model.diffusion_model.parameters()),
                )
                torch.cuda.empty_cache()
                losses.append(loss.item() / batch_size)
                if use_wandb:
                    wandb.log({
                        "loss": loss.item() / batch_size,
                        "remain_loss": remain_loss.item() / batch_size,
                        "scaled_remain_loss": scaled_remain_loss.item() / batch_size,
                        "forget_loss": forget_loss.item() / batch_size,
                    })

                if mask_path:
                    print("Applying mask")
                    assert False # Just testing in efficient unlearning.
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in parameters:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )
                            print(n)

                optimizer.step()

                if args.mtl_method == "eu":
                    with torch.no_grad():
                        """
                        remain_input, remain_emb = model.get_input(
                            remain_batch, model.first_stage_key
                        )
                        remain_noisy = model.q_sample(x_start=remain_input, t=t, noise=noise)

                        remain_out = model.apply_model(remain_noisy, t, remain_emb)

                        new_remain_loss = criteria(remain_out, noise)"""
                        new_remain_loss = model.shared_step(remain_batch)[0]
                        scaled_new_remain_loss = alpha * new_remain_loss
                        weight_method.method.update(scaled_new_remain_loss.detach())
                        torch.cuda.empty_cache()
                        if use_wandb:
                            wandb.log({
                                "EU Weight": weight_method.method.w,
                                "EU update Loss": new_remain_loss.item() / batch_size,
                                "EU scaled update Loss": scaled_new_remain_loss.item() / batch_size,
                                "EU weight grad": weight_method.method.w.grad,
                            })

                time.set_description("Epo-ch %i" % epoch)
                time.set_postfix({
                    "loss": loss.item() / batch_size,
                    "remain_loss": remain_loss.item() / batch_size,
                    "scaled_remain_loss": scaled_remain_loss.item() / batch_size,
                    "forget_loss": forget_loss.item() / batch_size,
                    "eu_weight": weight_method.method.w.detach().cpu().numpy()[0],
                })
                sleep(0.1)
                time.update(1)

    model.eval()
    save_model(
        model,
        name,
        num=None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )

    save_history(losses, name, classes)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--alpha",
        help="extra multiplier applied to the retain loss before EU weighting",
        type=float,
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument("--mtl", action="store_true", default=False, help="")
    parser.add_argument("--mtl_method", type=str, default=None, help="")
    parser.add_argument("--seed", type=int, default=42, help="random seed for execution")
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--w_lr", type=float, default=3.0, help="eu weight learning rate")
    parser.add_argument("--error", type=float, default=0.5, help="eu error")
    args = parser.parse_args()

    # Set seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.wandb:
        import wandb
        wandb.init(project="SD_unlearning")

    # classes = [int(d) for d in args.classes.split(',')]
    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    alpha = args.alpha
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    certain_label(
        classes,
        train_method,
        batch_size,
        epochs,
        lr,
        alpha,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
        args.w_lr,
        args.error,
        args.wandb,
    )
    if args.wandb:
        import wandb
        wandb.finish()
