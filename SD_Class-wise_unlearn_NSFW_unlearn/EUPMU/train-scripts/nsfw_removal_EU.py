import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import (
    setup_forget_nsfw_data,
    setup_model,
)
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
from weighted_methods.utils import extract_weight_method_parameters_from_args
from weighted_methods.weight_methods import WeightMethods

from types import SimpleNamespace

try:
    import wandb as _wandb
except Exception:
    _wandb = None


class _WandbStub:
    def __init__(self):
        self.config = SimpleNamespace()

    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


wandb = _wandb if _wandb is not None and hasattr(_wandb, "init") else _WandbStub()
wandb.init(project="SD_NSFW_unlearning")
try:
    # Retrieve hyperparameters from the sweep configuration
    weight_learning_rate_eu = wandb.config.weight_learning_rate_eu
    error_eu = wandb.config.error_eu
except:
    # Set default hyperparameters
    weight_learning_rate_eu = 10
    error_eu = 0.5
print(f"EU Weight learning rate: {weight_learning_rate_eu}, EU Error: {error_eu}")
# EU hyperparameters
OMD_METHODS = {"omd_tch", "omd_tch_eg", "omd_tch_pgd", "afleg", "afl"}


def _mtl_log_metrics(method_name, weight_method, extra_outputs):
    if not extra_outputs:
        return {}

    if method_name == "eu":
        metrics = {"EU Weight": float(weight_method.method.w.detach().cpu().item())}
        weight_grad = getattr(weight_method.method.w, "grad", None)
        if weight_grad is not None:
            metrics["EU weight grad"] = float(weight_grad.detach().cpu().item())
        return metrics

    if method_name in OMD_METHODS:
        updated_weights = extra_outputs.get("updated_omd_weights")
        metrics = {"OMD eta": float(extra_outputs.get("eta", 0.0))}
        if isinstance(updated_weights, torch.Tensor) and updated_weights.numel() >= 2:
            metrics["OMD retain weight"] = float(updated_weights[0].detach().cpu().item())
            metrics["OMD forget weight"] = float(updated_weights[1].detach().cpu().item())
        return metrics

    if method_name == "chebyshev":
        active_task = extra_outputs.get("active_task")
        if active_task is not None:
            return {"Chebyshev active task": int(active_task)}
    return {}


def _mtl_postfix(method_name, weight_method, extra_outputs):
    if not extra_outputs:
        return {}

    if method_name == "eu":
        return {"eu_weight": float(weight_method.method.w.detach().cpu().item())}

    if method_name in OMD_METHODS:
        updated_weights = extra_outputs.get("updated_omd_weights")
        if isinstance(updated_weights, torch.Tensor) and updated_weights.numel() >= 2:
            return {
                "retain_w": float(updated_weights[0].detach().cpu().item()),
                "forget_w": float(updated_weights[1].detach().cpu().item()),
            }
        return {}

    if method_name == "chebyshev" and extra_outputs.get("active_task") is not None:
        return {"active_task": int(extra_outputs["active_task"])}
    return {}



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


def nsfw_removal(
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size)

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
            # print(name)
            parameters.append(param)
    # set model to train
    model.train()

    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-nsfw-mask-method_{train_method}-lr_{lr}"
    else:
        name = f"compvis-nsfw-method_{train_method}-lr_{lr}"
    if args.mtl:
        weight_methods_parameters = extract_weight_method_parameters_from_args(args)
        method_kwargs = dict(weight_methods_parameters[args.mtl_method])
        if args.mtl_method == "eu":
            method_kwargs.update(dict(w_lr=weight_learning_rate_eu, error=error_eu))
        weight_method = WeightMethods(args.mtl_method, n_tasks=2, device=device, **method_kwargs)

    # NSFW Removal
    word_nude = "a photo of a nude person"
    word_wear = "a photo of a person wearing clothes"

    # TRAINING CODE
    for epoch in range(epochs):
        remain_iter = iter(remain_dl)
        with tqdm(total=len(forget_dl)) as time:
            # with tqdm(total=10) as time:

            for i, forget_images in enumerate(forget_dl):
                # for i in range(1):
                optimizer.zero_grad()

                try:
                    remain_images = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images = next(remain_iter)

                forget_prompts = [word_nude] * batch_size

                # player -> truck
                pseudo_prompts = [word_wear] * batch_size
                remain_prompts = [word_wear] * batch_size

                # remain stage
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                remain_loss = model.shared_step(remain_batch)[0]

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

                forget_loss = 100 * criteria(forget_out, pseudo_out)

                # total loss
                #loss = forget_loss + alpha * remain_loss
                #loss.backward()
                #losses.append(loss.item() / batch_size)
                remain_loss_alpha = alpha * remain_loss

                if args.mtl:
                    loss, method_outputs = weight_method.backward(
                        losses=torch.stack([remain_loss_alpha, forget_loss]),
                        shared_parameters=list(model.model.diffusion_model.parameters()),
                    )
                else:
                    method_outputs = {}
                    loss = forget_loss + remain_loss_alpha
                    loss.backward()
                losses.append(loss.item() / batch_size)
                train_metrics = {
                    "loss": loss.item() / batch_size,
                    "remain_loss": remain_loss.item() / batch_size,
                    "forget_loss": forget_loss.item() / batch_size,
                }
                if args.mtl:
                    train_metrics.update(
                        _mtl_log_metrics(args.mtl_method, weight_method, method_outputs)
                    )
                wandb.log(train_metrics)

                if False:#mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in parameters:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )
                            print(n)

                optimizer.step()

                if args.mtl and args.mtl_method == "eu":
                    with torch.no_grad():
                        """
                        remain_input, remain_emb = model.get_input(
                            remain_batch, model.first_stage_key
                        )
                        remain_noisy = model.q_sample(x_start=remain_input, t=t, noise=noise)

                        remain_out = model.apply_model(remain_noisy, t, remain_emb)

                        new_remain_loss = criteria(remain_out, noise)"""
                        new_remain_loss = model.shared_step(remain_batch)[0]
                        weight_method.method.update(new_remain_loss.detach())
                        eu_metrics = {
                            "EU Weight": float(weight_method.method.w.detach().cpu().item()),
                            "EU update Loss": new_remain_loss.item() / batch_size,
                        }
                        weight_grad = getattr(weight_method.method.w, "grad", None)
                        if weight_grad is not None:
                            eu_metrics["EU weight grad"] = float(
                                weight_grad.detach().cpu().item()
                            )
                        wandb.log(eu_metrics)

                time.set_description("Epo-ch %i" % epoch)
                progress = {
                    "loss": loss.item() / batch_size,
                    "remain_loss": remain_loss.item() / batch_size,
                    "forget_loss": forget_loss.item() / batch_size,
                }
                if args.mtl:
                    progress.update(
                        _mtl_postfix(args.mtl_method, weight_method, method_outputs)
                    )
                time.set_postfix(progress)
                sleep(0.1)
                time.update(1)

    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )


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
        try:
            from convertModels import savemodelDiffusers
        except Exception as exc:
            print(f"Skipping diffusers conversion because convertModels could not be imported: {exc}")
        else:
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
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
    )

    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
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
        default="0,0",
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
        default=50,
    )
    parser.add_argument("--mtl", action="store_true", default=False, help="")
    parser.add_argument("--mtl_method", type=str, default=None, help="")
    parser.add_argument("--cheby_retain_weight", default=1.0, type=float, help="Chebyshev weight for the retain loss")
    parser.add_argument("--cheby_forget_weight", default=1.0, type=float, help="Chebyshev weight for the forget loss")
    parser.add_argument("--cheby_retain_ref", default=0.0, type=float, help="Chebyshev reference value for the retain loss")
    parser.add_argument("--cheby_forget_ref", default=0.0, type=float, help="Chebyshev reference value for the forget loss")
    parser.add_argument("--cheby_rho", default=1e-3, type=float, help="Augmentation coefficient for Chebyshev scalarization")
    parser.add_argument("--omd_tch_retain_weight", default=1.0, type=float, help="OMD-TCH weight for the retain loss")
    parser.add_argument("--omd_tch_forget_weight", default=1.0, type=float, help="OMD-TCH weight for the forget loss")
    parser.add_argument("--omd_tch_retain_ref", default=0.0, type=float, help="OMD-TCH reference value for the retain loss")
    parser.add_argument("--omd_tch_forget_ref", default=0.0, type=float, help="OMD-TCH reference value for the forget loss")
    parser.add_argument("--omd_tch_eta", default=0.1, type=float, help="Mirror descent step size for OMD-TCH")
    parser.add_argument("--omd_tch_rho", default=0.0, type=float, help="Optional augmentation coefficient for OMD-TCH; paper-consistent default is 0.0")

    args = parser.parse_args()

    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    nsfw_removal(
        train_method=train_method,
        alpha=alpha,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask_path=mask_path,
        diffusers_config_path=diffusers_config_path,
        device=device,
        image_size=image_size,
        ddim_steps=ddim_steps,
    )
