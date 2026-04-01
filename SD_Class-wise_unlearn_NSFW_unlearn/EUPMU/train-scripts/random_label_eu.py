import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import setup_forget_data, setup_model, setup_remain_data
from tqdm import tqdm
from weighted_methods.utils import extract_weight_method_parameters_from_args
from weighted_methods.weight_methods import WeightMethods

import random
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
wandb.init(project="SD_unlearning")
wandb_weight_learning_rate_eu = None
wandb_error_eu = None
try:
    # Retrieve hyperparameters from the sweep configuration
    wandb_weight_learning_rate_eu = wandb.config.weight_learning_rate_eu
    wandb_error_eu = wandb.config.error_eu
except:
    pass

weight_learning_rate_eu = None
error_eu = None
# EU hyperparameters
OMD_METHODS = {"omd_tch", "omd_tch_eg", "omd_tch_pgd", "afleg", "afl"}


def _as_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return [float(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return [float(v) for v in value.flatten().tolist()]
    if isinstance(value, (float, int)):
        return float(value)
    return value


def _mtl_log_metrics(method_name, weight_method, extra_outputs):
    if not extra_outputs:
        return {}

    metrics = {}
    if method_name == "eu":
        metrics["EU Weight"] = float(weight_method.method.w.detach().cpu().item())
        weight_grad = getattr(weight_method.method.w, "grad", None)
        if weight_grad is not None:
            metrics["EU weight grad"] = float(weight_grad.detach().cpu().item())
        return metrics

    if method_name in OMD_METHODS:
        updated_weights = extra_outputs.get("updated_omd_weights")
        if isinstance(updated_weights, torch.Tensor) and updated_weights.numel() >= 2:
            metrics["OMD retain weight"] = float(updated_weights[0].detach().cpu().item())
            metrics["OMD forget weight"] = float(updated_weights[1].detach().cpu().item())
        metrics["OMD eta"] = float(extra_outputs.get("eta", 0.0))
        return metrics

    if method_name == "chebyshev":
        active_task = extra_outputs.get("active_task")
        if active_task is not None:
            metrics["Chebyshev active task"] = int(active_task)
        max_term = _as_scalar(extra_outputs.get("max_term"))
        if max_term is not None:
            metrics["Chebyshev max term"] = max_term
        augmentation = _as_scalar(extra_outputs.get("augmentation"))
        if augmentation is not None:
            metrics["Chebyshev augmentation"] = augmentation
    return metrics


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


def certain_label(
    class_to_forget,
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
    criteria = torch.nn.MSELoss()

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

        name = f"compvis-cl-mask-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
    else:
        name = f"compvis-cl-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}-random_{random.randint(0, 1000)}"
    if args.mtl:
        weight_methods_parameters = extract_weight_method_parameters_from_args(args)
        method_kwargs = dict(weight_methods_parameters[args.mtl_method])
        if args.mtl_method == "eu":
            method_kwargs.update(dict(w_lr=weight_learning_rate_eu, error=error_eu))
            name += f"-mtl_{args.mtl_method}-w_lr_{weight_learning_rate_eu}-err_{error_eu}"
        else:
            name += f"-mtl_{args.mtl_method}"
        weight_method = WeightMethods(args.mtl_method, n_tasks=2, device=device, **method_kwargs)
    # TRAINING CODE
    for epoch in range(epochs):
        remain_iter = iter(remain_dl)
        with tqdm(total=len(forget_dl)) as time:
            for i, (images, labels) in enumerate(forget_dl):
                optimizer.zero_grad()

                forget_images, forget_labels = images, labels
                try:
                    remain_images, remain_labels = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_labels = next(remain_iter)
                torch.cuda.empty_cache()
                #import pdb
                #pdb.set_trace()
                forget_prompts = [descriptions[label] for label in forget_labels]

                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + random.randint(1,9)) % 10]
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
                remain_loss_alpha = alpha * remain_loss
                torch.cuda.empty_cache()

                #loss = forget_loss + alpha * remain_loss
                #loss.backward()
                #print(f"forget_loss: {forget_loss.item() / batch_size}, remain_loss: {remain_loss.item() / batch_size}")
                if args.mtl:
                    loss, method_outputs = weight_method.backward(
                        losses=torch.stack([remain_loss_alpha, forget_loss]),
                        shared_parameters=list(model.model.diffusion_model.parameters()),
                    )
                else:
                    method_outputs = {}
                    loss = forget_loss + remain_loss_alpha
                    loss.backward()
                torch.cuda.empty_cache()
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
                        torch.cuda.empty_cache()
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
        default=128,
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
    parser.add_argument(
        "--eu_w_lr",
        default=3.0,
        type=float,
        help="learning rate for the EU weight update",
    )
    parser.add_argument(
        "--eu_error",
        default=0.0,
        type=float,
        help="target retain-loss error used by EU",
    )
    parser.add_argument(
        "--weight_init",
        default=0.0,
        type=float,
        help="initial EU task weight",
    )
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

    weight_learning_rate_eu = (
        wandb_weight_learning_rate_eu
        if wandb_weight_learning_rate_eu is not None
        else args.eu_w_lr
    )
    error_eu = wandb_error_eu if wandb_error_eu is not None else args.eu_error
    print(f"EU Weight learning rate: {weight_learning_rate_eu}, EU Error: {error_eu}")

    # classes = [int(d) for d in args.classes.split(',')]
    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
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
        alpha,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
    )
wandb.finish()
