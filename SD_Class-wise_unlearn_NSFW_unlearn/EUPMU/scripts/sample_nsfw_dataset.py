import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def load_model_from_config(config, ckpt, device):
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. "
            "Download it first (see scripts/download_models.sh)."
        )
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def save_batch(images, outdir, start_index, limit=None):
    os.makedirs(outdir, exist_ok=True)
    saved = 0
    for x in images:
        if limit is not None and saved >= limit:
            break
        img = Image.fromarray((x * 255).astype(np.uint8))
        img.save(os.path.join(outdir, f"{start_index + saved:05}.png"))
        saved += 1
    return saved


def generate(
    model,
    sampler,
    prompt,
    outdir,
    total,
    batch_size,
    seed,
    ddim_steps,
    ddim_eta,
    scale,
    precision,
    H,
    W,
    C,
    f,
    resume,
):
    os.makedirs(outdir, exist_ok=True)
    base_count = len([p for p in os.listdir(outdir) if p.lower().endswith(".png")]) if resume else 0
    remaining = max(0, total - base_count)
    if remaining == 0:
        print(f"{outdir} already has {base_count} images, skipping.")
        return

    device = next(model.parameters()).device
    precision_scope = autocast if precision == "autocast" else nullcontext

    print(f"Sampling {remaining} images for prompt: {prompt}")
    start_time = time.time()
    batch_index = 0
    saved_total = 0
    while saved_total < remaining:
        current_bs = min(batch_size, remaining - saved_total)
        seed_everything(seed + batch_index)
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(current_bs * [""])
        prompts = current_bs * [prompt]
        c = model.get_learned_conditioning(prompts)
        shape = [C, H // f, W // f]
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    samples, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=current_bs,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                    )
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        saved = save_batch(x_samples, outdir, base_count + saved_total, limit=current_bs)
        saved_total += saved
        batch_index += 1

    elapsed = (time.time() - start_time) / 60.0
    print(f"Saved {saved_total} images to {outdir} in {elapsed:.2f} minutes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsfw_prompt",
        type=str,
        default="a photo of a nude person",
        help="prompt for the NSFW concept (Df)",
    )
    parser.add_argument(
        "--safe_prompt",
        type=str,
        default="a photo of a person wearing clothes",
        help="prompt for the non-NSFW concept (Dr)",
    )
    parser.add_argument(
        "--nsfw_outdir",
        type=str,
        default="./data/nsfw",
        help="output directory for NSFW samples",
    )
    parser.add_argument(
        "--safe_outdir",
        type=str,
        default="./data/not-nsfw",
        help="output directory for non-NSFW samples",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=800,
        help="total number of images per prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per sampling step",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of DDIM steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM eta (0.0 = deterministic)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use PLMS sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action="store_true",
        help="use DPM-Solver sampling",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
        help="path to SD v1.4 checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="base seed (will increment per batch)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["full", "autocast"],
        default="autocast",
        help="evaluate at this precision",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from existing .png files in the output directories",
    )
    opt = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device=device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    generate(
        model=model,
        sampler=sampler,
        prompt=opt.nsfw_prompt,
        outdir=opt.nsfw_outdir,
        total=opt.total,
        batch_size=opt.batch_size,
        seed=opt.seed,
        ddim_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        scale=opt.scale,
        precision=opt.precision,
        H=opt.H,
        W=opt.W,
        C=opt.C,
        f=opt.f,
        resume=opt.resume,
    )

    generate(
        model=model,
        sampler=sampler,
        prompt=opt.safe_prompt,
        outdir=opt.safe_outdir,
        total=opt.total,
        batch_size=opt.batch_size,
        seed=opt.seed + 1000,
        ddim_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        scale=opt.scale,
        precision=opt.precision,
        H=opt.H,
        W=opt.W,
        C=opt.C,
        f=opt.f,
        resume=opt.resume,
    )


if __name__ == "__main__":
    main()
