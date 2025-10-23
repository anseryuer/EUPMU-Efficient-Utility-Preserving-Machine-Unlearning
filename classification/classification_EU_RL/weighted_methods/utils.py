import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .weight_methods import METHODS


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument(
    "--method", type=str, choices=list(METHODS.keys()), help="MTL weight method"
)
common_parser.add_argument(
    "--method-params-lr",
    type=float,
    default=0.025,
    help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")
# NashMTL
common_parser.add_argument(
    "--nashmtl-optim-niter", type=int, default=20, help="number of CCCP iterations"
)
common_parser.add_argument(
    "--update-weights-every",
    type=int,
    default=1,
    help="update task weights every x iterations.",
)
# stl
common_parser.add_argument(
    "--main-task",
    type=int,
    default=0,
    help="main task for stl. Ignored if method != stl",
)
# cagrad
common_parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
# famo
common_parser.add_argument("--gamma", type=float, default=0.01, help="gamma of famo")
common_parser.add_argument("--use_log", action='store_true', help="whether use log for famo")
common_parser.add_argument("--max_norm", type=float, default=1.0, help="beta for RMS_weight alg.")
common_parser.add_argument("--task", type=int, default=0, help="train single task number for (celeba)")
# dwa
common_parser.add_argument(
    "--dwa-temp",
    type=float,
    default=2.0,
    help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.",
)
#gps


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def extract_weight_method_parameters_from_args(args):
    """
    Extract the parameters of the weight method from the args.
    """
    method_parameters = defaultdict(dict)
    # eu
    method_parameters["eu"]["w_lr"] = args.eu_w_lr
    method_parameters["eu"]["error"] = args.eu_error
    if hasattr(args, "weight_init"):
        method_parameters["eu"]["weight_init"] = args.weight_init

    return method_parameters