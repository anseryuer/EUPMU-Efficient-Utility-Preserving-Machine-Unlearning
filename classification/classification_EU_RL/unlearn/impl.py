import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from pruner import extract_mask, prune_model_custom, remove_prune
import json

def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    save_dir_final = args.save_dir
    os.makedirs(save_dir_final, exist_ok=True)
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, save_dir_final, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        save_dir_final,
        args.unlearn,
        filename="eval_result.pth.tar",
    )

    data = json.dumps(evaluation_result)
    with open(os.path.join(save_dir_final, "evaluation_result.json"), 'w') as f:
        f.write(data)


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _update_running_average(avg_state, model, step_count):
    model_state = model.state_dict()
    if avg_state is None:
        return {name: tensor.detach().clone() for name, tensor in model_state.items()}

    for name, tensor in model_state.items():
        if not torch.is_floating_point(tensor):
            avg_state[name] = tensor.detach().clone()
            continue
        avg_state[name].mul_((step_count - 1) / step_count).add_(tensor.detach() / step_count)
    return avg_state


def update_omd_tch_average(args, model):
    omd_methods = {"omd_tch", "omd_tch_eg", "omd_tch_pgd", "afleg", "afl"}
    if not (getattr(args, "mtl", False) and getattr(args, "mtl_method", None) in omd_methods):
        return

    avg_step_count = getattr(args, "omd_tch_avg_steps", 0) + 1
    avg_state = getattr(args, "omd_tch_avg_state", None)
    avg_state = _update_running_average(avg_state, model, avg_step_count)
    setattr(args, "omd_tch_avg_state", avg_state)
    setattr(args, "omd_tch_avg_steps", avg_step_count)


def _clone_model_state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def update_ada_omd_marked_state(args, model, losses):
    ada_methods = {"ada_omd_tch_eg", "ada_afleg"}
    if not (getattr(args, "mtl", False) and getattr(args, "mtl_method", None) in ada_methods):
        return

    current_losses = [float(loss) for loss in losses]
    marked_losses = getattr(args, "ada_omd_marked_losses", None)
    marked_model_params = getattr(args, "ada_omd_marked_model_params", None)
    marked_model_weights = getattr(args, "ada_omd_marked_model_weights", None)

    if marked_losses is None or marked_model_params is None or marked_model_weights is None:
        marked_losses = []
        marked_model_params = []
        marked_model_weights = []

    if len(marked_losses) == 0:
        marked_losses.append(current_losses)
        marked_model_params.append(_clone_model_state(model))
        marked_model_weights.append(1.0)
    else:
        marked_array = np.array(marked_losses)
        compare_res = np.all(np.array(current_losses) >= marked_array, axis=1)
        row_ids = np.where(compare_res)[0]
        if len(row_ids) != 0:
            share = 1.0 / len(row_ids)
            for idx in row_ids:
                marked_model_weights[idx] += share
        else:
            compare_res = np.all(np.array(current_losses) <= marked_array, axis=1)
            remove_ids = np.where(compare_res)[0]
            new_weight = 1.0
            for idx in remove_ids[::-1]:
                new_weight += marked_model_weights[idx]
                del marked_losses[idx]
                del marked_model_params[idx]
                del marked_model_weights[idx]
            marked_losses.append(current_losses)
            marked_model_params.append(_clone_model_state(model))
            marked_model_weights.append(new_weight)

    total_weight = float(sum(marked_model_weights))
    result_state = None
    for weight, state_dict in zip(marked_model_weights, marked_model_params):
        if result_state is None:
            result_state = {name: tensor.detach().clone() * (weight / total_weight) for name, tensor in state_dict.items()}
            continue
        for name, tensor in state_dict.items():
            if torch.is_floating_point(result_state[name]):
                result_state[name].add_(tensor.detach() * (weight / total_weight))
            else:
                result_state[name] = tensor.detach().clone()

    setattr(args, "ada_omd_marked_losses", marked_losses)
    setattr(args, "ada_omd_marked_model_params", marked_model_params)
    setattr(args, "ada_omd_marked_model_weights", marked_model_weights)
    setattr(args, "ada_omd_result_state", result_state)


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, device=None, weight_method=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=device
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)
    
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()

        avg_state = None
        avg_step_count = 0
        if args.mtl and getattr(args, "mtl_method", None) in {"omd_tch", "omd_tch_eg", "omd_tch_pgd", "afleg", "afl"}:
            setattr(args, "omd_tch_avg_state", None)
            setattr(args, "omd_tch_avg_steps", 0)
        if args.mtl and getattr(args, "mtl_method", None) in {"ada_omd_tch_eg", "ada_afleg"}:
            setattr(args, "ada_omd_marked_losses", [])
            setattr(args, "ada_omd_marked_model_params", [])
            setattr(args, "ada_omd_marked_model_weights", [])
            setattr(args, "ada_omd_result_state", None)

        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args, mask, device=device, weight_method=weight_method, **kwargs
            )

            # OMD-TCH averaging is handled inside the batch loop so it matches the paper's
            # online-to-batch averaging over optimization iterates rather than epoch endpoints.

            scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))
            print("Train Acc:{}".format(round(train_acc,2)))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
