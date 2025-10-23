import numpy as np
import torch
import torch.nn as nn
from functools import singledispatch
import gc


def prune_by_percentile_gradient_perCell(model, time_para=1,reverse=False):
    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "norm" in name:
            new_mask = torch.zeros_like(param.data, device=param.data.device)
        elif "bias" in name:
            new_mask = torch.ones_like(param.data, device=param.data.device)
        elif len(param.shape)==1:
            new_mask = torch.ones_like(param.data, device=param.data.device)
        else:
            if len(param.shape)==4:
                tensor = param.grad.data
                B,C,H,W = tensor.shape
                tensor = tensor.reshape((B,-1))
            else:
                tensor = param.grad.data

            if time_para > tensor.size(1):
                time_para_=tensor.size(1)
            else:
                time_para_=time_para


            if reverse:
                topk_values, topk_indices = torch.topk(abs(tensor), time_para_, dim=1, largest=False)
            else:
                topk_values, topk_indices = torch.topk(abs(tensor), time_para_, dim=1, largest=True)
            new_mask = torch.zeros_like(tensor, device=param.data.device)
            new_mask.scatter_(1, topk_indices, 1)


            if len(param.shape)==4:
                new_mask = new_mask.reshape((B,C,H,W))

        trainable_param = len(torch.nonzero(new_mask))
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   , flush=True)

        new_masks[name] = new_mask


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    for na, [trainable_p, t_p] in statistic.items():
        trainable_withouthead = trainable_withouthead + trainable_p
        total_withouthead = total_withouthead + t_p

    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (total): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")

    print("#######################################################################")
    return new_masks





def prune_by_percentile_gradient_allLayer(model, percent_pruning_min, percent_pruning_max, tuning_head=None, select_head=False):
    # Calculate percentile value

    alive_all = np.array([])
    for name, param in model.named_parameters():
        device = param.device
        tensor = param.grad.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
        alive_all = np.concatenate([alive_all, alive])

    percentile_value_min = np.percentile(abs(alive_all), percent_pruning_min)
    percentile_value_max = np.percentile(abs(alive_all), percent_pruning_max)


    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "fc" in name and not select_head:
            if tuning_head:
                new_mask = np.ones_like(param.data.cpu().numpy())
            else:
                new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            tensor = param.grad.data.cpu().numpy()
            old_mask = np.ones_like(param.data.cpu().numpy())
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) <= percentile_value_max), 0, old_mask)

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).to(device)


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0.000001
    for na, [trainable_p, t_p] in statistic.items():
        if "fc" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks
