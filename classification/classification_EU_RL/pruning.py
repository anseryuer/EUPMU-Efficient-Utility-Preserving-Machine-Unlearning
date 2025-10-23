import numpy as np
import torch
import torch.nn as nn
from functools import singledispatch
import gc


def prune_by_percentile_gradient_perCell(model, time_para=1,reverse=False, tuning_head=None, select_head=False):
    statistic = {}
    new_masks = {}
    for name, param in model.named_parameters():
        if "bn" in name:
            new_mask = torch.zeros_like(param.data, device=param.data.device)
        elif "fc" in name and not select_head:
            if tuning_head:
                new_mask = torch.ones_like(param.data, device=param.data.device)
            else:
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
    trainable_head = 0
    total_head = 0.00001
    for na, [trainable_p, t_p] in statistic.items():
        if "fc" not in na and "classifier.4.weight" not in na:
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

    # statics_perBlock(statistic)

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

    # statics_perBlock(statistic)
    return new_masks


def statics_perBlock(statistic):
    print("#######################################################################")

    layers_s={"layer1":0,
            "layer2":0,
            "layer3":0,
            "layer4":0,
            }
    all_parameters_s={"layer1":0,
            "layer2":0,
            "layer3":0,
            "layer4":0,
            }
    all_parameters=0
    trainable_parameter_all=0
    for na, [trainable_p, t_p] in statistic.items():
        if "layer1" in na:
            layers_s["layer1"]+=trainable_p
            all_parameters_s["layer1"]+=t_p
            all_parameters+=t_p
            trainable_parameter_all+=trainable_p
        elif "layer2" in na:
            layers_s["layer2"]+=trainable_p
            all_parameters_s["layer2"]+=t_p
            all_parameters += t_p
            trainable_parameter_all+=trainable_p
        elif "layer3" in na:
            layers_s["layer3"]+=trainable_p
            all_parameters_s["layer3"]+=t_p
            all_parameters += t_p
            trainable_parameter_all+=trainable_p
        elif "layer4" in na:
            layers_s["layer4"]+=trainable_p
            all_parameters_s["layer4"]+=t_p
            all_parameters += t_p
            trainable_parameter_all+=trainable_p


    for n, num in layers_s.items():
        print(f"Out of all tunable parameters: {num}/{trainable_parameter_all}, {round(num/trainable_parameter_all, 4)*100}%")
    print("------------------")
    for n, num in layers_s.items():
        print(f"Out of specific layer parameters: {num}/{all_parameters_s[n]}, {round(num/all_parameters_s[n], 4)*100}%")



    print("#######################################################################")