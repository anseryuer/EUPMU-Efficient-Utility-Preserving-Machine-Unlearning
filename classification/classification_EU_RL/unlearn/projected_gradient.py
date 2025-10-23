import time
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.utils.data import ConcatDataset, DataLoader

from .impl import iterative_unlearn
from .unlearn_utils import get_feature_dict, compute_svd

@iterative_unlearn
def projected_gradient_unlearning(data_loaders, model, criterion, optimizer, epoch, args, mask = None, device=None, weight_method=None, **kwargs):
    """
    Unlearning through gradient projection, implemented in an iterative, per-epoch fashion.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
    else:
        device = torch.device("cpu")
    # 1. On the first epoch, compute the SVD and the projection matrix.
    if not hasattr(args, 'projection_matrix'):
        print("First epoch: computing SVD and projection matrix...")
        retain_loader = data_loaders["retain"]
        
        # Get feature dictionaries for the model
        conv_fea_dict, linear_fea_dict = get_feature_dict(args.arch)
        
        # Compute SVD on the retain set
        svd_results = compute_svd(model, retain_loader, conv_fea_dict, linear_fea_dict)
        
        # Compute and cache the projection matrix P
        P = {}
        retained_var = getattr(args, 'retained_var', 0.99)
        for layer in svd_results:
            k = torch.sum((torch.cumsum(svd_results[layer]['S'], dim=0) / torch.sum(svd_results[layer]['S'])) < retained_var)
            M_tmp = svd_results[layer]['U'][:, :k]
            P[layer] = torch.mm(M_tmp, M_tmp.t()).to(device).float()
            print(f'Layer: {layer} - Projection Matrix Shape: {P[layer].shape} - Retained Variance: {retained_var} - k: {k}')
        
        args.projection_matrix = P
        print("Projection matrix computed and cached in args.")

    # Retrieve the cached projection matrix
    P = args.projection_matrix

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    losses_forget = utils.AverageMeter()
    losses_retain = utils.AverageMeter()

    # Switch to train mode
    model.train()

    start = time.time()
    
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    # The number of batches is determined by the smaller of the two loaders for fair comparison
    num_batches = min(len(retain_loader), len(forget_loader))
    
    retain_iterator = iter(retain_loader)
    forget_iterator = iter(forget_loader)

    weight = getattr(args, 'weight', 0.5)

    for i in range(num_batches):
        retain_inputs, retain_targets = next(retain_iterator)
        forget_inputs, forget_targets = next(forget_iterator)

        retain_inputs, retain_targets = retain_inputs.cuda(), retain_targets.cuda()
        forget_inputs, forget_targets = forget_inputs.cuda(), forget_targets.cuda()

        # Standard loss for retain set
        output_retain = model(retain_inputs)
        loss_retain = criterion(output_retain, retain_targets)

        # Random labeling for forget set to encourage forgetting
        forget_targets_random = (forget_targets.cpu() + np.random.randint(1, args.num_classes, forget_targets.shape[0])) % args.num_classes
        forget_targets_random = forget_targets_random.cuda()
        output_forget = model(forget_inputs)
        loss_forget = criterion(output_forget, forget_targets_random)

        # Combine losses
        loss = weight / (1 + weight) * loss_retain + 1 / (1 + weight) * loss_forget
        
        optimizer.zero_grad()
        loss.backward()

        # Project gradients before optimizer step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Find the correct projection matrix to use for the current parameter
                    P_name = None
                    for layer_name in P.keys():
                        if name.startswith(layer_name):
                            P_name = layer_name
                            break
                    
                    # Some layers might not be in P (e.g., batchnorm), so we check
                    if P_name is not None and P_name in P:
                        # For conv layers, we need to handle the projection differently
                        if 'conv' in name or 'layer' in name:
                            # Reshape gradient to match the feature space projection
                            original_shape = param.grad.data.shape
                            if len(original_shape) == 4:  # Conv layer weights
                                # Flatten spatial dimensions but keep channel structure
                                grad_reshaped = param.grad.data.view(original_shape[0], -1)
                                # Apply projection if dimensions match
                                if grad_reshaped.shape[1] == P[P_name].shape[0]:
                                    projected_grad = grad_reshaped - torch.mm(grad_reshaped, P[P_name])
                                    param.grad.data = projected_grad.view(original_shape)
                            elif len(original_shape) == 2:  # Linear layer weights
                                grad_flat = param.grad.data
                                if len(grad_flat.shape) >= 2 and grad_flat.shape[1] == P[P_name].shape[0]:
                                    projected_grad = grad_flat - torch.mm(grad_flat, P[P_name])
                                    param.grad.data = projected_grad
                        elif 'fc' in name and P_name == 'fc':
                            # For final linear layer
                            grad_flat = param.grad.data
                            if len(grad_flat.shape) >= 2 and grad_flat.shape[1] == P[P_name].shape[0]:
                                projected_grad = grad_flat - torch.mm(grad_flat, P[P_name])
                                param.grad.data = projected_grad

        optimizer.step()

        # Logging
        prec1, _ = utils.accuracy(output_retain, retain_targets, topk=(1, 5))
        losses.update(loss.item(), retain_inputs.size(0))
        top1.update(prec1.item(), retain_inputs.size(0))
        losses_forget.update(loss_forget.item(), forget_inputs.size(0))
        losses_retain.update(loss_retain.item(), retain_inputs.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Forget Loss {forget_loss.val:.4f} ({forget_loss.avg:.4f})\t"
                "Retain Loss {retain_loss.val:.4f} ({retain_loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch,
                    i,
                    num_batches,
                    end - start,
                    loss=losses,
                    forget_loss=losses_forget,
                    retain_loss=losses_retain,
                    top1=top1,
                )
            )
            start = time.time()
            
            
    return top1.avg
