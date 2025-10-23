import sys
import time

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


@iterative_unlearn
def GA_mtl(data_loaders, model, criterion, optimizer, epoch, args, mask = None, device=None, weight_method=None, **kwargs):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    
    forget_iterator = iter(forget_loader)
    retain_iterator = iter(retain_loader)

    # In each epoch, we iterate through both forget and retain loaders
    # The number of iterations is the max of the two loader lengths
    for i in range(max(len(forget_loader), len(retain_loader))):
        try:
            image_forget, target_forget = next(forget_iterator)
        except StopIteration:
            # Reset iterator if it's exhausted
            forget_iterator = iter(forget_loader)
            image_forget, target_forget = next(forget_iterator)
        
        try:
            image_retain, target_retain = next(retain_iterator)
        except StopIteration:
            # Reset iterator if it's exhausted
            retain_iterator = iter(retain_loader)
            image_retain, target_retain = next(retain_iterator)

        if epoch < args.warmup:
            # Assuming one_epoch_step is the total number of iterations in an epoch
            one_epoch_step = max(len(forget_loader), len(retain_loader))
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=one_epoch_step, args=args
            )

        image_forget, target_forget = image_forget.to(device), target_forget.to(device)
        image_retain, target_retain = image_retain.to(device), target_retain.to(device)

        # compute output
        output_forget = model(image_forget)
        loss_forget = -criterion(output_forget, target_forget) # Gradient ascent for forget set

        output_retain = model(image_retain)
        loss_retain = criterion(output_retain, target_retain) # Gradient descent for retain set

        # Use weight_method to combine losses
        if weight_method:
            loss, _ = weight_method.backward(torch.stack([loss_retain, loss_forget]), **kwargs)
        else:
            # Default behavior if no weight_method is provided (e.g., simple sum)
            loss = loss_retain + loss_forget
            optimizer.zero_grad()
            loss.backward()
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.step()

        # measure accuracy and record loss for retain set
        prec1 = utils.accuracy(output_retain.data, target_retain)[0]
        losses.update(loss_retain.item(), image_retain.size(0))
        top1.update(prec1.item(), image_retain.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, max(len(forget_loader), len(retain_loader)), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
