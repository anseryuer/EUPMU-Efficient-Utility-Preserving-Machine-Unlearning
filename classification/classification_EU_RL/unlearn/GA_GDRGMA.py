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
def GA_gdr_gma(data_loaders, model, criterion, optimizer, epoch, args, mask = None, device=None, weight_method=None, **kwargs):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()

    if args.imagenet_arch:
        # This part is for imagenet and is not modified to handle both loaders for now.
        # It will need a similar adaptation as the 'else' block if imagenet is used with this method.
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        train_loader = forget_loader # Fallback for imagenet
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        retain_iterator = iter(retain_loader)
        forget_iterator = iter(forget_loader)
        
        # We will iterate min(len(retain_loader), len(forget_loader)) times
        # This is a simple way to handle datasets of different sizes.
        num_batches = min(len(retain_loader), len(forget_loader))

        for i in range(num_batches):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=num_batches, args=args
                )

            try:
                retain_image, retain_target = next(retain_iterator)
            except StopIteration:
                retain_iterator = iter(retain_loader)
                retain_image, retain_target = next(retain_iterator)

            try:
                forget_image, forget_target = next(forget_iterator)
            except StopIteration:
                forget_iterator = iter(forget_loader)
                forget_image, forget_target = next(forget_iterator)

            retain_image, retain_target = retain_image.cuda(), retain_target.cuda()
            forget_image, forget_target = forget_image.cuda(), forget_target.cuda()

            optimizer.zero_grad()

            # Retain loss
            output_retain = model(retain_image)
            loss_retain = criterion(output_retain, retain_target)

            # Forget loss (gradient ascent)
            output_forget = model(forget_image)
            loss_forget = -criterion(output_forget, forget_target)

            if weight_method:
                
                weight_method.backward(losses=torch.stack((loss_retain, loss_forget)), 
                                                 shared_parameters=list(model.parameters()))
            else:
                # Default behavior if no weight_method is provided: sum the losses.
                loss = loss_retain + loss_forget
                loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            # For logging purposes, we can track the retain accuracy.
            output = output_retain.float()
            loss = loss_retain.float() # Log retain loss
            prec1 = utils.accuracy(output.data, retain_target)[0]

            losses.update(loss.item(), retain_image.size(0))
            top1.update(prec1.item(), retain_image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, num_batches, end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@iterative_unlearn
def GA_l1(data_loaders, model, criterion, optimizer, epoch, args, device=None, weight_method=None):
    train_loader = data_loaders["forget"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target) + args.alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
