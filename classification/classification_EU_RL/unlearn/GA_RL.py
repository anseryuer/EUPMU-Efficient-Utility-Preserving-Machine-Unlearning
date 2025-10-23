import pdb
import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn
import wandb
import random
from torch.utils.data import Dataset, Subset


@iterative_unlearn
def GA_RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None, device=None, weight_method=None):
    forget_loader = data_loaders["forget"]
    retain_loader = deepcopy(data_loaders["retain"])
    forget_dataset = deepcopy(forget_loader.dataset)

    # if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "TinyImagenet":
    if True:
        if  args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "TinyImagenet":
            try:
                # forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
                print("not using random targets")
                ...
            except:
                print(forget_dataset.dataset.targets[:10])
                print("not using random targets")
                # forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
                print(forget_dataset.dataset.targets[:10])
        else:
            print("not using random targets")
            forget_dataset.labels = np.random.randint(0, args.num_classes, forget_dataset.labels.shape)

        retain_dataset = retain_loader.dataset

        if args.retainwithAllParamUpdate or args.mtl:
            try:
                forget_dataset.targets = list(zip(forget_dataset.targets, len(forget_dataset) * ["forget"]))
                retain_dataset.targets = list(zip(retain_dataset.targets, len(retain_dataset) * ["retain"]))
            except:
                forget_dataset.temp = len(forget_dataset) * ["forget"]
                retain_dataset.temp = len(retain_dataset) * ["retain"]

        if args.only_trainForgetSet:
            train_dataset = forget_dataset
        elif args.only_trainForgetSet_and_samesizeOfretain:
            indices = random.sample(range(len(retain_dataset)), len(forget_dataset))
            subset_retain_dataset = Subset(retain_dataset, indices)
            train_dataset = torch.utils.data.ConcatDataset([forget_dataset, subset_retain_dataset])
        else:
            train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        from weighted_methods.GDR_GMA import MemoryBank
        memory_bank = MemoryBank(size=10) # Or another size

        # switch to train mode
        model.train()

        start = time.time()
        if args.only_trainForgetSet:
            loader_len = len(forget_loader)
        elif args.only_trainForgetSet_and_samesizeOfretain:
            loader_len = len(forget_loader) * 2
        else:
            loader_len = len(forget_loader) + len(retain_loader)
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i + 1, optimizer,
                            one_epoch_step=loader_len, args=args)

        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.to(device)

            if args.retainwithAllParamUpdate or args.mtl:
                target_label=target[1]
                target=target[0]
            target = target.to(device)

            if args.arch == "clip":
                image = preprocess(image)
                # Calculate features
                image_features = model.encode_image(image)
                with torch.no_grad():
                    text_features = model.encode_text(text_inputs)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                output_clean = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            else:
                # compute output
                output_clean = model(image)

            if args.mtl:
                retain_indexes = [index for index, value in enumerate(target_label) if value == "retain"]
                forget_indexes = [index for index, value in enumerate(target_label) if value == "forget"]
                loss_retain = criterion(output_clean[retain_indexes], target[retain_indexes])*(len(retain_indexes)/len(target_label))
                loss_forget = - criterion(output_clean[forget_indexes], target[forget_indexes])*(len(forget_indexes)/len(target_label))

                loss, extra_outputs = weight_method.backward(
                    losses=torch.stack([loss_retain, loss_forget]),
                    shared_parameters=list(model.parameters()),
                    # Pass the required kwargs
                    bank=memory_bank,
                    epoch=epoch,
                    n_loss=loss_retain
                )

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                optimizer.step()

                if ("famo" in args.mtl_method):
                    with torch.no_grad():
                        output_clean_ = model(image)
                        loss_retain = criterion(output_clean_[retain_indexes], target[retain_indexes]) * (
                                    len(retain_indexes) / len(target_label))
                        loss_forget = criterion(output_clean_[forget_indexes], target[forget_indexes]) * (
                                    len(forget_indexes) / len(target_label))
                        new_losses = torch.stack(
                            (
                                loss_retain,
                                loss_forget,
                            )
                        )
                        weight_method.method.update(new_losses.detach())
                if ("eu" == args.mtl_method):
                    with torch.no_grad():
                        output_clean_ = model(image)
                        loss_retain = criterion(output_clean_[retain_indexes], target[retain_indexes]) * (
                                    len(retain_indexes) / len(target_label))
                        weight_method.method.update(loss_retain.detach())
                        # wandb.log({"EU_weight": weight_method.method.w})


            elif args.retainwithAllParamUpdate:
                print("retainwithAllParamUpdate")
                retain_indexes = [index for index, value in enumerate(target_label) if value == "retain"]
                forget_indexes = [index for index, value in enumerate(target_label) if value == "forget"]
                loss_retain = criterion(output_clean[retain_indexes], target[retain_indexes])*(len(retain_indexes)/len(target_label))
                loss_forget = - criterion(output_clean[forget_indexes], target[forget_indexes])*(len(forget_indexes)/len(target_label))

                optimizer.zero_grad()
                loss_forget.backward(retain_graph=True)

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                loss_retain.backward()

                optimizer.step()
                loss =  loss_forget+loss_retain
            else:
                # print("not retainwithAllParamUpdate")
                loss = criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                    epoch, i, loader_len, end - start, loss=losses, top1=top1))
                start = time.time()

    # elif args.dataset == "svhn":
    #     losses = utils.AverageMeter()
    #     top1 = utils.AverageMeter()
    #
    #     # switch to train mode
    #     model.train()
    #
    #     start = time.time()
    #
    #     loader_len = len(forget_loader) + len(retain_loader)
    #
    #     if epoch < args.warmup:
    #         utils.warmup_lr(epoch, i + 1, optimizer,
    #                         one_epoch_step=loader_len, args=args)
    #
    #     for i, (image, target) in enumerate(forget_loader):
    #         image = image.to(device)
    #         target = torch.randint(0, args.num_classes, target.shape).to(device)
    #
    #         # compute output
    #         output_clean = model(image)
    #         loss = criterion(output_clean, target)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         if mask:
    #             for name, param in model.named_parameters():
    #                 if param.grad is not None:
    #                     param.grad *= mask[name]
    #
    #         optimizer.step()
    #
    #     for i, (image, target) in enumerate(retain_loader):
    #         image = image.to(device)
    #         target = target.to(device)
    #
    #         # compute output
    #         output_clean = model(image)
    #         loss = criterion(output_clean, target)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         if mask:
    #             for name, param in model.named_parameters():
    #                 if param.grad is not None:
    #                     param.grad *= mask[name]
    #
    #         optimizer.step()
    #
    #         output = output_clean.float()
    #         loss = loss.float()
    #         # measure accuracy and record loss
    #         prec1 = utils.accuracy(output.data, target)[0]
    #
    #         losses.update(loss.item(), image.size(0))
    #         top1.update(prec1.item(), image.size(0))
    #
    #         if (i + 1) % args.print_freq == 0:
    #             end = time.time()
    #             print('Epoch: [{0}][{1}/{2}]\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
    #                   'Time {3:.2f}'.format(
    #                 epoch, i, loader_len, end - start, loss=losses, top1=top1))
    #             start = time.time()

    if wandb.run is not None:
        lrl = [param_group['lr'] for param_group in optimizer.param_groups if param_group["params"] != []]
        lr = sum(lrl) / len(lrl)
        wandb.log({"lr": lr}, step=epoch)
        wandb.log({"Train Top1 Acc": top1.avg}, step=epoch)
        wandb.log({"Train Loss": losses.avg}, step=epoch)
        wandb.log({"epoch": epoch})

    return top1.avg