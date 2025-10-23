from trainer import train

from .impl import iterative_unlearn


@iterative_unlearn
def retrain(data_loaders, model, criterion, optimizer, epoch, args, mask, device=None, weight_method=None, **kwargs):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, device=device, mask=mask)
