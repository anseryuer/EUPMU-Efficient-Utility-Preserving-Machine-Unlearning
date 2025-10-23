from .GA import GA,GA_l1
from .RL import RL
from .FT import FT,FT_l1
from .FT_prune import FT_prune
from .retrain import retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .Wfisher import Wfisher

from .RL_pro import RL_proximal

from .GA_GDRGMA import GA_gdr_gma
from .GA_RL import GA_RL
from .GA_mtl import GA_mtl
from .projected_gradient import projected_gradient_unlearning

from .boundary_sh import boundary_shrink
from .boundary_ex import boundary_expanding

def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "FT_prune":
        return FT_prune
    elif name == "retrain":
        return retrain
    elif name == "wfisher":
        return Wfisher
    elif name == "GA_l1":
        return GA_l1
    elif name == "GA_RL":
        return GA_RL
    elif name == "GA_mtl":
        return GA_mtl
    elif name == "RL_proximal":
        return RL_proximal
    elif name == "projected_gradient_unlearning":
        return projected_gradient_unlearning
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == "boundary_expanding":
        return boundary_expanding
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
