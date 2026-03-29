import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


import torch
import torch.nn.functional as F
from itertools import combinations
import gc
try:
    import wandb
except:
    pass
EPS = 1e-8 # for numerical stability
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if kwargs["scaler"] != None:
                if i < self.n_tasks - 1:
                    scaled_grad_params = torch.autograd.grad(outputs=kwargs["scaler"].scale(losses[i]),
                                             inputs=shared_parameters,
                                             retain_graph=True)
                    inv_scale = 1. / kwargs["scaler"].get_scale()
                    gs = [p * inv_scale for p in scaled_grad_params]
                else:
                    scaled_grad_params = torch.autograd.grad(outputs=kwargs["scaler"].scale(losses[i]),
                                             inputs=shared_parameters,
                                             )
                    inv_scale = 1. / kwargs["scaler"].get_scale()
                    gs = [p * inv_scale for p in scaled_grad_params]
            else:
                if i < self.n_tasks-1:
                    gs = torch.autograd.grad(outputs=losses[i],
                                             inputs=shared_parameters,
                                             retain_graph=True)
                else:
                    gs = torch.autograd.grad(outputs=losses[i],
                                             inputs=shared_parameters,
                                             )
            self.grad2vec(gs, grads, grad_dims, i)

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        # del grads
        # del g
        # del gs
        # del grad_dims
        # cleanup()

        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(gs, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for grad in gs:
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters, **kwargs)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return torch.mean(losses), {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods





class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 1e-5,
            w_lr: float = 0.3,
            task_weights: Union[List[float], torch.Tensor] = None,
            max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()


class EU(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.025,   # the learning rate of the task lambda
        max_norm: float = 1.0, # the maximum gradient norm
        error: float = 0.001, # the error term

    ):
        print("!!EU ACTIVATED!!")
        super().__init__(2, device=device)
        self.min_losses = torch.zeros(2).to(device)
        self.w = torch.tensor([0.], device=device, requires_grad=True)
        try:
            w_lr = wandb.config.weight_learning_rate_eu
        except:
            print("No wandb weight learning rate found")
            pass
        print("w_lr", w_lr, "error", error)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = 2
        self.device = device
        try:
            self.error = wandb.config.error_eu
        except:
            self.error = error

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses,**kwargs,):
        self.prev_ret_loss = losses[0]
        D = losses - self.min_losses + 1e-8
        D_log = D.log()
        D_copy = D_log.clone()
        if self.w < 0:
            D_copy[0] = D_log[0] * 0
        else:
            D_copy[0] = D_log[0] * (self.w/(1 + self.w))
            D_copy[1] = D_log[1] * (1/(1 + self.w))
        loss = D_copy.sum()
        torch.cuda.empty_cache()
        return loss, {"weights": self.w.detach().clone(), "logits": self.w.detach().clone()}

    def update(self, curr_ret_loss):
        delta = (self.prev_ret_loss - self.min_losses[0] + 1e-8).log() - \
                (curr_ret_loss      - self.min_losses[0] + 1e-8).log() - self.error
        d = delta.unsqueeze(0)
        #print("delta", delta)
        self.w_opt.zero_grad(set_to_none=False)
        #print(d, delta, self.prev_ret_loss, curr_ret_loss)
        #print(self.w.grad.size(), d.size())
        self.w.grad = d
        self.w_opt.step()
        #print("Updated weight", self.w.detach().clone())

class Chebyshev(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights=None,
        reference_point=None,
        rho: float = 1e-3,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device, max_norm=max_norm)
        if task_weights is None:
            task_weights = [1.0] * n_tasks
        if reference_point is None:
            reference_point = [0.0] * n_tasks
        if len(task_weights) != n_tasks:
            raise ValueError(f"Expected {n_tasks} task weights, got {len(task_weights)}")
        if len(reference_point) != n_tasks:
            raise ValueError(f"Expected {n_tasks} reference values, got {len(reference_point)}")

        self.task_weights = torch.tensor(task_weights, dtype=torch.float32, device=device)
        self.reference_point = torch.tensor(reference_point, dtype=torch.float32, device=device)
        self.rho = rho
        self.max_norm = max_norm

    def get_weighted_loss(self, losses, **kwargs):
        task_weights = self.task_weights.to(losses.device, losses.dtype)
        reference_point = self.reference_point.to(losses.device, losses.dtype)
        deviations = torch.clamp(losses - reference_point, min=0.0)
        weighted_deviations = task_weights * deviations
        max_term, active_idx = torch.max(weighted_deviations, dim=0)
        augmentation = self.rho * weighted_deviations.sum()
        loss = max_term + augmentation
        extra_outputs = {
            "weights": task_weights.detach().clone(),
            "reference_point": reference_point.detach().clone(),
            "deviations": deviations.detach().clone(),
            "weighted_deviations": weighted_deviations.detach().clone(),
            "active_task": int(active_idx.detach().item()),
            "rho": self.rho,
            "max_term": max_term.detach().clone(),
            "augmentation": augmentation.detach().clone(),
        }
        return loss, extra_outputs


class OMDTCHBase(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights=None,
        reference_point=None,
        eta: float = 0.1,
        rho: float = 0.0,
        update_rule: str = "eg",
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device, max_norm=max_norm)
        if task_weights is None:
            task_weights = [1.0] * n_tasks
        if reference_point is None:
            reference_point = [0.0] * n_tasks
        if len(task_weights) != n_tasks:
            raise ValueError(f"Expected {n_tasks} task weights, got {len(task_weights)}")
        if len(reference_point) != n_tasks:
            raise ValueError(f"Expected {n_tasks} reference values, got {len(reference_point)}")
        if eta <= 0:
            raise ValueError("eta must be positive for OMD-TCH")
        if update_rule not in {"eg", "pgd"}:
            raise ValueError("update_rule must be 'eg' or 'pgd'")
        if any(float(v) != 0.0 for v in reference_point):
            raise ValueError("OMD-TCH in this repo follows the paper and requires zero reference_point.")
        if float(rho) != 0.0:
            raise ValueError("OMD-TCH in this repo follows the paper and does not use rho augmentation.")

        self.task_weights = torch.tensor(task_weights, dtype=torch.float32, device=device)
        self.eta = eta
        self.update_rule = update_rule
        self.simplex_weights = torch.full((n_tasks,), 1.0 / n_tasks, dtype=torch.float32, device=device)
        self.dual_state = torch.zeros(n_tasks, dtype=torch.float32, device=device) if update_rule == "eg" else None

    @property
    def variant_name(self) -> str:
        return self.update_rule

    def _weighted_losses(self, losses):
        task_weights = self.task_weights.to(losses.device, losses.dtype)
        weighted_losses = task_weights * losses
        return task_weights, weighted_losses

    def _project_simplex(self, values: torch.Tensor) -> torch.Tensor:
        sorted_vals, _ = torch.sort(values, descending=True)
        cssv = torch.cumsum(sorted_vals, dim=0) - 1.0
        ind = torch.arange(1, values.numel() + 1, device=values.device, dtype=values.dtype)
        cond = sorted_vals - cssv / ind > 0
        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho.to(values.dtype) + 1.0)
        projected = torch.clamp(values - theta, min=0.0)
        return projected / projected.sum().clamp_min(1e-12)

    def get_weighted_loss(self, losses, **kwargs):
        task_weights, weighted_losses = self._weighted_losses(losses)
        if self.update_rule == "eg":
            simplex_weights = torch.softmax(self.dual_state.to(losses.device, losses.dtype), dim=0)
        else:
            simplex_weights = self.simplex_weights.to(losses.device, losses.dtype)
        scalarized = torch.sum(simplex_weights.detach() * weighted_losses)

        with torch.no_grad():
            scores = weighted_losses.detach().to(self.device, torch.float32)
            if self.update_rule == "eg":
                self.dual_state = self.dual_state + self.eta * scores.to(self.dual_state.device, self.dual_state.dtype)
                updated_weights = torch.softmax(self.dual_state, dim=0)
            else:
                updated = self.simplex_weights + self.eta * scores.to(self.simplex_weights.device, self.simplex_weights.dtype)
                updated_weights = self._project_simplex(updated)
            self.simplex_weights = updated_weights.detach().clone()

        extra_outputs = {
            "task_weights": task_weights.detach().clone(),
            "weighted_losses": weighted_losses.detach().clone(),
            "omd_weights": simplex_weights.detach().clone(),
            "updated_omd_weights": updated_weights.detach().clone(),
            "eta": self.eta,
            "variant": self.variant_name,
            "scalarized_term": scalarized.detach().clone(),
        }
        return scalarized, extra_outputs


class OMDTCHEG(OMDTCHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, update_rule="eg", **kwargs)


class OMDTCHPGD(OMDTCHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, update_rule="pgd", **kwargs)


METHODS = dict(
    cagrad=CAGrad,
    famo=FAMO,
    eu = EU,
    chebyshev = Chebyshev,
    omd_tch = OMDTCHEG,
    omd_tch_eg = OMDTCHEG,
    omd_tch_pgd = OMDTCHPGD,
    afleg = OMDTCHEG,
    afl = OMDTCHPGD,
)
