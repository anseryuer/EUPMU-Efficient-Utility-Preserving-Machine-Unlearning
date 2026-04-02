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
except ImportError:
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
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
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
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
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
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return torch.mean(losses), {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class GDR_GMA(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm
        print(f"GDR_GMA initialized with c={c} and max_norm={max_norm}")

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Implements the full GDR-GMA algorithm: memory bank, gradient rectification, lambda-weighted combination.
        Assumes losses[0] = retain, losses[1] = forget.
        kwargs can include: bank (memory bank), epoch, n_loss (retain loss)
        """
        bank = kwargs.get('bank', None)
        epoch = kwargs.get('epoch', 0)
        n_loss = kwargs.get('n_loss', None)
        gamma = kwargs.get('gamma', 100)
        epsilon = kwargs.get('epsilon', 0.02)

        grad_dims = [param.data.numel() for param in shared_parameters]
        grads = torch.zeros(sum(grad_dims), self.n_tasks, device=self.device)

        # Compute gradients for each task
        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            for p in shared_parameters:
                p.grad = None

        # Update memory bank with forget gradient (last param vector)
        if bank is not None:
            bank.update(grads[-1, 1].detach().clone())

        # Rectify gradients
        r_n_grads, r_t_grads = self.rectify_gradient(grads[:, 0], grads[:, 1])

        # If using memory bank, further rectify forget gradient with mean
        if bank is not None and epoch > 0:
            mean_grad = bank.mean_grads(r_t_grads)
            if mean_grad is not None:
                r_t_grads = self.rectify_gradient(r_t_grads, mean_grad)[0]

        # Compute lambda_weight
        if n_loss is not None:
            with torch.no_grad():
                lambda_weight = 1 / (1 + torch.exp(torch.tensor(gamma, dtype=torch.float32, device=self.device) * (n_loss - epsilon)))
        else:
            lambda_weight = 0.5  # fallback

        # Final convex combination
        new_grad = (1 - lambda_weight) * r_n_grads + lambda_weight * r_t_grads
        self.overwrite_grad(shared_parameters, new_grad, grad_dims)

        GTG = grads.t().mm(grads).cpu().numpy()
        w_cpu = np.array([float(1 - lambda_weight), float(lambda_weight)])
        return GTG, w_cpu

    @staticmethod
    def rectify_gradient(x, y):
        # x, y: 1D tensors (flattened gradients)
        # Returns rectified x, y as in the original code
        if torch.cosine_similarity(x, y, dim=0) < 0:
            InP_xy = torch.dot(y, x)
            Inp_xx = torch.norm(x, p=2) ** 2
            Inp_yy = torch.norm(y, p=2) ** 2
            x = x - InP_xy / Inp_yy * y
            y = y - InP_xy / Inp_xx * x
        return x, y

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        grads[:, task].fill_(0.0)
        cnt = 0
        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        cnt = 0
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
        # n_loss should be passed in kwargs for lambda_weight calculation
        GTG, w = self.get_weighted_loss(losses, shared_parameters, **kwargs)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return torch.mean(losses), {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        grads[:, task].fill_(0.0)
        cnt = 0
        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        cnt = 0
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
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return torch.mean(losses), {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class ImplicitGradientSurgery(WeightMethod):
    """Implements the closed-form direction from Proposition 3.2 for two-task unlearning."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        epsilon: float = 0.0,
        max_norm: float = 1.0,
    ):
        assert (
            n_tasks == 2
        ), "ImplicitGradientSurgery expects exactly two tasks: retain (0) and forget (1)."
        super().__init__(n_tasks, device=device, max_norm=max_norm)
        self.default_epsilon = float(epsilon)

    def get_weighted_loss(self, losses, shared_parameters, **kwargs):
        epsilon_value = kwargs.get("epsilon", self.default_epsilon)

        grad_dims = [param.data.numel() for param in shared_parameters]
        grads = torch.zeros(sum(grad_dims), self.n_tasks, device=self.device)

        for task_idx in range(self.n_tasks):
            retain_graph = task_idx < self.n_tasks - 1
            losses[task_idx].backward(retain_graph=retain_graph)
            self.grad2vec(shared_parameters, grads, grad_dims, task_idx)
            for param in shared_parameters:
                param.grad = None

        retain_grad = grads[:, 0]
        forget_grad = grads[:, 1]

        eps_tensor = torch.as_tensor(
            epsilon_value, device=self.device, dtype=retain_grad.dtype
        )

        denom = torch.dot(retain_grad, retain_grad) + EPS
        lambda_star = (-torch.dot(retain_grad, forget_grad) - eps_tensor) / denom

        if lambda_star.detach().item() > 0:
            combined_grad = forget_grad + lambda_star * retain_grad
            applied_lambda = lambda_star.detach().item()
        else:
            combined_grad = forget_grad
            applied_lambda = 0.0

        self.overwrite_grad(shared_parameters, combined_grad, grad_dims)

        extras = {
            "lambda_star": lambda_star.detach().item(),
            "applied_lambda": applied_lambda,
            "epsilon": float(eps_tensor.detach().cpu().item()),
        }
        return None, extras

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        grads[:, task].fill_(0.0)
        cnt = 0
        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    @staticmethod
    def overwrite_grad(shared_parameters, newgrad, grad_dims):
        cnt = 0
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            reshaped = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = reshaped.data.clone()
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
        _, extras = self.get_weighted_loss(losses, shared_parameters, **kwargs)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return torch.mean(losses), extras


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
            w_lr: float = 0.025,
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
        w_lr: float = 1,   # the learning rate of the task lambda
        max_norm: float = 1.0, # the maximum gradient norm
        error: float = 0.01, # the error term
        weight_init: float = 0.0, # initial weight

    ):
        super().__init__(2, device=device)
        self.min_losses = torch.zeros(2).to(device)
        self.w = torch.tensor([weight_init], device=device, requires_grad=True)
        with torch.no_grad():
            self.w.clamp_(min=0.0)

        try:
            w_lr = wandb.config.weight_learning_rate_eu
        except:
            print("No wandb weight learning rate found")
            pass
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = 2
        self.device = device
        try:
            self.error = wandb.config.error_eu
        except:
            self.error = error
        print(f"EU initialized with error {self.error} and w_lr {w_lr} and weight_init {weight_init}")

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
            D_copy[0] = D_log[0] * self.w
        loss = D_copy.sum()
        # print(f"EU: current retention loss {self.prev_ret_loss}, current forget loss {D_log}, weight {self.w}")
        return loss, {"weights": self.w.detach().clone(), "logits": self.w.detach().clone()}

    def update(self, curr_ret_loss):
        delta = (self.prev_ret_loss - self.min_losses[0] + 1e-8).log() - \
                (curr_ret_loss      - self.min_losses[0] + 1e-8).log() - self.error
        d = delta.unsqueeze(0)
        self.w_opt.zero_grad(set_to_none=False)
        #print(d, delta, self.prev_ret_loss, curr_ret_loss)
        #print(self.w.grad.size(), d.size())
        self.w.grad = d
        self.w_opt.step()
        with torch.no_grad():
            self.w.clamp_(min=0.0)


class EU_fast(WeightMethod):
    """EU method with fast update mechanism, using the next step's result for update."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 1,   # the learning rate of the task lambda
        max_norm: float = 1.0, # the maximum gradient norm
        error: float = 0.01, # the error term
        weight_init: float = 1.0, # initial weight

    ):
        super().__init__(2, device=device)
        self.min_losses = torch.zeros(2).to(device)
        self.w = torch.tensor([weight_init], device=device, requires_grad=True)
        with torch.no_grad():
            self.w.clamp_(min=0.0)

        try:
            w_lr = wandb.config.weight_learning_rate_eu
        except:
            print("No wandb weight learning rate found")
            pass
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma, betas=(0.9, 0.999))
        self.max_norm = max_norm
        self.n_tasks = 2
        self.device = device
        try:
            self.error = wandb.config.error_eu
        except:
            self.error = error
        print(f"EU initialized with error {self.error} and w_lr {w_lr}")

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses,**kwargs,):
        # Update weight `w` based on the previous step's retention loss and the current one.
        if hasattr(self, 'prev_ret_loss'):
            # Calculate the change in retention loss
            delta = (self.prev_ret_loss - self.min_losses[0] + 1e-8).log() - \
                    (losses[0].detach() - self.min_losses[0] + 1e-8).log() - self.error
            d = delta.unsqueeze(0)
            
            # Update the weight `w` for the *next* iteration
            self.w_opt.zero_grad(set_to_none=True)
            self.w.grad = d
            self.w_opt.step()
            with torch.no_grad():
                self.w.clamp_(min=0.0)


        # Calculate the current weighted loss using the current `w` (from the previous step)
        D = losses - self.min_losses + 1e-8
        D_log = D.log()
        D_copy = D_log.clone()
        
        current_w = self.w.detach().clone() # Use the weight before the update for this step's loss
        if current_w < 0:
            D_copy[0] = 0
        else:
            D_copy[0] = D_log[0] * current_w/(1 + current_w)
            D_copy[1] = D_log[1] * (1/(1 + current_w))

        loss = D_copy.sum()

        # Store the current retention loss for the next update cycle.
        self.prev_ret_loss = losses[0].detach().clone()
        # print(f"EU_fast: current retention loss {self.prev_ret_loss}, weight {current_w}")
        return loss, {"weights": current_w, "logits": current_w}

    # def update(self, curr_ret_loss):
    #     delta = (self.prev_ret_loss - self.min_losses[0] + 1e-8).log() - \
    #             (curr_ret_loss      - self.min_losses[0] + 1e-8).log() - self.error
    #     d = delta.unsqueeze(0)
    #     self.w_opt.zero_grad(set_to_none=False)
    #     #print(d, delta, self.prev_ret_loss, curr_ret_loss)
    #     #print(self.w.grad.size(), d.size())
    #     self.w.grad = d
    #     self.w_opt.step()


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
        adaptive: bool = False,
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
        self.adaptive = adaptive
        self.simplex_weights = torch.full((n_tasks,), 1.0 / n_tasks, dtype=torch.float32, device=device)

        if self.update_rule == "eg":
            self.dual_state = torch.zeros(n_tasks, dtype=torch.float32, device=device)
        else:
            self.dual_state = None

        self.grad_sum = None if adaptive else None
        self.initial_w = None if adaptive else None
        self.latest_task_grads = None

    @property
    def variant_name(self) -> str:
        if self.adaptive:
            return f"ada_{self.update_rule}"
        return self.update_rule

    def _weighted_losses(self, losses):
        task_weights = self.task_weights.to(losses.device, losses.dtype)
        weighted_losses = task_weights * losses
        return task_weights, weighted_losses

    def _project_simplex(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() != 1:
            raise ValueError("Simplex projection expects a 1D tensor.")
        sorted_vals, _ = torch.sort(values, descending=True)
        cssv = torch.cumsum(sorted_vals, dim=0) - 1.0
        ind = torch.arange(1, values.numel() + 1, device=values.device, dtype=values.dtype)
        cond = sorted_vals - cssv / ind > 0
        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho.to(values.dtype) + 1.0)
        projected = torch.clamp(values - theta, min=0.0)
        return projected / projected.sum().clamp_min(1e-12)

    def set_task_gradients(self, task_gradients: torch.Tensor):
        if not self.adaptive:
            return
        if task_gradients is None:
            self.latest_task_grads = None
            return
        self.latest_task_grads = task_gradients.detach().to(self.device, torch.float32)

    def _compute_scores(self, weighted_losses: torch.Tensor) -> torch.Tensor:
        if not self.adaptive:
            return weighted_losses.detach().to(self.device, torch.float32)

        if self.latest_task_grads is None:
            raise RuntimeError("AdaOMD-TCH requires per-task gradients; call set_task_gradients before backward.")

        grads = self.latest_task_grads.to(self.device, torch.float32)
        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(grads)

        scores = torch.zeros(self.n_tasks, device=self.device, dtype=torch.float32)
        for i in range(self.n_tasks):
            self.grad_sum[i] += grads[i] ** 2
            h_i = torch.sqrt(self.grad_sum[i] + 1e-7)
            scores[i] = torch.dot(grads[i] / h_i, grads[i])

        if self.initial_w is None:
            self.initial_w = scores.clone().clamp_min(1e-12)
        normalized = scores / self.initial_w.clamp_min(1e-12)
        self.latest_task_grads = None
        return normalized

    def _update_simplex_weights(self, scores: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
        if self.update_rule == "eg":
            self.dual_state = self.dual_state + self.eta * scores.to(self.dual_state.device, self.dual_state.dtype)
            updated_weights = torch.softmax(self.dual_state, dim=0)
        else:
            updated = self.simplex_weights + self.eta * scores.to(self.simplex_weights.device, self.simplex_weights.dtype)
            updated_weights = self._project_simplex(updated)
        self.simplex_weights = updated_weights.detach().clone()
        return updated_weights

    def get_weighted_loss(self, losses, **kwargs):
        task_weights, weighted_losses = self._weighted_losses(losses)
        if self.update_rule == "eg":
            simplex_weights = torch.softmax(self.dual_state.to(losses.device, losses.dtype), dim=0)
        else:
            simplex_weights = self.simplex_weights.to(losses.device, losses.dtype)
        scalarized = torch.sum(simplex_weights.detach() * weighted_losses)

        with torch.no_grad():
            scores = self._compute_scores(weighted_losses)
            updated_weights = self._update_simplex_weights(scores, losses)

        extra_outputs = {
            "task_weights": task_weights.detach().clone(),
            "weighted_losses": weighted_losses.detach().clone(),
            "omd_weights": simplex_weights.detach().clone(),
            "updated_omd_weights": updated_weights.detach().clone(),
            "eta": self.eta,
            "variant": self.variant_name,
            "scalarized_term": scalarized.detach().clone(),
        }
        if self.adaptive:
            extra_outputs["adaptive_scores"] = scores.detach().clone()
        return scalarized, extra_outputs


class OMDTCHEG(OMDTCHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, update_rule="eg", adaptive=False, **kwargs)


class OMDTCHPGD(OMDTCHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, update_rule="pgd", adaptive=False, **kwargs)


class AdaOMDTCHEG(OMDTCHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, update_rule="eg", adaptive=True, **kwargs)


# Backward-compatible alias: the original omd_tch entry now refers to the EG variant.
OMDTCH = OMDTCHEG


METHODS = dict(
    cagrad=CAGrad,
    famo=FAMO,
    eu = EU,
    eu_fast = EU_fast,
    gdr_gma = GDR_GMA,
    igs = ImplicitGradientSurgery,
    chebyshev = Chebyshev,
    omd_tch = OMDTCHEG,
    omd_tch_eg = OMDTCHEG,
    omd_tch_pgd = OMDTCHPGD,
    ada_omd_tch_eg = AdaOMDTCHEG,
    ada_afleg = AdaOMDTCHEG,
    afleg = OMDTCHEG,
    afl = OMDTCHPGD,
)
