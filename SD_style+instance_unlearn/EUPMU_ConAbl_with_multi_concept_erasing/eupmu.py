import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union


class EU:
    """
    Fast Adaptive Multitask Optimization.
    """
    def __init__(
        self,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 10,   # the learning rate of the task lambda
        max_norm: float = 1.0, # the maximum gradient norm
        error: float = 0.1, # the error term

    ):
        self.min_losses = torch.zeros(2).to(device)
        self.w = torch.tensor([0.], device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = 2
        self.device = device
        self.error = error
        self.use_log = True
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, ret_loss, fgt_loss):
        losses = torch.stack([ret_loss, fgt_loss]).to(self.device)
        self.prev_ret_loss = ret_loss
        D = losses - self.min_losses + 1e-10
        if self.use_log:
            D_log = D.log()
        else:
            D_log = D
        D_copy = D_log.clone()
        if self.w < 0:
            D_copy[0] = D_log[0] * 0
        else:
            D_copy[0] = D_log[0] * (self.w/(1 + self.w))
            D_copy[1] = D_log[1] * (1/(1 + self.w))
        loss = D_copy.sum()
        return loss

    def update(self, curr_ret_loss):
        if self.use_log:
            delta = (self.prev_ret_loss - self.min_losses[0] + 1e-10).log() - \
                    (curr_ret_loss      - self.min_losses[0] + 1e-10).log() - self.error
        else:
            delta = (self.prev_ret_loss - self.min_losses[0] + 1e-10) - \
                    (curr_ret_loss      - self.min_losses[0] + 1e-10) - self.error
        # delta = (self.prev_ret_loss - self.min_losses[0] + 1e-10).log() - \
        #         (curr_ret_loss      - self.min_losses[0] + 1e-10).log() - self.error
        d = delta.unsqueeze(0)# * F.softmax(self.w, -1)
        self.w_opt.zero_grad()
        #print(d, delta, self.prev_ret_loss, curr_ret_loss)
        #assert False
        print(f" d: {d.cpu().detach().numpy()}, before update loss: {self.prev_ret_loss.cpu().detach().numpy()}, current loss: {curr_ret_loss.cpu().detach().numpy()}")
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        not_backward: bool = False,
    ) -> Union[torch.Tensor, None]:
        """
        Modified to work with Accelerator by ensuring gradients are properly detached.
        """
        loss = self.get_weighted_loss(losses[0], losses[1])
        print(f"losses before weight update {losses.cpu().detach().numpy()}, weighted loss: {loss.item()}, weights: {self.w.cpu().detach().numpy()}")
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        if not_backward:
            return loss
        loss.backward()
        return loss
    
if __name__ == "__main__":

    n   = 1000 # number of datapoints
    dim = 20   # dimension of data
    K   = 2  # number of tasks
    X = torch.randn(n, dim)
    Y = torch.randn(n, K)
    Y[:,0] = 2*X[:,0] + 3*X[:,1] + 4*X[:,2]
    Y[:,1] = 3*X[:,0] + 2*X[:,1] + 4*X[:,3]
    model = torch.nn.Linear(dim, K)
    weight_opt = EU(device="cpu")
    opt = torch.optim.Adam(model.parameters())
    for it in range(250):
        loss = (Y[:,0] - model(X)[:,0]).pow(2).mean(0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    for it in range(250):
        loss = (Y - model(X)).pow(2).mean(0) # (K,)
        opt.zero_grad()
        weight_opt.backward(loss)
        opt.step()
        # update the task weighting
        with torch.no_grad():
            new_loss = (Y - model(X)).pow(2).mean(0) # (K,)
            weight_opt.update(new_loss[0])
        print(f"[info] iter {it:3d} | avg loss {loss.mean().item():.4f}")