import torch
from abc import abstractmethod
from typing import List, Tuple, Union


class WeightMethod:
    """Minimal base class inlined from classification/weighted_methods/weight_methods.py."""

    def __init__(self, n_tasks: int, device: torch.device, max_norm: float = 1.0):
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(self, losses, **kwargs):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
    ) -> Tuple[torch.Tensor, dict]:
        loss, extra_outputs = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss, extra_outputs


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


# Backward-compatible alias
OMDTCH = OMDTCHEG
