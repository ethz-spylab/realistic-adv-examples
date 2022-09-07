from functools import partial
from turtle import forward
from typing import Callable, Tuple

import functorch as ft
import torch
from torch import nn

DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

GradAndValue = Tuple[torch.Tensor, torch.Tensor]
ScoreFn = Callable[[torch.Tensor, torch.Tensor], GradAndValue]
ChangeOfVarsFn = Callable[[torch.Tensor], torch.Tensor]
Bounds = Tuple[float, float]


def onehot_like(x: torch.Tensor, indices: torch.Tensor, *, value: float = 1) -> torch.Tensor:
    """From
    https://github.com/jonasrauber/eagerpy/blob/master/eagerpy/tensor/pytorch.py#L274"""
    if x.ndim != 2:
        raise ValueError("onehot_like only supported for 2D tensors")
    if indices.ndim != 1:
        raise ValueError("onehot_like requires 1D indices")
    if len(indices) != len(x):
        raise ValueError("length of indices must match length of tensor")
    x = torch.zeros_like(x)
    rows = torch.arange(x.shape[0])
    x[rows, indices] = value

    return x


def cw_to_attack_space(x: torch.Tensor, bounds: Bounds, eps: float = 1e-6) -> torch.Tensor:
    """From
    https://github.com/bethgelab/foolbox/blob/12abe74e2f1ec79edb759454458ad8dd9ce84939/foolbox/attacks/carlini_wagner.py#L207"""
    min_, max_ = bounds
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b  # map from [min_, max_] to [-1, +1]
    x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
    return torch.arctan(x)  # from (-1, +1) to (-inf, +inf)


def cw_to_model_space(x: torch.Tensor, bounds: Bounds, eps: float = 1e-6) -> torch.Tensor:
    """From
    https://github.com/bethgelab/foolbox/blob/12abe74e2f1ec79edb759454458ad8dd9ce84939/foolbox/attacks/carlini_wagner.py#L217"""
    min_, max_ = bounds
    x = torch.tanh(x)  # from (-inf, +inf) to (-1, +1)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    return x * b + a  # map from (-1, +1) to (min_, max_)


class CWAdvLoss(nn.Module):
    k: torch.Tensor

    def __init__(self, k: float = 0.) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor([k]))

    def forward(self, logits: torch.Tensor, y: torch.Tensor):
        other_classes_logits = logits - onehot_like(logits, y, value=torch.inf)
        best_other_classes, _ = torch.max(other_classes_logits, dim=-1)
        correct_classes = logits[torch.arange(logits.size(0)), y]
        return torch.maximum(correct_classes - best_other_classes, self.k)


def find_boundary_interpolation(x: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor, model: nn.Module,
                                step_size: float, max_steps: int) -> Tuple[torch.Tensor, int]:
    boundary_x = x_adv.detach().clone()
    alpha = torch.tensor([0.], device=x.device)
    step_size_ = torch.tensor([step_size], device=x.device)

    for step in range(max_steps + 1):
        new_boundary_x = alpha * x + (1 - alpha) * x_adv
        if model(new_boundary_x).argmax(-1).item() == y.item():
            return boundary_x, step - 1
        boundary_x = new_boundary_x
        alpha = alpha + step_size_

    raise ValueError(f"Could not cross the boundary in the given number of steps ({max_steps})."
                     f"Try with a larger step size (current: {step_size}).")


def estimate_gradient_interpolation(x: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor, model: nn.Module,
                                    n_points: int, distance_fn: DistanceFn, step_size: float,
                                    max_steps: int) -> torch.Tensor:
    random_samples = torch.clamp(x_adv.detach() + torch.empty_like(x_adv).repeat(n_points, 1, 1, 1).uniform_(), 0, 1)
    gradient_est = torch.zeros_like(x)
    total_distances = 0

    for sample in random_samples:
        x_boundary, _ = find_boundary_interpolation(x, sample, y, model, step_size, max_steps)
        distance = distance_fn(x_adv, x_boundary)
        gradient_est = gradient_est + distance * x_boundary
        total_distances = total_distances + distance

    return (gradient_est / total_distances).unsqueeze(0)


def find_boundary_direction(x_adv: torch.Tensor, direction: torch.Tensor, y: torch.Tensor, model: nn.Module,
                            step_size: float, max_steps: int):
    boundary_x = x_adv.detach().clone()
    direction_mult = torch.tensor([0.], device=boundary_x.device)
    step_size_ = torch.tensor([step_size], device=boundary_x.device)

    for step in range(max_steps):
        new_boundary_x = torch.clamp(direction_mult * direction + x_adv, 0, 1)
        if model(new_boundary_x).argmax(-1).item() == y.item():
            return boundary_x, step - 1
        boundary_x = new_boundary_x
        direction_mult = direction_mult + step_size_

    return boundary_x, max_steps + 1


def estimate_gradient_direction(x: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor, model: nn.Module, n_points: int,
                                distance_fn: DistanceFn, step_size: float, max_steps: int) -> torch.Tensor:
    random_directions = torch.empty_like(x_adv).repeat(n_points, 1, 1, 1).uniform_()
    total_distances = 0
    conf_component = torch.zeros_like(x)

    for direction in random_directions:
        x_boundary, _ = find_boundary_direction(x, direction, y, model, step_size, max_steps)
        distance = distance_fn(x_adv, x_boundary)
        conf_component = conf_component + distance * direction
        total_distances = total_distances + distance

    return conf_component / total_distances


def grad_exact(x: torch.Tensor, y: torch.Tensor, model: nn.Module, loss_fn: nn.Module) -> GradAndValue:
    x.requires_grad_(True)
    loss = loss_fn(model(x), y)
    grad = torch.autograd.grad(loss.unsqueeze(0), x)[0]
    return grad, loss


class BoundaryHitException(Exception):
    ...


def multi_point_grad_nes(f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, y: torch.Tensor,
                         num_samples: int, radius: float) -> GradAndValue:
    """From
    https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/tools/helpers.py#L20"""
    B, *_ = x.shape
    Q = num_samples // 2
    N = len(x.shape) - 1
    with torch.no_grad():
        # Q * B * C * H * W
        extender = [1] * N
        queries = x.repeat(Q, *extender)
        noise = torch.randn_like(queries)
        norm = noise.view(B * Q, -1).norm(dim=-1).view(B * Q, *extender)
        noise = noise / norm
        noise = torch.cat([-noise, noise])
        queries = torch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        loss = f(queries + radius * noise, y.repeat(2 * Q, *y_shape)).view(-1, *extender)
        if (loss == 0.).any():
            raise BoundaryHitException()
        grad = (loss.view(2 * Q, B, *extender) * noise.view(2 * Q, B, *noise.shape[1:])).mean(dim=0)

    return grad, f(x, y)


def score_based_attack(x: torch.Tensor,
                       x_target: torch.Tensor,
                       y_target: torch.Tensor,
                       model: nn.Module,
                       distance_fn: DistanceFn,
                       grad_and_score_fn: ScoreFn,
                       attack_steps: int,
                       step_size: float,
                       conf_factor: float,
                       dist_factor: float,
                       bounds: Bounds = (0, 1),
                       log_freq: int = 100) -> Tuple[torch.Tensor, int]:
    delta = torch.zeros_like(x_target)

    to_model_space = partial(cw_to_model_space, bounds=bounds)
    to_attack_space = partial(cw_to_attack_space, bounds=bounds)

    x_attack = to_attack_space(x_target)
    grad_and_distance_fn = ft.grad_and_value(distance_fn)

    for step in range(attack_steps):
        iter_x = to_model_space(x_attack + delta)
        if model(iter_x).argmax(-1).item() != y_target.item():
            print("Hit boundary after update, returning earlier")
            return iter_x, step
        try:
            # Compute score and (approx of) gradient of the score
            score_grad_est, score_est = grad_and_score_fn(iter_x, y_target)
        except BoundaryHitException:
            print("Hit boundary in gradient estimation, returning earlier")
            return iter_x, step
        # Compute distance and gradient of the distance
        distance_grad, distance = grad_and_distance_fn(iter_x, x)
        # Compute overall gradient and score
        tot_grad = dist_factor * distance_grad - conf_factor * score_grad_est
        f = dist_factor * distance - conf_factor * score_est
        # Update delta
        delta = delta - step_size * tot_grad
        if step % log_freq == 0:
            print(f"step {step}. f = {f.item()}, score = {score_est.item()}, distance = {distance.item()}")

    return to_model_space(x_attack + delta), attack_steps


def score_based_attack_exact_loss(x: torch.Tensor, x_target: torch.Tensor, y_target: torch.Tensor, model: nn.Module,
                                  distance_fn: DistanceFn, attack_steps: int, step_size: float, conf_factor: float,
                                  dist_factor: float, loss: nn.Module, n_points: int,
                                  smooth_radius: float) -> Tuple[torch.Tensor, int]:
    loss_fn = lambda x_, y_: loss(model(x_), y_)
    grad_and_score_fn = lambda x_, y_: multi_point_grad_nes(loss_fn, x_, y_, n_points, smooth_radius)
    return score_based_attack(x, x_target, y_target, model, distance_fn, grad_and_score_fn, attack_steps, step_size,
                              conf_factor, dist_factor)


def gradient_based_attack(x: torch.Tensor, x_target: torch.Tensor, y_target: torch.Tensor, model: nn.Module,
                          distance_fn: DistanceFn, attack_steps: int, step_size: float, conf_factor: float,
                          dist_factor: float, loss: nn.Module) -> Tuple[torch.Tensor, int]:
    grad_and_score_fn = lambda x_, y_: grad_exact(x_, y_, model, loss)
    return score_based_attack(x, x_target, y_target, model, distance_fn, grad_and_score_fn, attack_steps, step_size,
                              conf_factor, dist_factor)
