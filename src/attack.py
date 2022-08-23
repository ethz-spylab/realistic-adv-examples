from typing import Callable, Tuple

import torch
from torch import nn

DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def find_boundary(x: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor,
                  model: nn.Module, step_size: float, distance_fn: DistanceFn,
                  max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    boundary_x = x_adv.detach().clone()
    previous_boundary_x = x_adv.detach().clone()

    steps = 0
    for step in range(max_steps):
        boundary_x.requires_grad_(True)
        previous_boundary_x = boundary_x
        distance = distance_fn(boundary_x, x)
        grad = torch.autograd.grad(distance, boundary_x)[0]
        boundary_x = boundary_x.detach() - step_size * grad
        steps += 1
        if model(boundary_x).argmax(-1).item() == y.item():
            return previous_boundary_x, distance, step

    raise ValueError(
        f"Could not cross the boundary in the given steps ({max_steps})."
        f"Try with a larger step size (current: {step_size}).")


def estimate_gradient(x: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor,
                      model: nn.Module, eps: float, n_points: int,
                      distance_fn: DistanceFn, step_size: float,
                      max_steps: int) -> torch.Tensor:
    random_samples = x_adv.detach() + torch.empty_like(x_adv).repeat(n_points).normal_() * eps
    distances = []
    for sample in random_samples:
        _, distance, _ = find_boundary(x, sample, y, model, step_size,
                                       distance_fn, max_steps)
        distances.append(distance)

    distances_t = torch.as_tensor(distances, device=random_samples.device)
    gradient_estimation = (random_samples * distances_t).mean()

    return gradient_estimation / n_points
