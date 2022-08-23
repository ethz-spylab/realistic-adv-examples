import pytest
import torch
from torch import nn

from src.attack import estimate_gradient
from src.attack import find_boundary


class DummyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.values = torch.Tensor([1.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            torch.heaviside(x, self.values),
            torch.heaviside(-x, 0 * self.values)
        ],
                         dim=0)


def test_find_boundary():
    x = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.2
    distance_fn = nn.L1Loss()
    max_steps = 10
    boundary_x, distance, steps = find_boundary(x, x_adv, y, model, step_size,
                                                distance_fn, max_steps)
    assert pytest.approx(boundary_x.item(), abs=1e-6) == 0
    assert pytest.approx(distance.item(), abs=1e-6) == 1
    assert steps == 5


def test_find_boundary_max_steps():
    x = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.2
    distance_fn = nn.L1Loss()
    max_steps = 2
    with pytest.raises(ValueError):
        _ = find_boundary(x, x_adv, y, model, step_size, distance_fn, max_steps)


def test_estimate_gradient():
    x = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.2
    distance_fn = nn.L1Loss()
    max_steps = 10
    n_points = 10
    eps = 0.01
    gradient_estimation = estimate_gradient(x, x_adv, y, model, eps, n_points,
                                            distance_fn, step_size, max_steps)
    assert gradient_estimation.sign().item() == -1
