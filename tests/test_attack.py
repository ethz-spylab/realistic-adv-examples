import pytest
import torch
from torch import nn

from src.attacks import estimate_gradient_interpolation
from src.attacks import find_boundary_direction
from src.attacks import find_boundary_interpolation


class DummyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.values = torch.Tensor([1.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.heaviside(x, 0 * self.values), torch.heaviside(-x, self.values)], dim=0)


def test_find_boundary_interpolation():
    x = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.1
    distance_fn = nn.L1Loss()
    max_steps = 10
    boundary_x, steps = find_boundary_interpolation(x, x_adv, y, model, step_size, max_steps)
    assert pytest.approx(boundary_x.item(), abs=1e-6) == 0
    assert steps == 5

    distance = distance_fn(boundary_x, x_adv)
    assert pytest.approx(distance.item(), abs=1e-6) == 1


def test_find_boundary_interpolation_max_steps():
    x = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.1
    max_steps = 2
    with pytest.raises(ValueError):
        _ = find_boundary_interpolation(x, x_adv, y, model, step_size, max_steps)


def test_boundary_direction():
    direction = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.2
    distance_fn = nn.L1Loss()
    max_steps = 10
    boundary_x, steps = find_boundary_direction(x_adv, direction, y, model, step_size, max_steps)
    assert pytest.approx(boundary_x.item(), abs=1e-6) == 0
    assert steps == 5

    distance = distance_fn(boundary_x, x_adv)
    assert pytest.approx(distance.item(), abs=1e-6) == 1


def test_find_boundary_direction_max_steps():
    direction = torch.Tensor([1])
    x_adv = torch.Tensor([-1])
    y = torch.Tensor([0])
    model = DummyModel()
    step_size = 0.2
    max_steps = 2
    _, steps = find_boundary_direction(x_adv, direction, y, model, step_size, max_steps)
    assert steps == max_steps + 1
