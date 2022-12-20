import abc
from enum import Enum
from typing import NamedTuple

import numpy as np
import torch
from foolbox.distances import LpDistance

from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.image_utils import encode_decode
from src.model_wrappers import ModelWrapper


class Bounds(NamedTuple):
    lower: float = 0.
    upper: float = 1.


class BaseAttack(abc.ABC):

    def __init__(self, epsilon: float | None, distance: LpDistance, bounds: Bounds, discrete: bool):
        self.epsilon = epsilon
        self.discrete = discrete
        self.bounds = bounds
        self.distance = distance

    @staticmethod
    def is_correct_boundary_side(model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None,
                                 queries_counter: QueriesCounter,
                                 attack_phase: AttackPhase) -> tuple[torch.Tensor, QueriesCounter]:
        if target is not None:
            success = model.predict_label(x) == target
        else:
            success = model.predict_label(x) != y
        return success, queries_counter.increase(attack_phase, safe=success)  # type: ignore

    def clamp_and_discretize(self, out: torch.Tensor) -> torch.Tensor:
        out = torch.clamp(out, self.bounds.lower, self.bounds.upper)
        if self.discrete:
            assert torch.allclose(out, torch.round(out * 255) / 255)  # type: ignore
            decoded_out = encode_decode(out)
            assert torch.allclose(out, decoded_out, atol=1 / 256)
            out = decoded_out
        return out

    @abc.abstractmethod
    def __call__(self,
                 model: ModelWrapper,
                 x: torch.Tensor,
                 label: torch.Tensor,
                 target: torch.Tensor | None = None,
                 query_limit: int = 10_000) -> tuple[torch.Tensor, QueriesCounter, float, bool, dict[str, float | int]]:
        ...


class DirectionAttackPhase(AttackPhase):
    search = "search"
    initialization = "initialization"
    direction_probing = "direction_probing"


class DirectionAttack(BaseAttack, abc.ABC):
    """
    Base class for attacks which optimize a direction instead of the perturbation directly
    """

    def get_x_adv(self, x: torch.Tensor, v: torch.Tensor, d: float) -> torch.Tensor:
        if self.discrete and not np.isinf(d):
            assert int(d) == d
            d = d / 255
        out: torch.Tensor = x + d * v  # type: ignore
        out = self.clamp_and_discretize(out)
        return out


class PerturbationAttack(BaseAttack, abc.ABC):

    def get_x_adv(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.discrete:
            assert torch.round(delta) == delta
            delta = delta / 255
        out: torch.Tensor = x + delta  # type: ignore
        out = self.clamp_and_discretize(out)
        return out


class SearchMode(str, Enum):
    binary = "binary"
    line = "line"
