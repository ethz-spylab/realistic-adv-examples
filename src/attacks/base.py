import abc
import math
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
    def __init__(self, distance: LpDistance, bounds: Bounds, discrete: bool):
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

    @abc.abstractmethod
    def __call__(self,
                 model: ModelWrapper,
                 x: torch.Tensor,
                 label: torch.Tensor,
                 target: torch.Tensor | None = None,
                 query_limit: int = 10_000) -> tuple[torch.Tensor, QueriesCounter, float, bool, dict[str, float]]:
        ...


class DirectionAttackPhase(AttackPhase):
    search = "search"
    initialization = "initialization"
    direction_probing = "direction_probing"


class DirectionAttack(BaseAttack, abc.ABC):
    """
    Base class for attacks which optimize a direction instead of the perturbation directly
    """
    init_line_search_radius = 10
    n_early_stopping = 0

    def __init__(self, distance: LpDistance, bounds: Bounds, discrete: bool, line_search_tol: float | None):
        super().__init__(distance, bounds, discrete)
        self.line_search_tol = line_search_tol

    @staticmethod
    def _check_input_size(x: torch.Tensor) -> None:
        if len(x.shape) != 4 or x.shape[0] != 1:
            raise ValueError("Search works only on batched inputs of batch size 1.")

    def get_x_adv(self, x: torch.Tensor, v: torch.Tensor, d: float) -> torch.Tensor:
        if self.discrete:
            assert int(d) == d
            d = d / 255
        out: torch.Tensor = x + d * v  # type: ignore
        out = torch.clamp(out, self.bounds.lower, self.bounds.upper)
        if self.discrete:
            assert torch.allclose(out, torch.round(out * 255) / 255)  # type: ignore
            decoded_out = encode_decode(out)
            assert torch.allclose(out, decoded_out, atol=1 / 256)
            out = decoded_out
        return out

    def initial_line_search(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None,
                            direction: torch.Tensor, queries_counter: QueriesCounter) -> tuple[float, QueriesCounter]:
        self._check_input_size(x)
        d_end = np.inf
        start = 1
        end = self.init_line_search_radius
        if self.discrete:
            start *= 255
            end *= 255

        updated_queries_counter = queries_counter
        for distance in range(start, end + 1):
            x_adv = self.get_x_adv(x, direction, distance)
            success, updated_queries_counter = self.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                             DirectionAttackPhase.direction_probing)
            if success.item():
                d_end = distance
                print("Found initial perturbation")
                break

        return d_end, updated_queries_counter

    def binary_search(self,
                      model: ModelWrapper,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      target: torch.Tensor | None,
                      direction: torch.Tensor,
                      best_distance: float,
                      queries_counter: QueriesCounter,
                      tol: float = 1e-3) -> tuple[float, QueriesCounter, bool]:
        self._check_input_size(x)
        stopped_early = False

        d_start = 0
        d_end, updated_queries_counter = self._init_search(model, x, y, target, best_distance, direction,
                                                           queries_counter)
        if np.isinf(d_end):
            return d_end, updated_queries_counter, stopped_early

        if not self.discrete:
            condition = lambda end, start: (end - start) > tol
        else:
            condition = lambda end, start: (end - start) > 1

        while condition(d_end, d_start):
            if not self.discrete:
                d_mid = (d_start + d_end) / 2.0
            else:
                d_mid = math.ceil((d_start + d_end) / 2.0)
            x_adv = self.get_x_adv(x, direction, d_mid)
            success, updated_queries_counter = self.is_correct_boundary_side(model, x_adv, y, target,
                                                                             updated_queries_counter,
                                                                             DirectionAttackPhase.search)
            if success.item():
                d_end = d_mid
            else:
                d_start = d_mid

        return d_end, updated_queries_counter, stopped_early

    def line_search(self,
                    model: ModelWrapper,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    target: torch.Tensor | None,
                    direction: torch.Tensor,
                    best_distance: torch.Tensor,
                    queries_counter: QueriesCounter,
                    max_steps=200) -> tuple[float, QueriesCounter, bool]:
        self._check_input_size(x)

        d_end, updated_queries_counter = self._init_search(model, x, y, target, best_distance, direction,
                                                           queries_counter)
        stopped_early = False
        if np.isinf(d_end):
            return d_end, updated_queries_counter, stopped_early

        if not self.discrete:
            step_size = d_end / max_steps
        else:
            step_size = math.ceil(d_end / max_steps)

        initial_d_end = d_end
        for i in range(1, max_steps):
            d_end_tmp = initial_d_end - step_size * i
            x_adv = self.get_x_adv(x, direction, d_end_tmp)
            success, updated_queries_counter = self.is_correct_boundary_side(model, x_adv, y, target,
                                                                             updated_queries_counter,
                                                                             DirectionAttackPhase.search)
            if not success.item():
                break
            d_end = d_end_tmp
            # Check whether we can early stop and save an unsafe query
            if self.line_search_tol is not None and 1 - (d_end / best_distance) >= self.line_search_tol:
                stopped_early = True
                break

        return d_end, updated_queries_counter, stopped_early

    def _init_search(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None,
                     best_distance: float, direction: torch.Tensor, queries_counter: QueriesCounter):
        # In case there is already the best distance, probe the direction at that distance
        if not np.isinf(best_distance):
            x_adv = self.get_x_adv(x, direction, best_distance)
            success, updated_queries_counter = self.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                             DirectionAttackPhase.direction_probing)
            # If the example is on the safe side then search, otherwise discard direction
            if success.item():
                d_end = best_distance
            else:
                d_end = np.inf
        # Otherwise initialize the best distance
        else:
            d_end, updated_queries_counter = self.initial_line_search(model, x, y, target, direction, queries_counter)
        return d_end, updated_queries_counter


class PerturbationAttack(BaseAttack, abc.ABC):
    ...


class SearchMode(str, Enum):
    binary = "binary"
    line = "line"
