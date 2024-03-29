import abc
from enum import Enum
from typing import NamedTuple

import torch
from foolbox.distances import LpDistance

from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.model_wrappers import ModelWrapper


class Bounds(NamedTuple):
    lower: float = 0.
    upper: float = 1.


ExtraResultsDictContent = float | int | list[float] | list[int]
ExtraResultsDict = dict[str, ExtraResultsDictContent]


class BaseAttack(abc.ABC):

    def __init__(self, epsilon: float | None, distance: LpDistance, bounds: Bounds, discrete: bool,
                 queries_limit: int | None, unsafe_queries_limit: int | None):
        self.epsilon = epsilon
        self.discrete = discrete
        self.bounds = bounds
        self.distance = distance
        self.queries_limit = queries_limit
        self.unsafe_queries_limit = unsafe_queries_limit

    def _make_queries_counter(self) -> QueriesCounter:
        return QueriesCounter(self.queries_limit, self.unsafe_queries_limit)

    def is_correct_boundary_side(self, model: ModelWrapper, x_adv: torch.Tensor, y: torch.Tensor,
                                 target: torch.Tensor | None, queries_counter: QueriesCounter,
                                 attack_phase: AttackPhase,
                                 original_x: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter]:
        if len(x_adv.size()) != 4:
            x_adv = x_adv.unsqueeze(0)
        if len(original_x.size()) != 4:
            original_x = original_x.unsqueeze(0)
        if target is not None:
            success = model.predict_label(x_adv) == target
        else:
            success = model.predict_label(x_adv) != y
        distance = self.distance(original_x, x_adv)
        return success, queries_counter.increase(attack_phase,
                                                 safe=success,
                                                 distance=distance,
                                                 equivalent_simulated_queries=1)

    def is_correct_boundary_side_batched(self,
                                         model: ModelWrapper,
                                         x_adv: torch.Tensor,
                                         y: torch.Tensor,
                                         target: torch.Tensor | None,
                                         queries_counter: QueriesCounter,
                                         attack_phase: AttackPhase,
                                         original_x: torch.Tensor,
                                         equivalent_simulated_queries: int = 0,
                                         count_simulated_if_unsafe: bool = False,
                                         first_batch: bool = False) -> tuple[torch.Tensor, QueriesCounter]:
        # Get success vector but ignore queries counter because we will update it later
        success, _ = self.is_correct_boundary_side(model, x_adv, y, target, queries_counter, attack_phase, original_x)
        # If we hit the boundary, then we we consider the output up to the first unsafe query inclusive
        # Otherwise we consider the whole output
        boundary_hit = not success.all()
        if not boundary_hit:
            distance = self.distance(original_x, x_adv)
            updated_queries_counter = queries_counter.increase(
                attack_phase,
                safe=success,
                distance=distance,
                equivalent_simulated_queries=equivalent_simulated_queries)
            return success, updated_queries_counter

        if count_simulated_if_unsafe or (first_batch and not success[0].item()):
            equivalent_unsafe_simulated_queries = 1
        else:
            equivalent_unsafe_simulated_queries = 0

        first_unsafe_query_idx = torch.argmin(success.to(torch.int))
        if first_unsafe_query_idx > 0:
            # Let's get only the safe queries for now
            safe_success = success[:first_unsafe_query_idx]
            distance = self.distance(original_x, x_adv[:first_unsafe_query_idx])
            updated_queries_counter = queries_counter.increase(
                attack_phase,
                safe=safe_success,
                distance=distance,
                equivalent_simulated_queries=equivalent_simulated_queries)
        else:
            updated_queries_counter = queries_counter
            safe_success = torch.tensor([], device=success.device, dtype=success.dtype)

        # Add the unsafe query
        unsafe_distance = self.distance(original_x, x_adv[first_unsafe_query_idx][None]).unsqueeze(0)
        updated_queries_counter_with_unsafe = updated_queries_counter.increase(
            attack_phase,
            safe=success[first_unsafe_query_idx].unsqueeze(0),
            distance=unsafe_distance,
            equivalent_simulated_queries=equivalent_unsafe_simulated_queries)
        relevant_success = torch.cat((safe_success, success[first_unsafe_query_idx].unsqueeze(0)))

        return relevant_success, updated_queries_counter_with_unsafe

    def clamp_and_discretize(self, out: torch.Tensor) -> torch.Tensor:
        out = torch.clamp(out, self.bounds.lower, self.bounds.upper)
        #if self.discrete:
        #    assert torch.allclose(out, torch.round(out * 255) / 255)  # type: ignore
        #    decoded_out = encode_decode(out)
        #    assert torch.allclose(out, decoded_out, atol=1 / 256)
        #    out = decoded_out
        return out

    @abc.abstractmethod
    def __call__(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            label: torch.Tensor,
            target: torch.Tensor | None = None) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        ...


class DirectionAttackPhase(AttackPhase):
    search = "search"
    initialization = "initialization"
    direction_probing = "direction_probing"


class DirectionAttack(BaseAttack, abc.ABC):
    """
    Base class for attacks which optimize a direction instead of the perturbation directly
    """

    def get_x_adv(self, x: torch.Tensor, v: torch.Tensor, d: float | torch.Tensor) -> torch.Tensor:
        #if self.discrete and not np.isinf(d):
        #    integer_d = d * 255
        #    if isinstance(integer_d, torch.Tensor):
        #        assert torch.allclose(integer_d, torch.round(integer_d))
        #        d = torch.round(integer_d) / 255
        #    else:
        #        assert np.allclose(round(integer_d), integer_d)
        #        d = round(integer_d) / 255
        out: torch.Tensor = x + d * v  # type: ignore
        out = self.clamp_and_discretize(out)
        return out


class PerturbationAttack(BaseAttack, abc.ABC):

    def get_x_adv(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        #if self.discrete:
        #    assert torch.round(delta) == delta
        #    delta = delta / 255
        out: torch.Tensor = x + delta  # type: ignore
        out = self.clamp_and_discretize(out)
        return out


class SearchMode(str, Enum):
    binary = "binary"
    line = "line"
    eggs_dropping = "eggs_dropping"
