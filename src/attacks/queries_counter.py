import dataclasses
from collections import defaultdict
from enum import Enum
from typing import TypeVar

import torch

K = TypeVar("K")
V = TypeVar("V")


class AttackPhase(str, Enum):
    ...


class QueryType(str, Enum):
    safe = "safe"
    unsafe = "unsafe"


def increase_dict(d: dict[K, int], k: K, n: int) -> dict[K, int]:
    return update_dict(d, k, d[k] + n)


def update_dict(d: dict[K, V], k: K, v: V) -> dict[K, V]:
    return d | {k: v}


@dataclasses.dataclass
class CurrentDistanceInfo:
    phase: AttackPhase
    safe: bool
    distance: float
    best_distance: float


@dataclasses.dataclass
class QueriesCounter:
    queries_limit: int | None
    unsafe_queries_limit: int | None = None
    _queries: dict[AttackPhase, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    _unsafe_queries: dict[AttackPhase, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    _distances: list[CurrentDistanceInfo] = dataclasses.field(default_factory=list)
    _best_distance: float = float("inf")

    @property
    def total_queries(self) -> int:
        return sum(self._queries.values())

    @property
    def queries(self) -> dict[AttackPhase, int]:
        return self._queries

    @property
    def total_unsafe_queries(self) -> int:
        return sum(self._unsafe_queries.values())

    @property
    def unsafe_queries(self) -> dict[AttackPhase, int]:
        return self._unsafe_queries

    @property
    def distances(self) -> list[CurrentDistanceInfo]:
        return self._distances

    @property
    def best_distance(self) -> float:
        return self._best_distance

    def increase(self, attack_phase: AttackPhase, safe: torch.Tensor, distance: torch.Tensor) -> "QueriesCounter":
        n_queries = safe.shape[0]
        updated_self = dataclasses.replace(self, _queries=increase_dict(self._queries, attack_phase, n_queries))
        n_unsafe = int((torch.logical_not(safe)).sum().item())
        new_distances, best_distance = self._make_distances_to_log(attack_phase, safe, distance)
        updated_distances = updated_self._distances + new_distances
        return dataclasses.replace(updated_self,
                                   _unsafe_queries=increase_dict(self._unsafe_queries, attack_phase, n_unsafe),
                                   _distances=updated_distances,
                                   _best_distance=best_distance)

    def _make_distances_to_log(self, attack_phase: AttackPhase, safe_list: torch.Tensor,
                               distance_list: torch.Tensor) -> tuple[list[CurrentDistanceInfo], float]:
        updated_distances = []
        best_distance = self.best_distance
        for safe, distance in zip(safe_list.tolist(), distance_list.tolist()):
            if safe and distance < best_distance:
                best_distance = distance
            updated_distances.append(CurrentDistanceInfo(attack_phase, safe, distance, best_distance))
        return updated_distances, best_distance

    def is_out_of_queries(self) -> bool:
        out_of_unsafe_queries = (self.unsafe_queries_limit is not None
                                 and self.total_unsafe_queries >= self.unsafe_queries_limit)
        out_of_safe_queries = self.queries_limit is not None and self.total_queries >= self.queries_limit
        return out_of_unsafe_queries or out_of_safe_queries
