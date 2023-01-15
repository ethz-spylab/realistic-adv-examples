import dataclasses
from collections import Counter, defaultdict
from enum import Enum
from typing import Callable, Iterable, TypeVar

import numpy as np
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
    equivalent_simulated_queries: int = 1

    def expand_equivalent_queries(self) -> list["CurrentDistanceInfo"]:
        no_equivalent_queries_self = dataclasses.replace(self, equivalent_simulated_queries=1)
        return [no_equivalent_queries_self] * self.equivalent_simulated_queries

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, CurrentDistanceInfo):
            return False
        return (self.phase == __o.phase and self.safe == __o.safe and np.allclose(self.distance, __o.distance)
                and np.allclose(self.best_distance, __o.best_distance)
                and self.equivalent_simulated_queries == __o.equivalent_simulated_queries)


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
    def total_simulated_queries(self) -> int:
        return sum(map(lambda x: x.equivalent_simulated_queries, self._distances))

    @property
    def total_simulated_unsafe_queries(self) -> int:
        return sum(map(lambda x: x.equivalent_simulated_queries, filter(lambda x: not x.safe, self._distances)))

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

    def increase(self,
                 attack_phase: AttackPhase,
                 safe: torch.Tensor,
                 distance: torch.Tensor,
                 equivalent_simulated_queries: int = 1) -> "QueriesCounter":
        n_queries = safe.shape[0]
        updated_self = dataclasses.replace(self, _queries=increase_dict(self._queries, attack_phase, n_queries))
        n_unsafe = int((torch.logical_not(safe)).sum().item())
        new_distances, best_distance = self._make_distances_to_log(attack_phase, safe, distance,
                                                                   equivalent_simulated_queries)
        updated_distances = updated_self._distances + new_distances
        return dataclasses.replace(updated_self,
                                   _unsafe_queries=increase_dict(self._unsafe_queries, attack_phase, n_unsafe),
                                   _distances=updated_distances,
                                   _best_distance=best_distance)

    def expand_simulated_distances(self, decompress_safe: bool = False) -> "QueriesCounter":
        expanded_distances: list[CurrentDistanceInfo] = []
        for distance_info in self.distances:
            expanded_distances += distance_info.expand_equivalent_queries()
        safe_distances, unsafe_distances = partition(lambda x: x.safe, expanded_distances)
        if decompress_safe:
            expanded_queries = dict(Counter(map(lambda x: x.phase, safe_distances)))
        else:
            expanded_queries = self.queries
        expanded_unsafe_queries = dict(Counter(map(lambda x: x.phase, unsafe_distances)))
        return dataclasses.replace(self,
                                   _distances=expanded_distances,
                                   _queries=expanded_queries,
                                   _unsafe_queries=expanded_unsafe_queries)

    def _make_distances_to_log(self,
                               attack_phase: AttackPhase,
                               safe_list: torch.Tensor,
                               distance_list: torch.Tensor,
                               equivalent_simulated_queries: int = 0) -> tuple[list[CurrentDistanceInfo], float]:
        updated_distances = []
        best_distance = self.best_distance
        for safe, distance in zip(safe_list.tolist(), distance_list.tolist()):
            if isinstance(distance, list):
                # TODO: remove this hack and understand where it comes from
                distance = distance[0]
            if safe and distance < best_distance:
                best_distance = distance
            updated_distances.append(
                CurrentDistanceInfo(attack_phase, safe, distance, best_distance, equivalent_simulated_queries))
        return updated_distances, best_distance

    def is_out_of_queries(self) -> bool:
        out_of_unsafe_queries = (self.unsafe_queries_limit is not None
                                 and self.total_unsafe_queries >= self.unsafe_queries_limit)
        out_of_safe_queries = self.queries_limit is not None and self.total_queries >= self.queries_limit
        return out_of_unsafe_queries or out_of_safe_queries


T = TypeVar("T")


def partition(pred: Callable[[T], bool], iterable: Iterable[T]) -> tuple[list[T], list[T]]:
    """Adapted with added types from https://stackoverflow.com/a/4578605"""
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses