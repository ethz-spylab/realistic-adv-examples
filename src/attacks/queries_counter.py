import dataclasses
from collections import defaultdict
from enum import Enum
from typing import DefaultDict, TypeVar

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
class QueriesCounter:
    queries_limit: int
    limit_unsafe_queries: bool = False
    _queries: defaultdict[AttackPhase, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    _unsafe_queries: defaultdict[AttackPhase, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

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

    def increase(self, attack_phase: AttackPhase, safe: torch.Tensor) -> "QueriesCounter":
        n_queries = safe.shape[0]
        updated_self = dataclasses.replace(self, _queries=increase_dict(self._queries, attack_phase, n_queries))
        n_unsafe = int((torch.logical_not(safe)).sum().item())
        return dataclasses.replace(updated_self,
                                   _unsafe_queries=increase_dict(self._unsafe_queries, attack_phase, n_unsafe))

    def is_out_of_queries(self) -> bool:
        if self.limit_unsafe_queries:
            return self.total_unsafe_queries >= self.queries_limit
        return self.total_queries >= self.queries_limit
