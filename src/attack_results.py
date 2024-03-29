import dataclasses
import json
from pathlib import Path

import numpy as np

from src.attacks.base import ExtraResultsDict, ExtraResultsDictContent
from src.attacks.queries_counter import QueriesCounter
from src.json_list import JSONList


@dataclasses.dataclass
class AttackResults:
    path: Path
    successes: int = 0
    distances: list[float] = dataclasses.field(default_factory=list)
    queries_counters: list[QueriesCounter] = dataclasses.field(default_factory=list)
    extra_results: list[ExtraResultsDict] = dataclasses.field(default_factory=list)
    failures: int = 0
    failed_distances: list[float] = dataclasses.field(default_factory=list)
    failed_queries_counters: list[QueriesCounter] = dataclasses.field(default_factory=list)
    failed_extra_results: list[ExtraResultsDict] = dataclasses.field(default_factory=list)
    _distances_traces_jsonlist: JSONList = dataclasses.field(init=False)
    _failed_distances_traces_jsonlist: JSONList = dataclasses.field(init=False)

    def __post_init__(self):
        self._distances_traces_jsonlist = JSONList(self.path / "distances_traces.json")
        self._failed_distances_traces_jsonlist = JSONList(self.path / "failed_distances_traces.json")

    def update_with_success(self, distance: float, queries_counter: QueriesCounter,
                            extra_results: ExtraResultsDict) -> "AttackResults":
        return dataclasses.replace(self,
                                   successes=self.successes + 1,
                                   distances=self.distances + [distance],
                                   queries_counters=self.queries_counters + [queries_counter],
                                   extra_results=self.extra_results + [extra_results])

    def update_with_failure(self, distance: float, queries_counter: QueriesCounter,
                            extra_results: ExtraResultsDict) -> "AttackResults":
        return dataclasses.replace(self,
                                   failures=self.failures + 1,
                                   failed_distances=self.failed_distances + [distance],
                                   failed_queries_counters=self.failed_queries_counters + [queries_counter],
                                   failed_extra_results=self.failed_extra_results + [extra_results])

    def log_results(self, idx: int, simulated: bool = False):
        print(f"index: {idx:4d} avg dist: {np.mean(np.array(self.distances)):.4f} "
              f"median dist: {np.median(np.array(self.distances)):.4f} "
              f"avg queries: {np.mean(np.array(self._get_overall_queries(simulated))):.4f} "
              f"median queries: {np.median(np.array(self._get_overall_queries(simulated))):.4f} "
              f"avg bad queries: {np.mean(np.array(self._get_overall_unsafe_queries(simulated))):.4f} "
              f"median bad queries: {np.median(np.array(self._get_overall_unsafe_queries(simulated))):.4f} "
              f"asr: {np.mean(np.array(self.asr)):.4f} \n")

    @property
    def simulated_self(self) -> "AttackResults":
        if not self.has_simulated_counters:
            raise ValueError("No simulated counters to expand")
        simulated_queries_counters = map(lambda c: c.expand_simulated_distances(), self.queries_counters)
        simulated_failed_queries_counters = map(lambda c: c.expand_simulated_distances(), self.failed_queries_counters)
        return dataclasses.replace(self,
                                   queries_counters=list(simulated_queries_counters),
                                   failed_queries_counters=list(simulated_failed_queries_counters))

    @property
    def has_simulated_counters(self) -> bool:
        if self.queries_counters:
            return (len(list(filter(lambda d: d.equivalent_simulated_queries > 0, self.queries_counters[0].distances)))
                    > 1)
        if self.failed_queries_counters:
            return (len(
                list(filter(lambda d: d.equivalent_simulated_queries > 0, self.failed_queries_counters[0].distances))) >
                    1)
        return False

    def get_aggregated_results_dict(self) -> dict[str, float]:
        results_dict = {
            "asr": self.asr,
            "distortion": np.mean(np.array(self.distances)),
            "median_distortion": np.median(np.array(self.distances)),
            "mean_queries": np.mean(np.array(self._get_overall_queries())),
            "median_queries": np.median(np.array(self._get_overall_queries())),
            "mean_unsafe_queries": np.mean(np.array(self._get_overall_unsafe_queries())),
            "median_unsafe_queries": np.median(np.array(self._get_overall_unsafe_queries())),
        }
        aggregated_queries, aggregated_unsafe_queries = aggregate_queries_counters_list(self.queries_counters)

        for stat, stat_fn in (("mean", np.mean), ("median", np.median)):
            for phase, queries_list in aggregated_queries.items():
                results_dict[f"{stat}_queries_{phase}"] = stat_fn(np.array(queries_list))
            for phase, queries_list in aggregated_unsafe_queries.items():
                results_dict[f"{stat}_unsafe_queries_{phase}"] = stat_fn(np.array(queries_list))
            for key, value_list in aggregate_extra_results(self.extra_results).items():
                if len(value_list) != 0 and isinstance(value_list[0], list):
                    continue
                results_dict[f"{stat}_{key}"] = stat_fn(np.array(value_list))

        return results_dict

    def save_results(self, verbose: bool = True):
        if not self.path.exists():
            self.path.mkdir()
        with open(self.path / "aggregated_results.json", 'w') as f:
            json.dump(self.get_aggregated_results_dict(), f, indent=4)
        with open(self.path / "full_results.json", 'w') as f:
            json.dump(self.get_full_results_dict(), f, indent=4)
        np.save(self.path / "distances.npy", np.array(self.distances))
        np.save(self.path / "queries.npy", np.array(self._get_overall_queries()))
        np.save(self.path / "unsafe_queries.npy", np.array(self._get_overall_unsafe_queries()))
        np.save(self.path / "failed_distances.npy", np.array(self.failed_distances))
        np.save(self.path / "failed_queries.npy", np.array(self._get_overall_failed_queries()))
        if self.queries_counters:
            self._distances_traces_jsonlist.append(
                list(map(lambda distance_info: distance_info.__dict__, self.queries_counters[-1].distances)))
        if self.failed_queries_counters:
            self._failed_distances_traces_jsonlist.append(
                list(map(lambda distance_info: distance_info.__dict__, self.failed_queries_counters[-1].distances)))
        if verbose:
            print(f"Saved results to {self.path}")

    def get_full_results_dict(self) -> dict[str, float | list[float]]:
        d = {
            "successes": self.successes,
            "distances": self.distances,
            "failures": self.failures,
            "failed_distances": self.failed_distances,
        }

        aggregated_queries, aggregated_unsafe_queries = aggregate_queries_counters_list(self.queries_counters)
        for phase, queries_list in aggregated_queries.items():
            d[f"queries_{phase}"] = queries_list
        for phase, queries_list in aggregated_unsafe_queries.items():
            d[f"unsafe_queries_{phase}"] = queries_list
        for key, value_list in aggregate_extra_results(self.extra_results).items():
            d[f"{key}"] = value_list

        return d

    @property
    def asr(self) -> float:
        return self.successes / (self.successes + self.failures)

    def _get_overall_queries(self, simulated: bool = False) -> list[int]:
        if simulated:
            return list(map(lambda counter: counter.total_simulated_queries, self.queries_counters))
        return list(map(lambda counter: counter.total_queries, self.queries_counters))

    def _get_overall_failed_queries(self, simulated: bool = False) -> list[int]:
        if simulated:
            return list(map(lambda counter: counter.total_simulated_queries, self.failed_queries_counters))
        return list(map(lambda counter: counter.total_queries, self.failed_queries_counters))

    def _get_overall_unsafe_queries(self, simulated: bool = False) -> list[int]:
        if simulated:
            return list(map(lambda counter: counter.total_simulated_unsafe_queries, self.queries_counters))
        return list(map(lambda counter: counter.total_unsafe_queries, self.queries_counters))

    def _get_overall_failed_unsafe_queries(self, simulated: bool = False) -> list[int]:
        if simulated:
            return list(map(lambda counter: counter.total_simulated_unsafe_queries, self.failed_queries_counters))
        return list(map(lambda counter: counter.total_unsafe_queries, self.failed_queries_counters))


def aggregate_extra_results(extra_results_list: list[ExtraResultsDict]) -> dict[str, list[ExtraResultsDictContent]]:
    aggregated_extra_results = {}
    for extra_results in extra_results_list:
        for key, value in extra_results.items():
            aggregated_extra_results[key] = aggregated_extra_results.get(key, []) + [value]
    return aggregated_extra_results


def aggregate_queries_counters_list(q_list: list[QueriesCounter]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    aggregated_queries: dict[str, list[int]] = {}
    for safe_queries in [q.queries for q in q_list]:
        for phase, n_queries in safe_queries.items():
            aggregated_queries[phase] = aggregated_queries.get(phase, []) + [n_queries]

    aggregated_unsafe_queries: dict[str, list[int]] = {}
    for unsafe_queries in [q.unsafe_queries for q in q_list]:
        for phase, n_queries in unsafe_queries.items():
            aggregated_unsafe_queries[phase] = aggregated_unsafe_queries.get(phase, []) + [n_queries]

    return aggregated_queries, aggregated_unsafe_queries
