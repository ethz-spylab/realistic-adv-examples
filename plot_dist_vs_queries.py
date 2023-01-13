import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.attacks.queries_counter import CurrentDistanceInfo


def get_distances(distances: list[list[CurrentDistanceInfo]], max_queries: int | None, unsafe_only: bool) -> np.ndarray:
    if unsafe_only:
        queries_to_plot = list(
            map(lambda sample_distances: list(filter(lambda query: not query.safe, sample_distances)), distances))
    else:
        queries_to_plot = distances
    plot_up_to = min(map(lambda sample_queries: len(sample_queries), queries_to_plot))
    if max_queries is not None:
        plot_up_to = min(max_queries, plot_up_to)
    limited_queries_to_plot = list(map(lambda sample_distances: sample_distances[:plot_up_to], queries_to_plot))
    best_distance_up_to_query = list(
        map(lambda sample_distances: list(map(lambda x: x.best_distance, sample_distances)), limited_queries_to_plot))
    return np.asarray(best_distance_up_to_query)


def load_distances(exp_path: Path) -> list[list[CurrentDistanceInfo]]:
    with (exp_path / "distances_traces.json").open() as f:
        raw_results = json.load(f)
    return list(map(lambda x: list(map(lambda y: CurrentDistanceInfo(**y), x)), raw_results))


def plot_median_distances_per_query(exp_paths: list[Path], names: list[str] | None, max_queries: int | None,
                                    unsafe_only: bool, out_path: Path):
    names = names or ["" for _ in exp_paths]
    for exp_path, name in zip(exp_paths, names):
        distances = load_distances(exp_path)
        median_distances = np.median(get_distances(distances, max_queries, unsafe_only), axis=0)
        plt.plot(median_distances, label=name)
    plt.title(f"Median distances per query {'(unsafe only)' if unsafe_only else ''}")
    plt.xlabel("Query number")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=False, default=None)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--unsafe-only", action="store_true", default=False)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    plot_median_distances_per_query(args.exp_paths, args.names, args.max_queries, args.unsafe_only, args.out_path)