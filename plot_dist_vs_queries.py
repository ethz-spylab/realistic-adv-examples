import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.attacks.queries_counter import CurrentDistanceInfo, WrongCurrentDistanceInfo
from src.utils import sha256sum, read_sha256sum, write_sha256sum


def load_wrong_distances(exp_path: Path) -> list[list[WrongCurrentDistanceInfo]]:
    with (exp_path / "distances_traces.json").open("r") as f:
        raw_results = json.load(f)
    return list(map(lambda x: list(map(lambda y: WrongCurrentDistanceInfo(**y), x)), raw_results))


def save_correct_distances(exp_path: Path, distances: list[list[CurrentDistanceInfo]]) -> None:
    distances_dicts = list(map(lambda x: list(map(lambda y: y.__dict__, x)), distances))
    with (exp_path / "distances_traces_fixed.json").open("w") as f:
        json.dump(distances_dicts, f)
    print("Saving checksum of distances_traces.json to distances_traces.json.sha256")
    write_sha256sum(exp_path / "distances_traces.json", exp_path / "distances_traces.json.sha256")


def fix_distances(wrong_distance_infos: list[list[WrongCurrentDistanceInfo]]) -> list[list[CurrentDistanceInfo]]:
    distance_infos: list[list[CurrentDistanceInfo]] = []

    for sample_distances in wrong_distance_infos:
        best_distance = float("inf")
        sample_distance_infos: list[CurrentDistanceInfo] = []
        for wrong_info in sample_distances:
            if wrong_info.safe[0]:
                best_distance = min(best_distance, wrong_info.distance)
            distance_info = CurrentDistanceInfo(wrong_info.phase, wrong_info.safe[0], wrong_info.distance,
                                                best_distance, wrong_info.equivalent_simulated_queries)
            sample_distance_infos.append(distance_info)
        distance_infos.append(sample_distance_infos)

    return distance_infos


def fix_distances_traces(path: Path) -> list[list[CurrentDistanceInfo]]:
    print("Loading wrong distances")
    wrong_distances = load_wrong_distances(path)
    print("Loaded wrong distances, fixing distances")
    fixed_distances = fix_distances(wrong_distances)
    print("Fixed distances, saving correct distances")
    save_correct_distances(path, fixed_distances)
    print("Saved correct distances")
    return fixed_distances


def are_distances_wrong(exp_path: Path) -> bool:
    with (exp_path / "distances_traces.json").open("r") as f:
        t = f.read(100)
    return t.split("\"safe\": ")[1][0] == "["


def convert_distances_to_array(distances: list[list[CurrentDistanceInfo]], unsafe_only: bool) -> np.ndarray:
    if unsafe_only:
        queries_to_plot = list(
            map(lambda sample_distances: list(filter(lambda query: not query.safe, sample_distances)), distances))
    else:
        queries_to_plot = distances
    plot_up_to = min(map(lambda sample_queries: len(sample_queries), queries_to_plot))
    limited_queries_to_plot = list(map(lambda sample_distances: sample_distances[:plot_up_to], queries_to_plot))
    best_distance_up_to_query = list(
        map(lambda sample_distances: list(map(lambda x: x.best_distance, sample_distances)), limited_queries_to_plot))
    return np.asarray(best_distance_up_to_query)


def load_distances_from_json(exp_path: Path, checksum_check: bool) -> list[list[CurrentDistanceInfo]]:
    if not are_distances_wrong(exp_path):
        print("Loading distances from `distances_traces.json`")
        with (exp_path / "distances_traces.json").open() as f:
            raw_results = json.load(f)
        return list(map(lambda x: list(map(lambda y: CurrentDistanceInfo(**y), x)), raw_results))

    print("Distances were originally wrong for the experiment")
    fixed_distances_path = (exp_path / "distances_traces_fixed.json")
    recompute_fixed_distances = not fixed_distances_path.exists()
    if not fixed_distances_path.exists():
        print("The fixed distances file does not exist. Fixing distances first.")
    if (checksum_check and fixed_distances_path.exists() and
            sha256sum(exp_path / "distances_traces.json") != read_sha256sum(exp_path / "distances_traces.json.sha256")):
        print("`distances_traces`.json has been modified since distances_traces_fixed.json was created. "
              "Fixing distances first.")
        recompute_fixed_distances = True

    if recompute_fixed_distances:
        return fix_distances_traces(exp_path)

    print("Loading fixed distances from `distances_traces_fixed.json`")
    with fixed_distances_path.open() as f:
        raw_results = json.load(f)
    return list(map(lambda x: list(map(lambda y: CurrentDistanceInfo(**y), x)), raw_results))


def load_distances_from_array(exp_path: Path, unsafe_only: bool, check_checksum: bool) -> np.ndarray:
    array_path = exp_path / f"distances_array{'_unsafe_only' if unsafe_only else ''}.npy"
    recompute_array = not array_path.exists()
    if recompute_array:
        print("The distances array file does not exist. Reading distances_traces.json and re-creating the array.")
    if check_checksum and array_path.exists() and sha256sum(exp_path / "distances_traces.json") != read_sha256sum(
            exp_path / "distances_traces-to_numpy.json.sha256"):
        print("The distances array is outdated. Re-reading distances_traces.json and re-creating the array.")
        recompute_array = True
    if recompute_array:
        distances = convert_distances_to_array(load_distances_from_json(exp_path, check_checksum), unsafe_only)
        save_distances_array(exp_path, distances, unsafe_only, check_checksum)
        return distances
    return np.load(array_path)


def save_distances_array(exp_path: Path, distances_array: np.ndarray, unsafe_only: bool, save_checksum: bool):
    np.save(exp_path / f"distances_array{'_unsafe_only' if unsafe_only else ''}.npy", distances_array)
    if save_checksum:
        print("Saving checksum of distances_traces.json to distances_traces-to_numpy.json.sha256")
        checksum_filename = f"distances_traces-to_numpy-{'unsafe_only' if unsafe_only else ''}.json.sha256"
        checksum_file_destination = exp_path / checksum_filename
        write_sha256sum(exp_path / "distances_traces.json", checksum_file_destination)


def plot_median_distances_per_query(exp_paths: list[Path], names: list[str] | None, max_queries: int | None,
                                    unsafe_only: bool, out_path: Path, checksum_check: bool):
    names = names or ["" for _ in exp_paths]
    for exp_path, name in zip(exp_paths, names):
        distances = load_distances_from_array(exp_path, unsafe_only, checksum_check)
        n_to_plot = max_queries or distances.shape[1]
        median_distances = np.median(distances[:, :n_to_plot], axis=0)
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
    parser.add_argument("--checksum-check", action="store_true", default=False)
    args = parser.parse_args()
    plot_median_distances_per_query(args.exp_paths, args.names, args.max_queries, args.unsafe_only, args.out_path,
                                    args.checksum_check)
