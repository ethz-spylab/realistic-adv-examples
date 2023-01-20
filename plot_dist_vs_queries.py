import argparse
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator

import ijson
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from ijson.common import IncompleteJSONError

from src.attacks.queries_counter import CurrentDistanceInfo, WrongCurrentDistanceInfo
from src.utils import sha256sum, read_sha256sum, write_sha256sum
from src.json_list import JSONList

OPENED_FILES: list[TextIOWrapper] = []
MAX_SAMPLES = 1000


def wrap_ijson_iterator(iterator: Iterator[Any]) -> Iterator[Any]:
    for item in iterator:    
        try:
            yield item
        except IncompleteJSONError:
            yield {
                "phase": "direction_search",
                "distance": float("inf"),
                "safe": False,
                "best_distance": float("inf"),
            }


def load_wrong_distances(exp_path: Path) -> Iterator[list[WrongCurrentDistanceInfo]]:
    path = exp_path / "distances_traces.json"
    f = path.open("r")
    OPENED_FILES.append(f)
    raw_results = wrap_ijson_iterator(ijson.items(f, "item", use_float=True))
    return map(lambda x: list(map(lambda y: WrongCurrentDistanceInfo(**y), x)), raw_results)


def save_correct_distances(exp_path: Path, distances: Iterator[list[CurrentDistanceInfo]]) -> None:
    distances_dicts = map(lambda x: [y.__dict__ for y in x], distances)
    json_list = JSONList(exp_path / "distances_traces_fixed.json")
    for distances_dict in tqdm.tqdm(distances_dicts, total=MAX_SAMPLES):
        json_list.append(distances_dict)
    print("Saving checksum of distances_traces.json to distances_traces.json.sha256")
    write_sha256sum(exp_path / "distances_traces.json", exp_path / "distances_traces.json.sha256")


def fix_distances(
        wrong_distance_infos: Iterator[list[WrongCurrentDistanceInfo]]) -> Iterator[list[CurrentDistanceInfo]]:

    for sample_distances in wrong_distance_infos:
        best_distance = float("inf")
        sample_distance_infos: list[CurrentDistanceInfo] = []
        for wrong_info in sample_distances:
            if wrong_info.safe[0]:
                best_distance = min(best_distance, wrong_info.distance)
            distance_info = CurrentDistanceInfo(wrong_info.phase, wrong_info.safe[0], wrong_info.distance,
                                                best_distance, wrong_info.equivalent_simulated_queries)
            sample_distance_infos.append(distance_info)
        yield sample_distance_infos


def fix_distances_traces(path: Path) -> None:
    print("Loading wrong distances")
    wrong_distances = load_wrong_distances(path)
    print("Loaded wrong distances, fixing distances")
    fixed_distances = fix_distances(wrong_distances)
    print("Fixed distances, saving correct distances")
    save_correct_distances(path, fixed_distances)
    print("Saved correct distances")


def are_distances_wrong(exp_path: Path) -> bool:
    with (exp_path / "distances_traces.json").open("r") as f:
        t = f.read(100)
    return t.split("\"safe\": ")[1][0] == "["


def pad_to_len(list_: list[float], n: int) -> np.ndarray:
    to_pad = n - len(list_)
    if to_pad > 0:
        return np.pad(np.asarray(list_), (0, to_pad), "edge")
    return np.asarray(list_[:n])


MAX_UNSAFE_QUERIES = 5000
MAX_QUERIES = 20_000


def convert_distances_to_array(distances: Iterator[list[CurrentDistanceInfo]], unsafe_only: bool) -> np.ndarray:
    if unsafe_only:
        queries_to_plot = map(lambda sample_distances: list(filter(lambda query: not query.safe, sample_distances)),
                              distances)
    else:
        queries_to_plot = distances

    if not unsafe_only:
        plot_up_to = MAX_QUERIES
    else:
        plot_up_to = MAX_UNSAFE_QUERIES

    best_distance_up_to_query = map(lambda sample_distances: [x.best_distance for x in sample_distances],
                                    queries_to_plot)

    print("Converting distances to array")
    limited_queries_to_plot = np.fromiter(tqdm.tqdm((pad_to_len(l_, plot_up_to) for l_ in best_distance_up_to_query),
                                                    total=MAX_SAMPLES),
                                          dtype=np.dtype((float, plot_up_to)))
    return limited_queries_to_plot


def load_distances_from_json(exp_path: Path, checksum_check: bool) -> Iterator[list[CurrentDistanceInfo]]:
    if not are_distances_wrong(exp_path):
        print("Loading distances from `distances_traces.json`")
        path = exp_path / "distances_traces.json"
        f = path.open("r")
        OPENED_FILES.append(f)
        raw_results = wrap_ijson_iterator(ijson.items(f, "item", use_float=True))
        return map(lambda x: list(map(lambda y: CurrentDistanceInfo(**y), x)), raw_results)

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
        fix_distances_traces(exp_path)
        return load_distances_from_json(exp_path, checksum_check=True)

    print("Loading fixed distances from `distances_traces_fixed.json`")
    f = fixed_distances_path.open("r")
    OPENED_FILES.append(f)
    raw_results = wrap_ijson_iterator(ijson.items(f, "item", use_float=True))
    return map(lambda x: list(map(lambda y: CurrentDistanceInfo(**y), x)), raw_results)


def load_distances_from_array(exp_path: Path, unsafe_only: bool, check_checksum: bool) -> np.ndarray:
    array_path = exp_path / f"distances_array{'_unsafe_only' if unsafe_only else ''}.npy"
    recompute_array = not array_path.exists()
    if recompute_array:
        print("The distances array file does not exist. Reading distances_traces.json and re-creating the array.")
    checksum_filename = f"distances_traces-to_numpy{'-unsafe_only' if unsafe_only else ''}.json.sha256"
    if check_checksum and array_path.exists() and sha256sum(exp_path / "distances_traces.json") != read_sha256sum(
            exp_path / checksum_filename):
        print("The distances array is outdated. Re-reading distances_traces.json and re-creating the array.")
        recompute_array = True
    if recompute_array:
        print("Converting the distances to arrays")
        distances = convert_distances_to_array(load_distances_from_json(exp_path, check_checksum), unsafe_only)
        save_distances_array(exp_path, distances, unsafe_only, check_checksum)
        return distances
    return np.load(array_path)


def save_distances_array(exp_path: Path, distances_array: np.ndarray, unsafe_only: bool, save_checksum: bool):
    np.save(exp_path / f"distances_array{'_unsafe_only' if unsafe_only else ''}.npy", distances_array)
    if save_checksum:
        checksum_filename = f"distances_traces-to_numpy{'-unsafe_only' if unsafe_only else ''}.json.sha256"
        print(f"Saving checksum of distances_traces.json to {checksum_filename}")
        checksum_file_destination = exp_path / checksum_filename
        write_sha256sum(exp_path / "distances_traces.json", checksum_file_destination)


def plot_median_distances_per_query(exp_paths: list[Path], names: list[str] | None, max_queries: int | None,
                                    unsafe_only: bool, out_path: Path, checksum_check: bool):
    names = names or ["" for _ in exp_paths]
    distances_arrays = [load_distances_from_array(exp_path, unsafe_only, checksum_check) for exp_path in exp_paths]
    n_samples_to_plot = min(len(distances_array) for distances_array in distances_arrays)
    for distances, name in zip(distances_arrays, names):
        n_to_plot = max_queries or distances.shape[1]
        median_distances = np.median(distances[:n_samples_to_plot, :n_to_plot], axis=0)
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
    for f in OPENED_FILES:
        f.close()
