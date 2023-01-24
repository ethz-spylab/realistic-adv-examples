import argparse
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator
import warnings

import ijson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from ijson.common import IncompleteJSONError

from src.attacks.queries_counter import CurrentDistanceInfo, WrongCurrentDistanceInfo
from src.json_list import JSONList
from src.utils import read_sha256sum, sha256sum, write_sha256sum

OPENED_FILES: list[TextIOWrapper] = []
MAX_SAMPLES = 1000


def generate_simulated_distances(items: Iterator[list[dict[str, Any]]],
                                 unsafe_only: bool) -> Iterator[list[CurrentDistanceInfo]]:
    for distances_list in items:
        simulated_distances = []
        for distance in distances_list:
            if unsafe_only and distance["safe"]:
                continue
            simulated_distance = CurrentDistanceInfo(**(distance | {"equivalent_simulated_queries": 1}))  # type: ignore
            simulated_distances += [simulated_distance] * distance["equivalent_simulated_queries"]
            if unsafe_only and len(simulated_distances) >= MAX_UNSAFE_QUERIES:
                break
            elif len(simulated_distances) >= MAX_QUERIES:
                break
        yield simulated_distances


SIMULATED_DISTANCES_FILENAME = "simulated_distances_array{}.npy"


def get_simulated_array(exp_path: Path, unsafe_only: bool) -> np.ndarray:
    array_filename = SIMULATED_DISTANCES_FILENAME.format("_unsafe_only" if unsafe_only else "")
    if (exp_path / array_filename).exists():
        print("Loading simulated distances from file")
        return np.load(exp_path / array_filename)
    original_distances_filename = are_distances_wrong(
        exp_path) and "distances_traces_fixed.json" or "distances_traces.json"
    f = (exp_path / original_distances_filename).open("r")
    OPENED_FILES.append(f)
    raw_results = wrap_ijson_iterator(ijson.items(f, "item", use_float=True))
    simulated_distances = generate_simulated_distances(raw_results, unsafe_only)
    array = convert_distances_to_array(simulated_distances, unsafe_only)
    save_distances_array(exp_path, array, True, False, array_filename)
    return array


def wrap_ijson_iterator(iterator: Iterator[list[dict[str, Any]]]) -> Iterator[list[dict[str, Any]]]:
    try:
        for item in iterator:
            yield item
    except IncompleteJSONError as e:
        raise e


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


MAX_UNSAFE_QUERIES = 15_000
MAX_QUERIES = 50_000


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
        print(f"Loading distances from {exp_path / 'distances_traces.json'}")
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


def save_distances_array(exp_path: Path,
                         distances_array: np.ndarray,
                         unsafe_only: bool,
                         save_checksum: bool,
                         filename: str | None = None):
    filename = filename or f"distances_array{'_unsafe_only' if unsafe_only else ''}.npy"
    np.save(exp_path / filename, distances_array)
    if save_checksum:
        checksum_filename = f"distances_traces-to_numpy{'-unsafe_only' if unsafe_only else ''}.json.sha256"
        print(f"Saving checksum of distances_traces.json to {checksum_filename}")
        checksum_file_destination = exp_path / checksum_filename
        write_sha256sum(exp_path / "distances_traces.json", checksum_file_destination)


COLORS_STYLES_MARKERS = {
    "OPT": ("tab:blue", "--", "s"),
    "Stealthy OPT": ("tab:blue", "-", "s"),
    "SignOPT": ("tab:orange", "--", "x"),
    "Stealthy SignOPT": ("tab:orange", "-", "x"),
    "Boundary": ("tab:red", "--", "^"),
    "HSJA": ("tab:green", "--", "o"),
    "RayS": ("tab:blue", "--", "s"),
    "Stealthy RayS": ("tab:blue", "-", "s"),
    "k = 1.5": ("tab:blue", "-", "s"),
    "k = 2": ("tab:orange", "-", "x"),
    "k = 2.5": ("tab:green", "-", "o"),
    "k = 3": ("tab:red", "-", "^"),
}


def plot_median_distances_per_query(exp_paths: list[Path], names: list[str] | None, max_queries: int | None,
                                    max_samples: int | None, unsafe_only: bool, out_path: Path, checksum_check: bool,
                                    to_simulate: list[int] | None):
    names = names or ["" for _ in exp_paths]
    distances_arrays = []

    if "/linf/" in str(exp_paths[0]):
        epsilons = [4 / 255, 8 / 255, 16 / 255, 32 / 255, 64 / 255, 128 / 255]
    else:
        epsilons = [0.5, 1, 2, 5, 10, 20, 50, 100, 150]

    for i, exp_path in enumerate(exp_paths):
        if to_simulate is not None and i in to_simulate:
            distances_array = get_simulated_array(exp_paths[i], unsafe_only)
        else:
            distances_array = load_distances_from_array(exp_path, unsafe_only, checksum_check)
        distances_arrays.append(distances_array)

    n_samples_to_plot = min(len(distances_array) for distances_array in distances_arrays)
    n_samples_to_plot = min(n_samples_to_plot, max_samples or n_samples_to_plot)

    if max_samples is not None and n_samples_to_plot < max_samples:
        warnings.warn(f"Could not plot {max_samples} samples, only {n_samples_to_plot} were available.")

    RATIO = 3 / 4
    WIDTH = 6
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * RATIO))

    queries_per_epsilon_df = pd.DataFrame(columns=["attack", "epsilon", "n_queries"])

    attacks_distances_dict = {}
    for distances, name in zip(distances_arrays, names):
        attacks_distances_dict[name] = distances
        if name and name in COLORS_STYLES_MARKERS:
            color, style, marker = COLORS_STYLES_MARKERS[name]
        elif not name:
            warnings.warn("Attack name not specified. Using default color, style and marker.")
            color, style, marker = None, None, None
        else:
            warnings.warn(f"Could not find color, style, marker for {name}. Using default.")
            color, style, marker = None, None, None
        n_to_plot = max_queries or distances.shape[1]
        median_distances = np.median(distances[:n_samples_to_plot, :n_to_plot], axis=0)
        BASE_LINEWIDTH = 1.5
        full_median_distances = np.median(distances[:n_samples_to_plot], axis=0)
        for epsilon in epsilons:
            if ((full_median_distances) < epsilon).any():
                queries_per_epsilon_df = pd.concat([
                    queries_per_epsilon_df,
                    pd.DataFrame({
                        "attack": [name],
                        "epsilon": [epsilon],
                        "n_queries": [np.argmax(full_median_distances < epsilon)]
                    })
                ])
                n_queries_for_epsilon = np.argmax(full_median_distances < epsilon)
                print(f"Attack: {name}, epsilon = {epsilon}, n_queries = {n_queries_for_epsilon}")
            else:
                print(f"Attack: {name} didn't reach epsilon = {epsilon}")
                queries_per_epsilon_df = pd.concat([
                    queries_per_epsilon_df,
                    pd.DataFrame({
                        "attack": [name],
                        "epsilon": [epsilon],
                        "n_queries": [np.inf]
                    })
                ])

        if "Stealthy" in name:
            linewidth = 1.5 * BASE_LINEWIDTH
        else:
            linewidth = 1 * BASE_LINEWIDTH
        ax.plot(median_distances,
                label=name,
                color=color,
                linestyle=style,
                marker=marker,
                markevery=n_to_plot // 10,
                linewidth=linewidth)

    queries_per_epsilon_df.to_csv(out_path.parent / f"queries_per_epsilon_{out_path.stem}.csv", index=False)

    for attack, distances in attacks_distances_dict.items():
        if "Stealthy" not in attack:
            continue
        median_distances = np.median(distances[:n_samples_to_plot], axis=0)
        original_attack_name = attack.split("Stealthy ")[1]
        original_attack_median_distances = np.median(attacks_distances_dict[original_attack_name][:n_samples_to_plot],
                                                     axis=0)
        maximum_improvement = np.max(original_attack_median_distances / median_distances)
        minimum_improvement = np.min(original_attack_median_distances / median_distances)
        
        maximum_improvement_query = np.argmax(original_attack_median_distances / median_distances)
        maximum_improvement_distance_original = original_attack_median_distances[maximum_improvement_query]
        maximum_improvement_distance = median_distances[maximum_improvement_query]
       
        minimum_improvement_query = np.argmin(original_attack_median_distances / median_distances)
        minimum_improvement_distance = median_distances[minimum_improvement_query]
        minimum_improvement_distance_original = original_attack_median_distances[minimum_improvement_query]
        print(
            f"Attack: {attack}, maximum improvement = {maximum_improvement:.3f} at query {maximum_improvement_query} "
            f"and distance {maximum_improvement_distance:.3f} (original attack distance {maximum_improvement_distance_original:.3f})\n"  # noqa
            f"minimum improvement = {minimum_improvement:.3f} at query {minimum_improvement_query} and "
            f"distance {minimum_improvement_distance:.3f} (original attack distance {minimum_improvement_distance_original:.3f})"  # noqa
        )

    if "/l2/" in str(exp_paths[0]) and "k" not in names[0]:
        ax.set_ylim(5e-0, 1e2)
    elif "/linf/" in str(exp_paths[0]):
        ax.set_ylim(2e-2, 1e0)
    ax.set_yscale("log")
    ax.set_xlabel(f"Number of {'bad ' if unsafe_only else ''}queries")
    ax.set_ylabel("Median distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path))
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=False, default=None)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--unsafe-only", action="store_true", default=False)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--checksum-check", action="store_true", default=False)
    parser.add_argument("--to-simulate", type=int, nargs="+", required=False, default=None)
    args = parser.parse_args()
    plot_median_distances_per_query(args.exp_paths, args.names, args.max_queries, args.max_samples, args.unsafe_only,
                                    args.out_path, args.checksum_check, args.to_simulate)
    for f in OPENED_FILES:
        f.close()
