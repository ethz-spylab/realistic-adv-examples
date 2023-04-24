import sys
from pathlib import Path
from typing import Iterator

import ijson
import tqdm

from src.json_list import JSONList

exp_directory = Path(sys.argv[1])

f = open(exp_directory / "distances_traces.json")
f_fix = open(exp_directory / "distances_traces_init_fix.json")
items: Iterator[list[dict[str, float | bool | str]]] = ijson.items(f, "item", use_float=True)
item_fixes: Iterator[list[dict[str, float | bool | str]]] = ijson.items(f_fix, "item", use_float=True)
fixed_items_jsonlist = JSONList(Path("distances_traces_fixed.json"))

MAX_ITEMS = 1000

for distances_list, distances_list_fixes in tqdm.tqdm(zip(items, item_fixes), total=MAX_ITEMS):
    fixed_distances_list = []

    i = 0
    for i, distance_fix in enumerate(distances_list_fixes):
        fixed_distances_list.append(distance_fix)

    best_distance = fixed_distances_list[-1]["best_distance"]
    for wrong_distance in distances_list[i + 1:]:
        if wrong_distance["safe"]:
            best_distance = min(best_distance, wrong_distance["distance"])
        wrong_distance["best_distance"] = best_distance
        fixed_distances_list.append(wrong_distance)

    assert len(fixed_distances_list) == len(distances_list)

    fixed_items_jsonlist.append(fixed_distances_list)

f.close()
f_fix.close()
