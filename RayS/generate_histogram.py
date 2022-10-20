from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    bad_queries_array = np.load(args.results_dir / "bad_queries.npy")
    plt.hist(bad_queries_array, 50, (1, 500))
    plt.savefig(args.results_dir / "bad_queries_histogram.pdf")
    queries_quantiles = {}
    for queries in (1, 2, 5, 10, 20, 50, 100):
        images_fraction = (bad_queries_array <= queries).mean()
        print(f"{images_fraction * 100}% images are perturbed with <= {queries} bad queries")
        queries_quantiles[queries] = images_fraction
    with open(args.results_dir / "queries_quantiles.json", "w") as f:
        json.dump(queries_quantiles, f, indent=True)
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    main(args)