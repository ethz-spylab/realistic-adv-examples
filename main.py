import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

import setup
from attacks.queries_counter import QueriesCounter
from setup import setup_out_dir
from src.setup import setup_model_and_data


def aggregate_queries_counters_list(q_list: list[QueriesCounter]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    aggregated_queries = {}
    for safe_queries in [q.queries for q in q_list]:
        for phase, n_queries in safe_queries.items():
            aggregated_queries[phase] = aggregated_queries.get(phase, []) + [n_queries]

    aggregated_unsafe_queries = {}
    for unsafe_queries in [q.unsafe_queries for q in q_list]:
        for phase, n_queries in unsafe_queries.items():
            aggregated_unsafe_queries[phase] = aggregated_unsafe_queries.get(phase, []) + [n_queries]

    return aggregated_queries, aggregated_unsafe_queries


def aggregate_extra_results(extra_results_list: list[dict[str, float]]) -> dict[str, list[float]]:
    aggregated_extra_results = {}
    for extra_results in extra_results_list:
        for key, value in extra_results.items():
            aggregated_extra_results[key] = aggregated_extra_results.get(key, []) + [value]
    return aggregated_extra_results


@dataclasses.dataclass
class AttackResults:
    successes: int = 0
    distances: list[float] = dataclasses.field(default_factory=list)
    queries_counters: list[QueriesCounter] = dataclasses.field(default_factory=list)
    extra_results: list[dict[str, float]] = dataclasses.field(default_factory=list)
    failures: int = 0
    failed_distances: list[float] = dataclasses.field(default_factory=list)
    failed_queries_counters: list[QueriesCounter] = dataclasses.field(default_factory=list)
    failed_extra_results: list[dict[str, float]] = dataclasses.field(default_factory=list)

    def update_with_success(self, distance: float, queries_counter: QueriesCounter,
                            extra_results: dict[str, float]) -> "AttackResults":
        return dataclasses.replace(self,
                                   successes=self.successes + 1,
                                   distances=self.distances + [distance],
                                   queries_counters=self.queries_counters + [queries_counter],
                                   extra_results=self.extra_results + [extra_results])

    def update_with_failure(self, distance: float, queries_counter: QueriesCounter,
                            extra_results: dict[str, float]) -> "AttackResults":
        return dataclasses.replace(self,
                                   failures=self.failures + 1,
                                   failed_distances=self.failed_distances + [distance],
                                   failed_queries_counters=self.failed_queries_counters + [queries_counter],
                                   failed_extra_results=self.failed_extra_results + [extra_results])

    def log_results(self, idx: int):
        print(f"index: {idx:4d} avg dist: {np.mean(self.distances):.4f} "
              f"avg queries: {np.mean(self._get_overall_queries()):.4f} "
              f"median queries: {np.median(self._get_overall_queries()):.4f} "
              f"avg bad queries: {np.mean(self._get_overall_unsafe_queries()):.4f} "
              f"median bad queries: {np.median(self._get_overall_unsafe_queries()):.4f} "
              f"asr: {np.mean(np.array(self.asr)):.4f} \n")

    def get_results_dict(self) -> dict[str, float]:
        results_dict = {"asr": self.asr, "distortion": np.mean(self.distances)}
        aggregated_queries, aggregated_unsafe_queries = aggregate_queries_counters_list(self.queries_counters)
        for stat, stat_fn in (("mean", np.mean), ("median", np.median)):
            for phase, queries_list in aggregated_queries.items():
                results_dict[f"{stat}_queries_{phase}"] = stat_fn(queries_list)
            for phase, queries_list in aggregated_unsafe_queries.items():
                results_dict[f"{stat}_unsafe_queries_{phase}"] = stat_fn(queries_list)
            for key, value_list in aggregate_extra_results(self.extra_results).items():
                results_dict[f"{stat}_{key}"] = stat_fn(value_list)

        return results_dict

    def save_results(self, out_dir: Path):
        with open(out_dir / "aggregated_results.json", 'w') as f:
            json.dump(self.get_results_dict(), f, indent=4)
        with open(out_dir / "full_results.json", 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        np.save(out_dir / "distances.npy", np.array(self.distances))
        np.save(out_dir / "queries.npy", np.array(self._get_overall_queries()))
        np.save(out_dir / "unsafe_queries.npy", np.array(self._get_overall_unsafe_queries()))
        np.save(out_dir / "failed_distances.npy", np.array(self.failed_distances))
        np.save(out_dir / "failed_queries.npy", np.array(self._get_overall_failed_queries()))
        np.save(out_dir / "failed_unsafe_queries.npy", np.array(self._get_overall_failed_unsafe_queries()))

    @property
    def asr(self) -> float:
        return self.successes / (self.successes + self.failures)

    def _get_overall_queries(self) -> list[int]:
        return list(map(lambda counter: counter.total_queries, self.queries_counters))

    def _get_overall_failed_queries(self) -> list[int]:
        return list(map(lambda counter: counter.total_queries, self.failed_queries_counters))

    def _get_overall_unsafe_queries(self) -> list[int]:
        return list(map(lambda counter: counter.total_unsafe_queries, self.queries_counters))

    def _get_overall_failed_unsafe_queries(self) -> list[int]:
        return list(map(lambda counter: counter.total_unsafe_queries, self.failed_queries_counters))


def main():
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--attack', default='rays', type=str, help='The attack to run')
    parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset')
    parser.add_argument('--targeted', default='0', type=str, help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str, help='Norm for attack, linf only')
    parser.add_argument('--num', default=1000, type=int, help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--query', default=10000, type=int, help='Maximum queries for the attack')
    parser.add_argument('--batch', default=1, type=int, help='attack batch size.')
    parser.add_argument('--epsilon', default=0.05, type=float, help='attack strength')
    parser.add_argument('--early',
                        default='1',
                        type=str,
                        help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--search', default='binary', type=str, help='Type of search to use, binary or line')
    parser.add_argument('--line-search-tol',
                        default=None,
                        type=float,
                        help='Tolerance for line search w.r.t. previous iteration')
    parser.add_argument(
        '--out-dir',
        default='/local/home/edebenedetti/exp-results/realistic-adv-examples/rays',
        type=str,
    )
    parser.add_argument(
        '--save-img-every',
        default=None,
        type=int,
    )
    parser.add_argument('--flip-squares',
                        default='0',
                        type=str,
                        help='Whether the attack should flip squares and not chunks of a 1-d vector')
    parser.add_argument('--flip-rand-pixels',
                        default='0',
                        type=str,
                        help='Whether the attack should flip random pixels not chunks of a 1-d vector')
    parser.add_argument('--discrete',
                        default='0',
                        type=str,
                        help='Whether the attack should work in discrete space (i.e., int8)')
    parser.add_argument(
        '--strong-preprocessing',
        default='0',
        type=str,
        help=
        'Whether strong preprocessing (i.e., JPEG, Resize, Crop) '
        'should be applied before feeding the image to the classifier'
    )
    parser.add_argument('--model-threshold', default=0.25, type=float, help='The threshold to use for the API model')
    args = parser.parse_args()
    load_dotenv()

    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True

    if args.flip_squares == '1' and args.flip_rand_pixels == '1':
        raise ValueError("`--flip-squares` cannot be `1` if also `--flip-rand-pixels` is `1`")

    print(args)

    device = torch.device("cuda")

    model, test_loader = setup_model_and_data(args, device)
    exp_out_dir = setup_out_dir(args)
    attack = setup.setup_attack(args)
    attack_results = AttackResults()

    seeds = np.random.randint(10000, size=10000)

    count = 0
    misclassified = 0
    negatives = 0
    for i, batch in enumerate(test_loader):
        if count == args.num:
            break

        if isinstance(batch, dict):
            xi, yi = batch["image"], batch["label"]
        else:
            xi, yi = batch

        print(f"Sample {i}, class: {yi.item()}")
        xi, yi = xi.to(device), yi.to(device)

        if model.n_class == 2 and yi.item() == 0:
            negatives += 1
            print("Skipping as item is negative")
            continue

        if model.predict_label(xi) != yi:
            misclassified += 1
            print("Skipping as item is misclassified")
            continue

        np.random.seed(seeds[count])

        target = np.random.randint(model.n_class) * torch.ones(yi.shape,
                                                               dtype=torch.long).to(device) if targeted else None
        while target and torch.sum(target == yi) > 0:
            print('re-generate target label')
            target = np.random.randint(model.n_class) * torch.ones(len(xi), dtype=torch.long).to(device)

        adv, queries_counter, dist, succ, extra_results = attack(model, xi, yi, target, args.limit)

        if args.save_img_every is not None and count % args.save_img_every == 0:
            np.save(exp_out_dir / f"{i}_adv.npy", adv[0].cpu().numpy())
            np.save(exp_out_dir / f"{i}.npy", xi[0].cpu().numpy())

        if succ or early_stopping:
            attack_results = attack_results.update_with_success(dist, queries_counter, extra_results)
        else:
            attack_results = attack_results.update_with_failure(dist, queries_counter, extra_results)

        count += 1
        attack_results.log_results(i)

    attack_results.save_results(exp_out_dir)


if __name__ == "__main__":
    main()
