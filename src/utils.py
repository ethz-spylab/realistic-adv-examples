import functools
import math

import torch
from foolbox.distances import LpDistance, l2
from src.attacks.base import DirectionAttack
from src.attacks.queries_counter import AttackPhase, QueriesCounter

from src.model_wrappers.general_model import ModelWrapper


def compute_distance(x_ori: torch.Tensor, x_pert: torch.Tensor, distance: LpDistance = l2) -> float:
    # Compute the distance between two images.
    return distance(x_ori, x_pert).item()


def binomial_sum(x: int, max_unsafe_queries: int, max_steps: int):
    result = 0
    binomial_product = 1
    for i in range(1, max_steps + 1):
        binomial_product = math.comb(max_unsafe_queries, i)
        result += binomial_product
        if result > x:
            break
    return result


@functools.lru_cache(maxsize=None)
def compute_worst_case_queries(max_steps: int, max_unsafe_queries: int) -> int:
    upper = max_steps
    inf = 0
    mid = (upper - inf) // 2
    while upper - inf > 1:
        if binomial_sum(mid, max_unsafe_queries, max_steps) < max_unsafe_queries:
            upper = mid
        else:
            inf = mid
        mid = (upper + inf) // 2
    return inf + 1


def egg_dropping_search(model: ModelWrapper, attack: DirectionAttack, x: torch.Tensor, y: torch.Tensor,
                        target: torch.Tensor | None, direction: torch.Tensor, queries_counter: QueriesCounter,
                        step_size: float, max_steps: int, max_unsafe_queries: int,
                        attack_phase: AttackPhase) -> tuple[float, QueriesCounter]:
    distance = step_size
    steps_done = 0
    
    while max_steps > 0 and max_unsafe_queries > 0:
        worst_case_queries = compute_worst_case_queries(max_steps, max_unsafe_queries)
        tmp_steps_done = steps_done + worst_case_queries
        distance = tmp_steps_done * step_size
        x_adv = attack.get_x_adv(x, direction, distance)
        success, queries_counter = attack.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                   attack_phase)
        if success.item():
            max_steps -= tmp_steps_done
            steps_done = tmp_steps_done
        else:
            max_steps = tmp_steps_done - 1
            max_unsafe_queries -= 1

    return distance, queries_counter
