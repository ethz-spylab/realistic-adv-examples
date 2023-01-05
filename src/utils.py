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
    for i in range(1, max_unsafe_queries + 1):
        binomial_product = math.comb(x, i)
        result += binomial_product
        if result >= max_steps:
            break
    return result


@functools.lru_cache(maxsize=None)
def compute_steps_to_try(max_steps: int, max_unsafe_queries: int) -> int:
    upper = max_steps
    inf = 0

    while upper - inf > 1:
        mid = (upper + inf) // 2
        if binomial_sum(mid, max_unsafe_queries, max_steps) < max_steps:
            inf = mid
        else:
            upper = mid
    return inf + 1


def eggs_dropping_search(model: ModelWrapper, attack: DirectionAttack, x: torch.Tensor, y: torch.Tensor,
                         target: torch.Tensor | None, direction: torch.Tensor, initial_distance: float,
                         queries_counter: QueriesCounter, step_size: float, max_steps: int, max_unsafe_queries: int,
                         attack_phase: AttackPhase) -> tuple[float, QueriesCounter]:
    # Consider as starting distance the case where we do no steps
    distance = initial_distance
    steps_done = 0

    # Exit if we finished the steps (i.e., exhausted the search, or if we did too many unsafe queries)
    while max_steps > 0 and max_unsafe_queries > 0:
        # Compute how many steps we do in the worst case with the given max steps and max unsafe queries
        steps_to_try = compute_steps_to_try(max_steps, max_unsafe_queries)
        # Do the steps, compute the adversarial example, and see if the query was safe
        tmp_steps_done = steps_done + steps_to_try
        distance = initial_distance - (tmp_steps_done * step_size)
        x_adv = attack.get_x_adv(x, direction, distance)
        success, queries_counter = attack.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                   attack_phase, x)
        if success.item():
            # If the query was safe, then consider the case when we have progressed by `worst_case_steps` steps,
            # which means that we have `max_steps - steps_to_try` steps left and we did `steps_done` steps
            max_steps -= steps_to_try
            steps_done = tmp_steps_done
        else:
            # Otherwise, exclude the the current step, and consider the case where we haven't progressed,
            # and we have one less unsafe query to do
            max_steps = tmp_steps_done - 1
            max_unsafe_queries -= 1

    return distance, queries_counter
