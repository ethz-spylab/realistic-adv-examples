import math

import eagerpy as ep
import numpy as np
import torch

from src.attacks.base import DirectionAttack, PerturbationAttack
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.model_wrappers.general_model import ModelWrapper


def atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor:
    # From https://github.com/bethgelab/foolbox/blob/master/foolbox/devutils.py
    shape = x.shape + (1, ) * (k - x.ndim)
    return x.reshape(shape)


def flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor:
    # From https://github.com/bethgelab/foolbox/blob/master/foolbox/devutils.py
    return x.flatten(start=keep)


DEFAULT_LINE_SEARCH_TOL = 1e-5
MAX_BATCH_SIZE = 100


def opt_binary_search(attack: DirectionAttack | PerturbationAttack,
                      model: ModelWrapper,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      target: torch.Tensor | None,
                      theta: torch.Tensor,
                      queries_counter: QueriesCounter,
                      initial_lbd: float,
                      phase: AttackPhase,
                      first_step_phase: AttackPhase | None = None,
                      tol: float = DEFAULT_LINE_SEARCH_TOL) -> tuple[float, QueriesCounter, float]:
    lbd = initial_lbd

    if isinstance(attack, DirectionAttack):

        def is_correct_boundary_side_local(lbd_: float, qc: QueriesCounter) -> tuple[torch.Tensor, QueriesCounter]:
            x_adv_ = attack.get_x_adv(x, theta, lbd_)
            return attack.is_correct_boundary_side(model, x_adv_, y, target, qc, phase, x)

        x_adv = attack.get_x_adv(x, theta, lbd)
    elif isinstance(attack, PerturbationAttack):

        def is_correct_boundary_side_local(lbd_: float, qc: QueriesCounter) -> tuple[torch.Tensor, QueriesCounter]:
            x_adv_ = attack.get_x_adv(x, theta * lbd_)
            return attack.is_correct_boundary_side(model, x_adv_, y, target, qc, phase, x)

        x_adv = attack.get_x_adv(x, theta * lbd)

    initial_phase = first_step_phase or phase
    success, queries_counter = attack.is_correct_boundary_side(model, x_adv, y, target, queries_counter, initial_phase,
                                                               x)

    if not success:
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        while not (iter_result := is_correct_boundary_side_local(lbd_hi, queries_counter))[0].item():
            _, queries_counter = iter_result
            lbd_hi *= 1.01
            if lbd_hi > 20:
                # Here we return 2 * lbd_hi because inf breaks the attack
                return lbd_hi * 2, queries_counter, (lbd_hi / lbd) * 2
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        while (iter_result := is_correct_boundary_side_local(lbd_lo, queries_counter))[0].item():
            _, queries_counter = iter_result
            lbd_lo *= 0.99

    lbd_factor = lbd_hi / lbd
    diff = lbd_hi - lbd_lo
    while diff > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2
        # EDIT: add a break condition
        if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
            break
        success, queries_counter = is_correct_boundary_side_local(lbd_mid, queries_counter)
        if success.item():
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
        # EDIT: This is to avoid numerical issue with gpu tensor when diff is small
        if diff <= lbd_hi - lbd_lo:
            break
        diff = lbd_hi - lbd_lo
    return lbd_hi, queries_counter, lbd_factor


def opt_line_search(attack: PerturbationAttack | DirectionAttack,
                    model: ModelWrapper,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    target: torch.Tensor | None,
                    theta: torch.Tensor,
                    queries_counter: QueriesCounter,
                    initial_lbd: float,
                    phase: AttackPhase,
                    initial_phase: AttackPhase,
                    current_best: float | None,
                    n_searches: int,
                    max_search_steps: int,
                    batch_size: int,
                    lower_b: float | None = None,
                    upper_b: float | None = None) -> tuple[float, QueriesCounter]:
    if current_best is not None and initial_lbd > current_best:
        if isinstance(attack, DirectionAttack):
            x_adv = attack.get_x_adv(x, theta, current_best)
        elif isinstance(attack, PerturbationAttack):
            x_adv = attack.get_x_adv(x, theta * current_best)
        success, queries_counter = attack.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                   initial_phase, x)
        if not success.item():
            return float('inf'), queries_counter
        lbd = current_best
    else:
        lbd = initial_lbd

    if lower_b is not None:
        lower_lbd = lbd * lower_b
    else:
        lower_lbd = 0.

    if upper_b is not None:
        lbd = lbd * upper_b

    assert n_searches in {1, 2}
    if n_searches == 2:
        search_max_steps = math.ceil(math.sqrt(max_search_steps))
    else:
        search_max_steps = max_search_steps
    search_batch_size = min(search_max_steps, batch_size)
    first_search_step_size = (lbd - lower_lbd) / search_max_steps

    first_search_lbd, first_search_queries_counter, first_query_failed = _batched_line_search_body(
        attack,
        model,
        x,
        y,
        target,
        theta,
        queries_counter,
        lbd,
        phase,
        first_search_step_size,
        search_batch_size,
        # Here we count each query of the first search as equivalent to search_max_steps queries of when we do 1
        equivalent_simulated_queries=search_max_steps,
        # But we don't count the queries from the last batch as they will be counted in the second search
        count_last_batch_for_sim=False)

    if first_query_failed:
        lbd_to_return = lbd * 2
        if upper_b is not None:
            print("Warning: line search overshoot was not enough")
        return lbd_to_return, first_search_queries_counter

    if n_searches == 2:
        second_search_step_size = first_search_step_size / search_max_steps
        final_lbd, second_search_queries_counter, _ = _batched_line_search_body(
            attack,
            model,
            x,
            y,
            target,
            theta,
            first_search_queries_counter,
            first_search_lbd,
            phase,
            second_search_step_size,
            search_batch_size,
            # Here each query has the same step size as if we were doing one search only
            equivalent_simulated_queries=1,
            # And we count the queries from the last batch as they are not counted in the first search
            count_last_batch_for_sim=True)
    else:
        second_search_queries_counter = first_search_queries_counter
        final_lbd = first_search_lbd

    return final_lbd, second_search_queries_counter


def _batched_line_search_body(attack: PerturbationAttack | DirectionAttack,
                              model: ModelWrapper,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              target: torch.Tensor | None,
                              theta: torch.Tensor,
                              queries_counter: QueriesCounter,
                              initial_lbd: float,
                              phase: AttackPhase,
                              step_size: float,
                              batch_size: int = MAX_BATCH_SIZE,
                              equivalent_simulated_queries: int = 1,
                              count_last_batch_for_sim: bool = False) -> tuple[float, QueriesCounter, bool]:
    success = torch.tensor([True])
    batch_idx = 0
    lbds_inner_shape = tuple([1] * (len(x.shape) - 1))
    previous_last_lbd = torch.tensor([initial_lbd])
    lbds = np.array([initial_lbd])

    while success.all():
        # Update the last lbd (in case the whole next batch is unsafe) and the index
        previous_last_lbd = lbds[-1]
        # Get steps bounds based on the batch index
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        # Compute the steps to do
        steps_sizes = np.arange(start, end) * step_size
        # Subtract the steps from the original distance
        lbds = (initial_lbd - steps_sizes).reshape(-1, *lbds_inner_shape)
        # Compute advex and query the model
        lbds_torch = torch.from_numpy(lbds).float().to(device=x.device)
        if isinstance(attack, DirectionAttack):
            batch = attack.get_x_adv(x, theta, lbds_torch)
        elif isinstance(attack, PerturbationAttack):
            batch = attack.get_x_adv(x, theta * lbds_torch.unsqueeze(-1))
        success, queries_counter = attack.is_correct_boundary_side_batched(model, batch, y, target, queries_counter,
                                                                           phase, x, equivalent_simulated_queries,
                                                                           count_last_batch_for_sim, batch_idx == 0)
        batch_idx += 1

    assert lbds is not None
    # We get the index of the first unsafe query
    unsafe_query_idx = torch.argmin(success.to(torch.int))
    if unsafe_query_idx == 0:
        # If no query was safe in the latest batch, then we return the last lbd from the previous batch
        lbd = previous_last_lbd.item()
    else:
        lbd = lbds[unsafe_query_idx - 1].item()

    # If we exited the loop after the first batch and the very first element was unsafe, then it means that
    # the first query was unsafe
    first_query_failed = batch_idx == 1 and bool((unsafe_query_idx == 0).item())
    return lbd, queries_counter, first_query_failed