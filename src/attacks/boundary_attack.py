# Adapted from
# https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/boundary_attack.py
from typing import Callable

import eagerpy as ep
import numpy as np
import torch
from foolbox.distances import LpDistance, l2

from src.attacks.base import Bounds, ExtraResultsDict, PerturbationAttack
from src.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.attacks.utils import atleast_kd, flatten
from src.model_wrappers import ModelWrapper


class BoundaryAttackPhase(AttackPhase):
    initialization = "initialization"
    candidates_test = "candidates_test"
    stats_update = "stats_update"


class BoundaryAttack(PerturbationAttack):
    distance = l2

    def __init__(self,
                 epsilon: float | None,
                 distance: LpDistance,
                 bounds: Bounds,
                 discrete: bool,
                 queries_limit: int | None,
                 unsafe_queries_limit: int | None,
                 steps: int = 25000,
                 spherical_step: float = 1e-2,
                 source_step: float = 1e-2,
                 source_step_convergance: float = 1e-7,
                 step_adaptation: float = 1.5,
                 update_stats_every_k: int = 10):
        super().__init__(epsilon, distance, bounds, discrete, queries_limit, unsafe_queries_limit)
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergance = source_step_convergance
        self.step_adaptation = step_adaptation
        self.update_stats_every_k = update_stats_every_k

    def __call__(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            label: torch.Tensor,
            target: torch.Tensor | None = None) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:

        x, restore_type = ep.astensor_(x)
        queries_counter = self._make_queries_counter()

        is_adversarial: Callable[[ep.Tensor, QueriesCounter, AttackPhase], tuple[torch.Tensor, QueriesCounter]]
        is_adversarial = lambda x_, queries_counter_, attack_phase: self.is_correct_boundary_side(
            model, restore_type(x_), label, target, queries_counter_, attack_phase, x)
        init_attack = LinearSearchBlendedUniformNoiseAttack(self.epsilon,
                                                            self.distance,
                                                            self.bounds,
                                                            self.discrete,
                                                            None,
                                                            None,
                                                            BoundaryAttackPhase.initialization,
                                                            steps=50)
        best_advs = init_attack(model, x, label)
        is_adv_torch, queries_counter = is_adversarial(best_advs, queries_counter, BoundaryAttackPhase.initialization)
        is_adv = ep.astensor(is_adv_torch)

        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            print("Failed to find adversarial examples for {} samples".format(failed))
            return x, queries_counter, float("inf"), False, {}

        n_samples = len(x)
        ndim = x.ndim
        spherical_steps = ep.ones(x, n_samples) * self.spherical_step
        source_steps = ep.ones(x, n_samples) * self.source_step

        # create two queues for each sample to track success rates
        # (used to update the hyperparameters)
        stats_spherical_adversarial = ArrayQueue(maxlen=100, n_samples=n_samples)
        stats_step_adversarial = ArrayQueue(maxlen=30, n_samples=n_samples)

        bounds = model.bounds

        for step in range(1, self.steps + 1):
            converged = source_steps < self.source_step_convergance
            if converged.all():
                break  # pragma: no cover
            converged = atleast_kd(converged, ndim)

            unnormalized_source_directions = x - best_advs
            source_norms = ep.norms.l2(flatten(unnormalized_source_directions), axis=-1)
            source_directions = unnormalized_source_directions / atleast_kd(source_norms, ndim)

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % self.update_stats_every_k == 0

            candidates, spherical_candidates = draw_proposals(
                bounds,
                x,
                best_advs,
                unnormalized_source_directions,
                source_directions,
                source_norms,
                spherical_steps,
                source_steps,
            )

            is_adv_torch, queries_counter = is_adversarial(candidates, queries_counter,
                                                           BoundaryAttackPhase.candidates_test)
            is_adv = ep.astensor(is_adv_torch)

            if check_spherical_and_update_stats:
                spherical_is_adv, queries_counter = is_adversarial(spherical_candidates, queries_counter,
                                                                   BoundaryAttackPhase.stats_update)
                stats_spherical_adversarial.append(ep.astensor(spherical_is_adv))
                stats_step_adversarial.append(is_adv)

            # in theory, we are closer per construction
            # but limited numerical precision might break this
            distances = ep.norms.l2(flatten(x - candidates), axis=-1)
            closer = distances < source_norms
            is_best_adv = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)

            cond = converged.logical_not().logical_and(is_best_adv)
            best_advs = ep.where(cond, candidates, best_advs)

            if check_spherical_and_update_stats:
                full = stats_spherical_adversarial.isfull()
                if full.any():
                    probs = stats_spherical_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.5, full)
                    spherical_steps = ep.where(cond1, spherical_steps * self.step_adaptation, spherical_steps)
                    source_steps = ep.where(cond1, source_steps * self.step_adaptation, source_steps)
                    cond2 = ep.logical_and(probs < 0.2, full)
                    spherical_steps = ep.where(cond2, spherical_steps / self.step_adaptation, spherical_steps)
                    source_steps = ep.where(cond2, source_steps / self.step_adaptation, source_steps)
                    stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))

                full = stats_step_adversarial.isfull()
                if full.any():
                    probs = stats_step_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.25, full)
                    source_steps = ep.where(cond1, source_steps * self.step_adaptation, source_steps)
                    cond2 = ep.logical_and(probs < 0.1, full)
                    source_steps = ep.where(cond2, source_steps / self.step_adaptation, source_steps)
                    stats_step_adversarial.clear(ep.logical_or(cond1, cond2))

            if queries_counter.is_out_of_queries():
                print("Out of queries")
                break

        distance = ep.norms.l2(flatten(x - best_advs), axis=-1).item()

        return restore_type(best_advs), queries_counter, distance, True, {}


class ArrayQueue:

    def __init__(self, maxlen: int, n_samples: int):
        # we use NaN as an indicator for missing data
        self.data = np.full((maxlen, n_samples), np.nan)
        self.next = 0
        # used to infer the correct framework because this class uses NumPy
        self.tensor: ep.Tensor | None = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    def append(self, x: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = x
        x = x.numpy()
        assert x.shape == (self.n_samples, )
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = dims  # pragma: no cover
        dims = dims.numpy()
        assert dims.shape == (self.n_samples, )
        assert dims.dtype == np.bool
        self.data[:, dims] = np.nan

    def mean(self) -> ep.Tensor:
        assert self.tensor is not None
        result = np.nanmean(self.data, axis=0)
        return ep.from_numpy(self.tensor, result)

    def isfull(self) -> ep.Tensor:
        assert self.tensor is not None
        result = ~np.isnan(self.data).any(axis=0)
        return ep.from_numpy(self.tensor, result)


def draw_proposals(
    bounds: Bounds,
    originals: ep.Tensor,
    perturbed: ep.Tensor,
    unnormalized_source_directions: ep.Tensor,
    source_directions: ep.Tensor,
    source_norms: ep.Tensor,
    spherical_steps: ep.Tensor,
    source_steps: ep.Tensor,
) -> tuple[ep.Tensor, ep.Tensor]:
    # remember the actual shape
    shape = originals.shape
    assert perturbed.shape == shape
    assert unnormalized_source_directions.shape == shape
    assert source_directions.shape == shape

    # flatten everything to (batch, size)
    originals = flatten(originals)
    perturbed = flatten(perturbed)
    unnormalized_source_directions = flatten(unnormalized_source_directions)
    source_directions = flatten(source_directions)
    n_samples, dim = originals.shape

    assert source_norms.shape == (n_samples, )
    assert spherical_steps.shape == (n_samples, )
    assert source_steps.shape == (n_samples, )

    # draw from an iid Gaussian (we can share this across the whole batch)
    eta = ep.normal(perturbed, (dim, 1))

    # make orthogonal (source_directions are normalized)
    eta = eta.T - ep.matmul(source_directions, eta) * source_directions
    assert eta.shape == (n_samples, dim)

    # rescale
    norms = ep.norms.l2(eta, axis=-1)
    assert norms.shape == (n_samples, )
    eta = eta * atleast_kd(spherical_steps * source_norms / norms, eta.ndim)

    # project on the sphere using Pythagoras
    distances = atleast_kd((spherical_steps.square() + 1).sqrt(), eta.ndim)
    directions = eta - unnormalized_source_directions
    spherical_candidates = originals + directions / distances

    # clip
    min_, max_ = bounds
    spherical_candidates = spherical_candidates.clip(min_, max_)

    # step towards the original inputs
    new_source_directions = originals - spherical_candidates
    assert new_source_directions.ndim == 2
    new_source_directions_norms = ep.norms.l2(flatten(new_source_directions), axis=-1)

    # length if spherical_candidates would be exactly on the sphere
    lengths = source_steps * source_norms

    # length including correction for numerical deviation from sphere
    lengths = lengths + new_source_directions_norms - source_norms

    # make sure the step size is positive
    lengths = ep.maximum(lengths, 0)

    # normalize the length
    lengths = lengths / new_source_directions_norms
    lengths = atleast_kd(lengths, new_source_directions.ndim)

    candidates = spherical_candidates + lengths * new_source_directions

    # clip
    candidates = candidates.clip(min_, max_)

    # restore shape
    candidates = candidates.reshape(shape)
    spherical_candidates = spherical_candidates.reshape(shape)
    return candidates, spherical_candidates
