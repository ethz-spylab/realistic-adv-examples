# Adapted from
# https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/blended_noise.py
import warnings
from typing import Callable

import eagerpy as ep
import numpy as np
import torch
from foolbox.distances import LpDistance

from src.attacks.base import Bounds, ExtraResultsDict, PerturbationAttack
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.attacks.utils import atleast_kd
from src.model_wrappers import ModelWrapper


class LinearSearchBlendedUniformNoiseAttack(PerturbationAttack):
    """Blends the input with a uniform noise input until it is misclassified.
    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
        directions : Number of random directions in which the perturbation is searched.
        steps : Number of blending steps between the original image and the random
            directions.
    """

    def __init__(self,
                 epsilon: float | None,
                 distance: LpDistance,
                 bounds: Bounds,
                 discrete: bool,
                 queries_limit: int | None,
                 unsafe_queries_limit: int | None,
                 attack_phase: AttackPhase,
                 steps: int = 1000,
                 directions: int = 1000):
        super().__init__(epsilon, distance, bounds, discrete, queries_limit, unsafe_queries_limit)
        self.attack_phase = attack_phase
        self.directions = directions
        self.steps = steps

        if directions <= 0:
            raise ValueError("directions must be larger than 0")

    def __call__(
        self,
        model: ModelWrapper,
        x: torch.Tensor,
        label: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:

        x_ep, restore_type = ep.astensor_(x)
        queries_counter = self._make_queries_counter()

        is_adversarial: Callable[[ep.Tensor, QueriesCounter], tuple[torch.Tensor, QueriesCounter]]
        is_adversarial = lambda x_adv_, queries_counter_: self.is_correct_boundary_side(
            model, x_adv_.raw, label, target, queries_counter_, self.attack_phase, x_ep.raw)

        min_, max_ = self.bounds

        n_samples = len(x)

        is_adv = None
        random = None
        for j in range(self.directions):
            # random noise inputs tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random_ = ep.uniform(x_ep, x_ep.shape, min_, max_)
            success, queries_counter = is_adversarial(random_, queries_counter)
            is_adv_ = atleast_kd(ep.astensor(success), x_ep.ndim)

            if j == 0:
                random = random_
                is_adv = is_adv_
            else:
                assert random is not None
                assert is_adv is not None
                random = ep.where(is_adv, random, random_)
                is_adv = is_adv.logical_or(is_adv_)

            if is_adv.all():
                break

        assert is_adv is not None
        assert random is not None

        if not is_adv.all():
            warnings.warn(f"{self.__class__.__name__} failed to draw sufficient random"
                          f" inputs that are adversarial ({is_adv.sum()} / {n_samples}).")
            return x_ep.raw, queries_counter, np.nan, False, {}

        x0 = x_ep

        initial_step = 1 / (self.steps)
        epsilons = np.linspace(initial_step, 1, num=self.steps, dtype=np.float32)
        best = ep.ones(x_ep, (n_samples, ))

        for epsilon in epsilons:
            x_adv = (1 - epsilon) * x0 + epsilon * random
            is_adv, queries_counter = is_adversarial(x_adv, queries_counter)

            epsilon = epsilon.item()

            best = ep.minimum(ep.where(ep.astensor(is_adv), epsilon, 1.0), best)

            if (best < 1).all():
                break

        best = atleast_kd(best, x0.ndim)
        x_adv = (1 - best) * x0 + best * random
        distance = self.distance(restore_type(x0), restore_type(x_adv)).item()

        return x_adv.raw, queries_counter, distance, True, {}
