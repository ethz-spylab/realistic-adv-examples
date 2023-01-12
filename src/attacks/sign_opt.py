import warnings
from typing import Tuple

import numpy as np
import torch
from foolbox.distances import LpDistance

from src.attacks.base import Bounds, ExtraResultsDict, SearchMode
from src.attacks.opt import INITIAL_OVERSHOOT_EMA_VALUE, OPT, EMAValue, OPTAttackPhase, normalize
from src.attacks.queries_counter import QueriesCounter
from src.model_wrappers import ModelWrapper

start_learning_rate = 1.0


class SignOPT(OPT):
    def __init__(
        self,
        epsilon: float | None,
        distance: LpDistance,
        bounds: Bounds,
        discrete: bool,
        queries_limit: int | None,
        unsafe_queries_limit: int | None,
        max_iter: int,
        alpha: float,
        beta: float,
        num_grad_queries: int,
        search: SearchMode,
        grad_estimation_search: SearchMode,
        step_size_search: SearchMode,
        n_searches: int,
        max_search_steps: int,
        momentum: float = 0.,
        grad_batch_size: int | None = None,
    ):
        super().__init__(epsilon, distance, bounds, discrete, queries_limit, unsafe_queries_limit, max_iter, alpha,
                         beta, search, grad_estimation_search, step_size_search, n_searches, max_search_steps)
        self.num_grad_queries = num_grad_queries  # Num queries for grad estimate (default: 200)
        self.num_directions = 100
        self.momentum = momentum  # (default: 0)
        if grad_batch_size is not None:
            self.grad_batch_size = min(grad_batch_size, self.num_grad_queries)
        else:
            self.grad_batch_size = self.num_grad_queries

        # Args needed for targeted attack
        # self.tgt_init_query = args["signopt_tgt_init_query"]
        # self.targeted_dataloader = targeted_dataloader

    def __call__(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            label: torch.Tensor,
            target: torch.Tensor | None = None) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        if target is not None:
            if self.momentum > 0:
                warnings.warn("Currently, targeted Sign-OPT does not support momentum, ignoring argument.")
            raise NotImplementedError('Targeted attack is not implemented for OPT')
        return self.attack_untargeted(model, x, label)

    def attack_untargeted(self, model: ModelWrapper, x: torch.Tensor,
                          y: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        """Attack the original image and return adversarial example
        (x0, y0): original image
        """
        queries_counter = self._make_queries_counter()
        target = None

        # Calculate a good starting point.
        best_theta, g_theta = None, float("inf")
        if self.verbose:
            print(f"Searching for the initial direction on {self.num_directions} random directions: ")
        for i in range(self.num_directions):
            theta = torch.randn_like(x)
            success, queries_counter = self.is_correct_boundary_side(model, x + theta, y, target, queries_counter,
                                                                     OPTAttackPhase.direction_search, x)
            if success.item():
                theta, initial_lbd = normalize(theta)
                lbd, queries_counter, _, _, _ = self.fine_grained_search(model, x, y, target, theta, queries_counter,
                                                                         initial_lbd.item(), g_theta)
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    if self.verbose:
                        print("--------> Found distortion %.4f" % g_theta)

        if g_theta == float("inf"):
            print("Failed to find a good initial direction.")
            return x, queries_counter, float("inf"), False, {}
        else:
            assert best_theta is not None

        if self.verbose:
            print("==========> Found best distortion %.4f "
                  "using %d queries and %d unsafe queries" %
                  (g_theta, queries_counter.total_queries, queries_counter.total_unsafe_queries))

        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        best_pert = gg * xg
        vg = torch.zeros_like(xg)
        alpha, beta = self.alpha, self.beta
        search_lower_bound = EMAValue(1 - (INITIAL_OVERSHOOT_EMA_VALUE - 1), )

        for i in range(self.iterations):
            sign_gradient, queries_counter = self.sign_grad_v2(model,
                                                               x.squeeze(0),
                                                               y,
                                                               None,
                                                               xg.squeeze(0),
                                                               initial_lbd=gg,
                                                               queries_counter=queries_counter,
                                                               h=beta)

            # Line search
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if self.momentum > 0:
                    new_vg = self.momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta, _ = normalize(new_theta)
                new_g2, queries_counter, _, search_lower_bound, _ = self.step_size_search_search_fn(
                    model, x, y, target, new_theta, queries_counter, min_g2, beta / 500, search_lower_bound)
                alpha *= 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if self.momentum > 0:
                        min_vg = new_vg  # type: ignore
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha *= 0.25
                    if self.momentum > 0:
                        new_vg = self.momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, queries_counter, _, search_lower_bound, _ = self.step_size_search_search_fn(
                        model, x, y, target, new_theta, queries_counter, min_g2, beta / 500, search_lower_bound)
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        if self.momentum > 0:
                            min_vg = new_vg  # type: ignore
                        break

            if alpha < 1e-4:
                alpha = 1.0
                if self.verbose:
                    print("Warning: not moving")
                beta *= 0.1
                if beta < 1e-8:
                    break

            xg, gg = min_theta, min_g2
            vg = min_vg

            # EDIT: terminate as soon as max queries are used
            if queries_counter.is_out_of_queries():
                break
            best_pert = gg * xg

            if i % 5 == 0 and self.verbose:
                print("Iteration %3d distortion %.4f num_queries %d unsafe queries %d" %
                      (i + 1, gg, queries_counter.total_queries, queries_counter.total_unsafe_queries))

        if self.verbose:
            target = model.predict_label(x + best_pert)
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d unsafe queries %d" %
                  (gg, target, queries_counter.total_queries, queries_counter.total_unsafe_queries))

        x_adv = self.get_x_adv(x, xg, gg)

        return x_adv, queries_counter, gg, not queries_counter.is_out_of_queries(), {}

    def sign_grad_v2(self,
                     model,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     target: torch.Tensor | None,
                     theta: torch.Tensor,
                     initial_lbd: float,
                     queries_counter: QueriesCounter,
                     h: float = 0.001) -> Tuple[torch.Tensor, QueriesCounter]:
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \\sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = torch.zeros_like(theta)
        num_batches = int(np.ceil(self.num_grad_queries / self.grad_batch_size))
        assert num_batches * self.grad_batch_size == self.num_grad_queries
        x = x.unsqueeze(0)
        x_temp = self.get_x_adv(x, theta, initial_lbd)

        for _ in range(num_batches):
            u = torch.randn((self.grad_batch_size, ) + theta.shape, dtype=theta.dtype, device=x.device)
            u, _ = normalize(u, batch=True)

            sign_v = torch.ones((self.grad_batch_size, 1, 1, 1), device=x.device)
            new_theta: torch.Tensor = theta + h * u  # type: ignore
            new_theta, _ = normalize(new_theta, batch=True)

            x_ = self.get_x_adv(x, new_theta, initial_lbd)
            u = x_ - x_temp
            success, queries_counter = self.is_correct_boundary_side(model, x_, y, target, queries_counter,
                                                                     OPTAttackPhase.gradient_estimation, x)

            sign_v[success] = -1

            sign_grad += (u.sign() * sign_v).sum(0)

        sign_grad /= self.num_grad_queries
        return sign_grad, queries_counter


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = torch.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign
