import itertools
from typing import Callable

import torch
from foolbox.distances import LpDistance

from src.attacks.base import Bounds, DirectionAttack, ExtraResultsDict, SearchMode
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.attacks.utils import opt_line_search, opt_binary_search
from src.model_wrappers import ModelWrapper


class OPTAttackPhase(AttackPhase):
    direction_search = "direction_search"
    direction_probing = "direction_probing"
    gradient_estimation = "gradient_estimation"
    step_size_search = "step_size_search"
    step_size_search_start = "step_size_search_start"
    search = "search"


DEFAULT_LINE_SEARCH_TOL = 1e-5
MAX_STEPS_LINE_SEARCH = 100
MAX_STEPS_COARSE_LINE_SEARCH = 100
OVERSHOOT_VALUE = 1.01
MAX_BATCH_SIZE = 100

FineGrainedSearchFn = Callable[
    [ModelWrapper, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, QueriesCounter, float, float],
    tuple[float, QueriesCounter, float | None]]
GradientEstimationSearchFn = Callable[[
    ModelWrapper, torch.Tensor, torch.Tensor, torch.Tensor
    | None, torch.Tensor, QueriesCounter, float, float, float | None, float | None
], tuple[float, QueriesCounter, float | None]]
StepSizeSearchSearchFn = Callable[[
    ModelWrapper, torch.Tensor, torch.Tensor, torch.Tensor
    | None, torch.Tensor, QueriesCounter, float, float, float | None
], tuple[float, QueriesCounter, float | None]]


class OPT(DirectionAttack):
    verbose = True

    def __call__(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            label: torch.Tensor,
            target: torch.Tensor | None = None) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        if target is not None:
            raise NotImplementedError('Targeted attack is not implemented for OPT')
        return self.attack_untargeted(model, x, label)

    def __init__(self, epsilon: float | None, distance: LpDistance, bounds: Bounds, discrete: bool,
                 queries_limit: int | None, unsafe_queries_limit: int | None, max_iter: int | None, alpha: float,
                 beta: float, search: SearchMode, num_grad_queries: int, grad_estimation_search: SearchMode,
                 step_size_search: SearchMode, n_searches: int, max_search_steps: int, batch_size: int | None,
                 num_init_directions: int, get_one_init_direction: bool):
        super().__init__(epsilon, distance, bounds, discrete, queries_limit, unsafe_queries_limit)
        self.num_directions = num_init_directions
        self.get_one_init_direction = get_one_init_direction
        self.iterations = max_iter
        self.alpha = alpha  # 0.2
        self.beta = beta  # 0.001
        self.n_searches = n_searches
        self.max_search_steps = max_search_steps
        self.grad_estimation_search_type = grad_estimation_search
        self.batch_size = batch_size if batch_size is not None else MAX_BATCH_SIZE
        self.num_grad_queries = num_grad_queries

        if SearchMode.eggs_dropping in {search, grad_estimation_search, step_size_search}:
            raise ValueError("eggs dropping search not available for OPT and SignOPT")
        if self.n_searches not in {1, 2}:
            raise ValueError("Only 1 or 2 searches can be done in OPT.")

        self.fine_grained_search: FineGrainedSearchFn
        self.grad_estimation_search_fn: GradientEstimationSearchFn
        self.step_size_search_search_fn: StepSizeSearchSearchFn
        if search == SearchMode.binary:
            self.fine_grained_search = self.fine_grained_binary_search
        elif search == SearchMode.line:
            self.fine_grained_search = (
                lambda model, x, y, target, theta, queries_counter, initial_lbd, current_best: self.line_search(
                    model, x, y, target, theta, queries_counter, initial_lbd, OPTAttackPhase.search, current_best))

        if grad_estimation_search == SearchMode.binary:
            self.grad_estimation_search_fn = (
                lambda model, x, y, target, theta, queries_counter, initial_lbd, tol, _lower_b, _upper_b: self.
                fine_grained_binary_search_local(model, x, y, target, theta, queries_counter, initial_lbd,
                                                 OPTAttackPhase.gradient_estimation, None, tol))
        elif grad_estimation_search == SearchMode.line:
            self.grad_estimation_search_fn = (
                lambda model, x, y, target, theta, queries_counter, initial_lbd, tol, lower_b, upper_b: self.
                line_search(model, x, y, target, theta, queries_counter, initial_lbd, OPTAttackPhase.
                            gradient_estimation, None, lower_b, upper_b, tol))

        if step_size_search == SearchMode.binary:
            self.step_size_search_search_fn = (
                lambda model, x, y, target, theta,
                queries_counter, initial_lbd, tol, _lower_b: self.fine_grained_binary_search_local(
                    model, x, y, target, theta, queries_counter, initial_lbd, OPTAttackPhase.step_size_search,
                    OPTAttackPhase.step_size_search_start, tol))
        else:
            self.step_size_search_search_fn = (
                lambda model, x, y, target, theta, queries_counter, initial_lbd, tol, lower_b: self.line_search(
                    model, x, y, target, theta, queries_counter, initial_lbd, OPTAttackPhase.step_size_search, None,
                    lower_b, None, tol))

    def attack_untargeted(self, model: ModelWrapper, x: torch.Tensor,
                          y: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        """Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        """
        queries_counter = self._make_queries_counter()

        alpha, beta = self.alpha, self.beta
        grad_est_search_upper_bound = OVERSHOOT_VALUE
        grad_est_search_lower_bound = 1 - (OVERSHOOT_VALUE - 1)
        step_size_search_lower_bound = 1 - (OVERSHOOT_VALUE - 1)

        best_theta, prev_best_theta, g_theta = None, None, float("inf")
        print(f"Searching for the initial direction on {self.num_directions} random directions")

        for i in range(self.num_directions):
            theta = torch.randn_like(x)
            success, queries_counter = self.is_correct_boundary_side(model, x + theta, y, None, queries_counter,
                                                                     OPTAttackPhase.direction_search, x)
            if success.item():
                theta, initial_lbd = normalize(theta)
                lbd, queries_counter, _ = self.fine_grained_search(model, x, y, None, theta, queries_counter,
                                                                   initial_lbd.item(), g_theta)
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.log(f"---> Found distortion {g_theta:.4f}")
                if self.get_one_init_direction:
                    break

        if g_theta == float("inf"):
            # TODO: is this just trying the exact same again?
            best_theta, g_theta = None, float("inf")
            self.log(f"Searching for the initial direction on {self.num_directions} random directions")

            for i in range(self.num_directions):
                theta = torch.randn_like(x)
                success, queries_counter = self.is_correct_boundary_side(model, x + theta, y, None, queries_counter,
                                                                         OPTAttackPhase.direction_search, x)
                if success.item():
                    theta, initial_lbd = normalize(theta)
                    lbd, queries_counter, _ = self.fine_grained_search(model, x, y, None, theta, queries_counter,
                                                                       initial_lbd.item(), g_theta)
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        self.log(f"---> Found distortion {g_theta:.4f}")

        if g_theta == float("inf"):
            self.log("Couldn't find valid initial direction, failed")
            return x, queries_counter, float("inf"), False, {}

        self.log(f"====> Found best distortion {g_theta:.4f} using {queries_counter.total_queries} "
                 f"queries and {queries_counter.total_unsafe_queries} unsafe queries")

        g1 = 1.0
        assert best_theta is not None
        theta, g2 = best_theta, g_theta
        lbd_factors = []

        if self.iterations is not None:
            _range = range(self.iterations)
        else:
            _range = itertools.count()

        for i in _range:
            q = self.num_grad_queries
            min_g1 = float("inf")
            gradient = torch.zeros_like(theta)
            u = torch.randn((q, ) + theta.shape, device=theta.device)
            u, _ = normalize(u, batch=True)
            ttt = theta.unsqueeze(0) + beta * u
            ttt, _ = normalize(ttt, batch=True)
            for j in range(q):
                g1, queries_counter, lbd_factor = (self.grad_estimation_search_fn(model, x, y, None, ttt[j],
                                                                                  queries_counter, g2, beta / 500,
                                                                                  grad_est_search_lower_bound,
                                                                                  grad_est_search_upper_bound))
                gradient += (g1 - g2) / beta * u[j]
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt[j]
                lbd_factors.append(lbd_factor)
            gradient = 1.0 / q * gradient

            if (i + 1) % 10 == 0:
                dist = (g2 * theta).norm().item()
                self.log((f"Iteration {i + 1:3d} distortion {dist:.4f} num_queries {queries_counter.total_queries}, "
                          f"unsafe queries: {queries_counter.total_unsafe_queries}"))

            min_theta = theta
            min_g2 = g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta, _ = normalize(new_theta)
                new_g2, queries_counter, lbd_factor = self.step_size_search_search_fn(
                    model, x, y, None, new_theta, queries_counter, min_g2, beta / 500, step_size_search_lower_bound)
                lbd_factors.append(lbd_factor)
                alpha *= 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha *= 0.25
                    new_theta = theta - alpha * gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, queries_counter, lbd_factor = (self.step_size_search_search_fn(
                        model, x, y, None, new_theta, queries_counter, min_g2, beta / 500,
                        step_size_search_lower_bound))
                    lbd_factors.append(lbd_factor)
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1  # type: ignore

            if g2 < g_theta:
                best_theta, g_theta = theta, g2

            if alpha < 1e-4:
                alpha = 1.0
                self.log(f"Warning: not moving, g2 {g2:.4f} gtheta {g_theta:.4f}")
                beta *= 0.1
                if beta < 1e-8:
                    break

            # prev_best_theta is kept to make sure that we use the latest theta
            # before max query is reached
            if queries_counter.is_out_of_queries():
                print("Out of queries")
                break

            prev_best_theta = best_theta.clone()

        assert prev_best_theta is not None
        target = model.predict_label(self.get_x_adv(x, prev_best_theta, g_theta)).item()

        self.log(f"\nAdversarial example found: distortion {g_theta:.4f} predicted class {target} queries "
                 f"{queries_counter.total_queries}, unsafe queries: {queries_counter.total_unsafe_queries}")

        distance: float = (x + g_theta * prev_best_theta - x).norm().item()

        extra_results: ExtraResultsDict = {
            "lbd_factors": lbd_factors,
        }

        return (self.get_x_adv(x, prev_best_theta, g_theta), queries_counter, distance, True, extra_results)

    def log(self, arg):
        if self.verbose:
            print(arg)

    def fine_grained_binary_search_local(self,
                                         model: ModelWrapper,
                                         x: torch.Tensor,
                                         y: torch.Tensor,
                                         target: torch.Tensor | None,
                                         theta: torch.Tensor,
                                         queries_counter: QueriesCounter,
                                         initial_lbd: float,
                                         phase: OPTAttackPhase,
                                         first_step_phase: OPTAttackPhase | None = None,
                                         tol: float = DEFAULT_LINE_SEARCH_TOL) -> tuple[float, QueriesCounter, float]:
        return opt_binary_search(self, model, x, y, target, theta, queries_counter, initial_lbd, phase,
                                 first_step_phase, tol)

    def fine_grained_binary_search(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor,
                                   target: torch.Tensor | None, theta: torch.Tensor, queries_counter: QueriesCounter,
                                   initial_lbd: float, current_best: float) -> tuple[float, QueriesCounter, None]:
        if initial_lbd > current_best:
            x_adv = self.get_x_adv(x, theta, current_best)
            success, queries_counter = self.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                     OPTAttackPhase.direction_probing, x)
            if not success.item():
                return float('inf'), queries_counter, None
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        # EDIT: This tol check has a numerical issue and may never quit (1e-5)
        while lbd_hi - lbd_lo > DEFAULT_LINE_SEARCH_TOL:
            lbd_mid = (lbd_lo + lbd_hi) / 2
            # EDIT: add a break condition
            if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
                break
            x_adv = self.get_x_adv(x, theta, lbd_mid)
            success, queries_counter = self.is_correct_boundary_side(model, x_adv, y, target, queries_counter,
                                                                     OPTAttackPhase.search, x)
            if success.item():
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, queries_counter, None

    def line_search(self,
                    model: ModelWrapper,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    target: torch.Tensor | None,
                    theta: torch.Tensor,
                    queries_counter: QueriesCounter,
                    initial_lbd: float,
                    phase: OPTAttackPhase,
                    current_best: float | None,
                    lower_b: float | None = None,
                    upper_b: float | None = None,
                    tol: float = DEFAULT_LINE_SEARCH_TOL) -> tuple[float, QueriesCounter, None]:
        distance, queries_counter = opt_line_search(self, model, x, y, target, theta, queries_counter, initial_lbd,
                                                    phase, OPTAttackPhase.direction_probing, current_best,
                                                    self.n_searches, self.max_search_steps, self.batch_size, lower_b,
                                                    upper_b)
        return distance, queries_counter, None


def normalize(x: torch.Tensor, batch: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize x in-place
    Args:
        x (torch.Tensor): Tensor to normalize
        batch (bool): First dimension of x is batch
    """
    if batch:
        norm = x.reshape(x.size(0), -1).norm(2, 1)  # type: ignore
    else:
        norm = x.norm()
    for _ in range(x.ndim - 1):
        norm.unsqueeze_(-1)
    x /= (norm + 1e-9)
    if batch:
        norm = norm.view(x.size(0), 1)
    else:
        norm.squeeze_()
    return x, norm
