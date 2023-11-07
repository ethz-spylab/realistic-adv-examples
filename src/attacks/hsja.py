from enum import Enum
import itertools
import math

import torch
from foolbox.distances import LpDistance, l2, linf

from src.attacks.base import Bounds, ExtraResultsDict, PerturbationAttack, SearchMode
from src.attacks.opt import normalize
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.attacks.utils import opt_binary_search, opt_line_search
from src.model_wrappers import ModelWrapper
from src.utils import compute_distance

MAX_BATCH_SIZE = 100
OVERSHOOT_VALUE = 1.03


class HSJAttackPhase(AttackPhase):
    gradient_estimation = "gradient_estimation"
    boundary_projection = "boundary_projection"
    binary_search = "binary_search"  # deprecated, left for compatibiilitty
    step_size_search = "step_size_search"
    initialization = "initialization"
    initialization_search = "initialization_search"
    direction_probing = "direction_probing"
    gradient_estimation_search_start = "gradient_estimation_search_start"


class GradientEstimationMode(Enum):
    hsja = "hsja"
    opt = "opt"
    sign_opt = "sign_opt"


class HSJA(PerturbationAttack):
    OPT_GRAD_EVALS = 10
    GRAD_BOUNDARY_DISTANCE = 1e-3
    GRAD_EST_SEARCH_LOWER_BOUND = 1 - (OVERSHOOT_VALUE - 1)
    GRAD_EST_SEARCH_UPPER_BOUND = OVERSHOOT_VALUE

    def __init__(self,
                 epsilon: float | None,
                 distance: LpDistance,
                 bounds: Bounds,
                 discrete: bool,
                 queries_limit: int | None,
                 unsafe_queries_limit: int | None,
                 num_iterations: int,
                 gamma: float = 1.0,
                 gradient_estimation_mode: GradientEstimationMode = GradientEstimationMode.hsja,
                 search: SearchMode = SearchMode.binary,
                 grad_batch_size: int = 1024,
                 fixed_delta: float | None = None,
                 stepsize_search: str = "geometric_progression",
                 max_num_evals: int = int(1e4),
                 init_num_evals: int = 100,
                 max_opt_search_steps: int = 10_000,
                 n_searches: int = 2,
                 bias_coef: float = 0.0,
                 lower_bad_query_bound: int = 10,
                 upper_bad_query_bound: int = 20,
                 bias_coef_change_rate: float = 1e-1):
        super().__init__(epsilon, distance, bounds, discrete, queries_limit, unsafe_queries_limit)
        self.init_num_evals = init_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.fixed_delta = fixed_delta
        self.gradient_estimation_mode = gradient_estimation_mode
        self.grad_batch_size = grad_batch_size
        self.max_opt_search_steps = max_opt_search_steps
        self.search = search
        self.n_searches = n_searches
        self.bias_coef = bias_coef
        self.lower_bad_query_bound = lower_bad_query_bound
        self.upper_bad_query_bound = upper_bad_query_bound
        self.bias_coef_change_rate = bias_coef_change_rate

    def __call__(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            label: torch.Tensor,
            target: torch.Tensor | None = None) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        return self.hsja(model, x, label, self.bounds.upper, self.bounds.lower, self.distance, self.num_iterations,
                         self.gamma, self.fixed_delta, target, None, self.stepsize_search, self.max_num_evals,
                         self.init_num_evals, self.bias_coef, self.lower_bad_query_bound, self.upper_bad_query_bound,
                         self.bias_coef_change_rate)

    def hsja(self,
             model: ModelWrapper,
             sample: torch.Tensor,
             original_label: torch.Tensor,
             clip_max: float = 1,
             clip_min: float = 0,
             distance: LpDistance = l2,
             num_iterations: int | None = 40,
             gamma: float = 1.0,
             fixed_delta: float | None = None,
             target_label: torch.Tensor | None = None,
             target_image: torch.Tensor | None = None,
             stepsize_search: str = 'geometric_progression',
             max_num_evals: int = int(1e4),
             init_num_evals: int = 100,
             bias_coef: float = 0.0,
             lower_bad_query_bound: int = 10,
             upper_bad_query_bound: int = 20,
             bias_coef_change_rate: float = 1e-1,
             verbose: bool = True) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        """
        Main algorithm for HopSkipJumpAttack.

            Inputs:
            model: the object that has predict method.
            predict outputs probability scores.
            clip_max: upper bound of the image.
            clip_min: lower bound of the image.
            constraint: choose between [l2, linf].
            num_iterations: number of iterations.
            gamma: used to set binary search threshold theta. The binary search
            threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for
            linf attack.
            target_label: integer or None for nontargeted attack.
            target_image: an array with the same size as sample, or None.
            stepsize_search: choose between 'geometric_progression', 'grid_search'.
            max_num_evals: maximum number of evaluations for estimating gradient (for each iteration).
            This is not the total number of model evaluations for the entire algorithm, you need to
            set a counter of model evaluations by yourself to get that. To increase the total number
            of model evaluations, set a larger num_iterations.
            init_num_evals: initial number of evaluations for estimating gradient.
            bias_coef: used to move boundary point further before gradient estimation.
            lower_bad_query_bound: desired lower bound on the number of bad queries in the gradient esitmation phase.
            upper_bad_query_bound: desired upper bound on the number of bad queries in the gradient esitmation phase.
            bias_coef_change_rate: used to change the bias coefficient adaptively.

            Output:
            perturbed image.
            """
        # The attack works on a single image.
        sample = sample[0]

        params = {
            'clip_max': clip_max,
            'clip_min': clip_min,
            'shape': sample.shape,
            'original_label': original_label,
            'target_label': target_label,
            'target_image': target_image,
            'distance': distance,
            'num_iterations': num_iterations,
            'gamma': gamma,
            'd': int(math.prod(sample.shape)),
            'stepsize_search': stepsize_search,
            'max_num_evals': max_num_evals,
            'init_num_evals': init_num_evals,
            'verbose': verbose,
            'fixed_delta': fixed_delta,
            'opt_grad_evals': self.OPT_GRAD_EVALS,
            'grad_boundary_distance': self.GRAD_BOUNDARY_DISTANCE,
            'bias_coef': bias_coef,
            'lower_bad_query_bound': lower_bad_query_bound,
            'upper_bad_query_bound': upper_bad_query_bound,
            'bias_coef_change_rate': bias_coef_change_rate,
        }

        # Set binary search threshold.
        if params['distance'] == l2:
            params['theta'] = params['gamma'] / (math.sqrt(params['d']) * params['d'])
        else:
            params['theta'] = params['gamma'] / (params['d']**2)
        params['theta'] = torch.tensor([params['theta']], device=sample.device)

        queries_counter: QueriesCounter[HSJAttackPhase] = self._make_queries_counter()

        # Initialize.
        perturbed, queries_counter = self.initialize(model, sample, params, queries_counter)

        # Project the initialization to the boundary.
        if self.search == SearchMode.binary:
            perturbed, dist_post_update, queries_counter = self.binary_search_batch(sample,
                                                                                    torch.unsqueeze(perturbed, 0),
                                                                                    model, params, queries_counter)
        else:
            perturbed, dist_post_update, queries_counter = self.line_search(sample, perturbed, model, params,
                                                                            queries_counter)
        dist = compute_distance(perturbed, sample.unsqueeze(0), distance).item()

        if params['num_iterations'] is not None:
            _range = range(params['num_iterations'])
        else:
            _range = itertools.count()

        for j in _range:
            params['cur_iter'] = j + 1

            # Choose delta.
            delta = self.select_delta(params, dist_post_update)

            # Choose number of evaluations.
            num_evals = int(params['init_num_evals'] * math.sqrt(j + 1))
            num_evals = int(min([num_evals, params['max_num_evals']]))

            # approximate gradient.
            if self.gradient_estimation_mode == GradientEstimationMode.hsja:
                gradf, queries_counter = self.approximate_gradient_hsja(model, perturbed, num_evals, delta, params,
                                                                        queries_counter, sample)
            elif self.gradient_estimation_mode == GradientEstimationMode.sign_opt:
                gradf, queries_counter = self.approximate_gradient_sign_opt(model, perturbed, num_evals, delta, params,
                                                                            queries_counter, sample)
            else:
                gradf, queries_counter = self.approximate_gradient_opt(model, perturbed, num_evals, delta, params,
                                                                       queries_counter, sample)

            if params['distance'] == linf:
                update = torch.sign(gradf)
            else:
                update = gradf

            # search step size.
            if params['stepsize_search'] == 'geometric_progression':
                # find step size.
                epsilon, queries_counter = self.geometric_progression_for_stepsize(perturbed, update, dist, model,
                                                                                   params, queries_counter, sample)

                # Update the sample.
                perturbed = self.clip_image(perturbed + epsilon * update, clip_min, clip_max)

                # Binary search to return to the boundary.
                if self.search == SearchMode.binary:
                    perturbed, dist_post_update, queries_counter = self.binary_search_batch(
                        sample, perturbed[None], model, params, queries_counter)
                else:
                    perturbed, dist_post_update, queries_counter = self.line_search(sample, perturbed, model, params,
                                                                                    queries_counter)
            elif params['stepsize_search'] == 'grid_search':
                # Grid search for stepsize.
                epsilons = torch.logspace(-4, 0, steps=20) * dist
                epsilons_shape = [20] + len(params['shape']) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = self.clip_image(perturbeds, params['clip_min'], params['clip_max'])
                idx_perturbed, queries_counter = self.decision_function(model, perturbeds, params, queries_counter,
                                                                        HSJAttackPhase.step_size_search, sample)

                if torch.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    if self.search == SearchMode.binary:
                        perturbed, dist_post_update, queries_counter = self.binary_search_batch(
                            sample, perturbeds[idx_perturbed], model, params, queries_counter)
                    else:
                        perturbed, dist_post_update, queries_counter = self.line_search(
                            sample, perturbeds[idx_perturbed], model, params, queries_counter)

            # compute new distance.
            dist = compute_distance(perturbed, sample.unsqueeze(0), distance).item()
            if verbose:
                print(
                    'iteration: {:d}, l{:.0f} distance {:.4f}, total queries {:.4f} total unsafe queries {:.4f}'.format(
                        j + 1, distance.p, dist, queries_counter.total_queries, queries_counter.total_unsafe_queries))

            if queries_counter.is_out_of_queries():
                print("Out of queries")
                break

        return perturbed, queries_counter, dist, True, {}

    def decision_function(self, model: ModelWrapper, images: torch.Tensor, params, queries_counter: QueriesCounter,
                          attack_phase: HSJAttackPhase,
                          original_images: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter]:
        """
        Decision function output 1 on the desired side of the boundary,
        0 otherwise.
        """
        images = self.clip_image(images, params['clip_min'], params['clip_max'])
        label = model.predict_label(images)
        if params['target_label'] is None:
            success = label != params['original_label']
        else:
            success = label == params['target_label']
        distance = self.distance(images, original_images)

        return success, queries_counter.increase(attack_phase, safe=success, distance=distance)  # type: ignore

    def clip_image(self, image: torch.Tensor, clip_min: float | torch.Tensor,
                   clip_max: float | torch.Tensor) -> torch.Tensor:
        # Clip an image, or an image batch, with upper and lower threshold.
        return torch.clamp(image, clip_min, clip_max)  # type: ignore

    def approximate_gradient_hsja(self, model: ModelWrapper, sample: torch.Tensor, num_evals: int, delta, params,
                                  queries_counter: QueriesCounter,
                                  original_sample: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter]:
        clip_max, clip_min = params['clip_max'], params['clip_min']

        # Generate random vectors.
        noise_shape = [num_evals] + list(params['shape'])
        if params['distance'] == l2:
            rv = torch.randn(*noise_shape, device=sample.device)
        elif params['distance'] == linf:
            rv = torch.empty(*noise_shape, device=sample.device).uniform_(-1, 1)  # type: ignore
        else:
            raise ValueError(f'Unknown constraint {params["constraint"]}.')

        rv = rv / torch.sqrt(torch.sum(rv**2, dim=(1, 2, 3), keepdim=True))

        # Move the current boundary point further to limit the number of bad queries.
        bias = (sample - original_sample) / torch.norm(sample - original_sample) * delta * params['bias_coef']
        biased_sample = sample + bias

        # Estimate gradient similar to vanilla HSJA
        perturbed = biased_sample + delta * rv
        perturbed = self.clip_image(perturbed, clip_min, clip_max)
        rv = (perturbed - biased_sample) / delta

        # query the model.
        decisions, updated_queries_counter = self.decision_function(model, perturbed, params, queries_counter,
                                                                    HSJAttackPhase.gradient_estimation, original_sample)
        
        # As the algorithm goes by, in most cases, the bias coefficient should decrease \
        # to keep the number of bad queries sufficient for gradient estimation
        
        bad_queries_num = len(decisions) - decisions.sum()
        
        if bad_queries_num < params['lower_bad_query_bound']:
            params['bias_coef'] = (1 - params['bias_coef_change_rate']) * params['bias_coef']
        if bad_queries_num > params['upper_bad_query_bound']:
            params['bias_coef'] = (1 + params['bias_coef_change_rate']) * params['bias_coef']
        
        if params['verbose']:
            print('Gradient estimation results: number of total queries {:.0f}, nubmer of bad queries {:.0f}, bias coefficient {:.4f}'.format(
                        len(decisions), bad_queries_num, params['bias_coef']))
            
        # Use importance sampling for a better estimation. (This part is similar to vanilla HSJA)
        decision_shape = [len(decisions)] + [1] * len(params['shape'])
        fval = 2 * decisions.to(torch.float).reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if torch.mean(fval) == 1.0:  # label changes.
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval) == -1.0:  # label not change.
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)

        # Get the gradient direction.
        gradf = gradf / torch.linalg.norm(gradf, dim=None)

        return gradf, updated_queries_counter

    def approximate_gradient_opt(self, model: ModelWrapper, x_bd: torch.Tensor, num_evals: int, delta, params,
                                 queries_counter: QueriesCounter,
                                 x: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter]:
        theta, initial_lbd = normalize(x_bd - x)
        delta /= initial_lbd
        # SignOPT does 200 evaluations, and OPT does 10, so we get some sort of equivalent number
        sign_opt_queries = 200
        opt_queries = 10
        num_evals = max(num_evals // sign_opt_queries, 1) * opt_queries

        u = torch.randn((num_evals, ) + x.shape, device=x.device, dtype=x.dtype)
        u, _ = normalize(u, batch=True)
        new_thetas = theta + delta * u
        new_thetas, _ = normalize(new_thetas, batch=True)

        distances = torch.zeros(num_evals, device=x.device, dtype=x.dtype)
        for j in range(num_evals):
            queries_counter = queries_counter.increase(
                HSJAttackPhase.gradient_estimation_search_start,
                torch.tensor([True]),
                torch.tensor([123123]),
            )
            if self.search == SearchMode.binary:
                g1, queries_counter = self.opt_binary_search(model, x, params['original_label'],
                                                             params['target_label'], new_thetas[j], queries_counter,
                                                             initial_lbd.item(), HSJAttackPhase.gradient_estimation,
                                                             params['theta'])
            else:
                g1, queries_counter = self.opt_line_search(model, x, params['original_label'],
                                                           params['target_label'], new_thetas[j], queries_counter,
                                                           initial_lbd.item(), HSJAttackPhase.gradient_estimation)
            distances[j] = g1

        rv = u
        fval = distances
        # Baseline subtraction (when fval differs)
        if torch.all(distances < initial_lbd):  # label changes.
            gradf = torch.mean(rv, dim=0)
        elif torch.all(distances >= initial_lbd):  # label not change.
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval.reshape(-1, 1, 1, 1) * rv, dim=0)

        gradf = gradf / torch.linalg.norm(gradf, dim=None)

        return -gradf, queries_counter

    def approximate_gradient_sign_opt(self, model: ModelWrapper, x_bd: torch.Tensor, num_evals: int, delta, params,
                                      queries_counter: QueriesCounter,
                                      x: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter]:
        """
        Evaluate the sign of gradient by formula
        sign(g) = 1/Q [ \\sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """

        theta, initial_lbd = normalize(x_bd - x)
        x = x.unsqueeze(0)
        delta /= initial_lbd

        u = torch.randn((num_evals, ) + theta.shape, dtype=theta.dtype, device=x.device)
        u, _ = normalize(u, batch=True)

        sign_v = torch.ones((num_evals, 1, 1, 1), device=x.device)
        new_theta: torch.Tensor = theta + delta * u  # type: ignore
        new_theta, _ = normalize(new_theta, batch=True)
        x_ = self.get_x_adv(x, new_theta * initial_lbd)
        success, queries_counter = self.decision_function(model, x_, params, queries_counter,
                                                          HSJAttackPhase.gradient_estimation, x)
        sign_v[torch.logical_not(success)] = -1.

        rv = u
        fval = sign_v
        # Baseline subtraction (when fval differs)
        if torch.mean(fval) == 1.0:  # label changes.
            print("All samples are misclassified.")
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval) == -1.0:  # label not change.
            print("No sample is misclassified.")
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)

        gradf = gradf / torch.linalg.norm(gradf, dim=None)

        return gradf, queries_counter

    def project(self, original_image: torch.Tensor, perturbed_images: torch.Tensor, alphas: torch.Tensor,
                params) -> torch.Tensor:
        alphas_shape = [len(alphas)] + [1] * len(params['shape'])
        alphas = alphas.reshape(alphas_shape)
        if params['distance'] == l2:
            return (1 - alphas) * original_image + alphas * perturbed_images
        elif params['distance'] == linf:
            out_images = self.clip_image(perturbed_images, original_image - alphas, original_image + alphas)
            return out_images
        else:
            raise ValueError(f'Unknown constraint {params["constraint"]}.')

    def opt_binary_search(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None,
                          theta: torch.Tensor, queries_counter: QueriesCounter, initial_lbd: float,
                          phase: HSJAttackPhase, tol: float) -> tuple[float, QueriesCounter]:
        distance, queries_counter, _ = opt_binary_search(self, model, x, y, target, theta, queries_counter, initial_lbd,
                                                         phase, phase, tol)
        return distance, queries_counter

    def opt_line_search(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None,
                        theta: torch.Tensor, queries_counter: QueriesCounter, initial_lbd: float,
                        phase: HSJAttackPhase) -> tuple[float, QueriesCounter]:
        distance, queries_counter = opt_line_search(self, model, x, y, target, theta, queries_counter, initial_lbd,
                                                    phase, HSJAttackPhase.direction_probing, None, self.n_searches,
                                                    self.max_opt_search_steps, self.grad_batch_size,
                                                    self.GRAD_EST_SEARCH_LOWER_BOUND, self.GRAD_EST_SEARCH_UPPER_BOUND)
        return distance, queries_counter

    def binary_search_batch(
            self,
            original_image: torch.Tensor,
            perturbed_images: torch.Tensor,
            model: ModelWrapper,
            params,
            queries_counter: QueriesCounter,
            phase: HSJAttackPhase = HSJAttackPhase.boundary_projection) -> tuple[torch.Tensor, float, QueriesCounter]:
        """ Binary search to approach the boundary."""

        # Compute distance between each of perturbed image and original image.
        dists_post_update = compute_distance(original_image.unsqueeze(0), perturbed_images, params['distance'])

        highs: torch.Tensor
        lows: torch.Tensor
        # Choose upper thresholds in binary searchs based on constraint.
        if params['distance'] == linf:
            highs = dists_post_update
            # Stopping criteria.
            thresholds = torch.minimum(dists_post_update * params['theta'], params['theta'])
        else:
            highs = torch.ones(len(perturbed_images), device=original_image.device)
            thresholds = params['theta']

        lows = torch.zeros(len(perturbed_images), device=original_image.device)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs

        # Call recursive function.
        while torch.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids, params)

            # Update highs and lows based on model decisions.
            decisions, queries_counter = self.decision_function(model, mid_images, params, queries_counter, phase,
                                                                original_image)
            lows = torch.where(decisions == 0, mids, lows)  # type: ignore
            highs = torch.where(decisions == 1, mids, highs)  # type: ignore

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids

            if reached_numerical_precision:
                break

        out_images = self.project(original_image, perturbed_images, highs, params)

        # Compute distance of the output image to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = compute_distance(original_image.unsqueeze(0), out_images, params['distance'])
        idx = torch.argmin(dists)

        dist = dists_post_update[idx].item()
        out_image = out_images[idx]

        return out_image, dist, queries_counter

    def line_search(
            self,
            original_image: torch.Tensor,
            perturbed_images: torch.Tensor,
            model: ModelWrapper,
            params,
            queries_counter: QueriesCounter,
            phase: HSJAttackPhase = HSJAttackPhase.boundary_projection) -> tuple[torch.Tensor, float, QueriesCounter]:
        # Compute distance between each of perturbed image and original image.
        dists_post_update = compute_distance(original_image.unsqueeze(0), perturbed_images.unsqueeze(0),
                                             params['distance'])

        dists: torch.Tensor
        # Choose upper thresholds in binary searchs based on constraint.
        if params['distance'] == linf:
            dists = dists_post_update
            # Stopping criteria.
            step_size = torch.minimum(dists_post_update * params['theta'] / 2, params['theta'])
        else:
            dists = torch.ones_like(dists_post_update)
            step_size = params['theta'] / 2

        if self.n_searches == 2:
            search_max_steps = math.ceil(math.sqrt(dists_post_update / step_size))
            first_search_step_size = (dists_post_update / math.sqrt(dists_post_update / step_size)).cpu().item()
        else:
            search_max_steps = math.ceil((dists_post_update / step_size).item())
            first_search_step_size = step_size.item()

        search_batch_size = min(search_max_steps, self.grad_batch_size)

        first_search_distance, first_search_queries_counter = self._batched_line_search_body(
            model,
            original_image,
            params['original_label'],
            params['target_label'],
            perturbed_images,
            params,
            queries_counter,
            dists[0],
            phase,
            first_search_step_size,
            search_batch_size,
            # Here we count each query of the first search as equivalent to search_max_steps queries of when we do 1
            equivalent_simulated_queries=1 if self.n_searches == 1 else search_max_steps,
            # But we don't count the queries from the last batch as they will be counted in the second search
            count_last_batch_for_sim=self.n_searches == 1)

        if self.n_searches == 2:
            second_search_step_size = step_size.item()
            final_distance, second_search_queries_counter = self._batched_line_search_body(
                model,
                original_image,
                params['original_label'],
                params['target_label'],
                perturbed_images,
                params,
                first_search_queries_counter,
                first_search_distance,
                phase,
                second_search_step_size,
                search_batch_size,
                # Here each query has the same step size as if we were doing one search only
                equivalent_simulated_queries=1,
                # And we count the queries from the last batch as they are not counted in the first search
                count_last_batch_for_sim=True)
        else:
            second_search_queries_counter = first_search_queries_counter
            final_distance = first_search_distance

        out_images = self.project(original_image, perturbed_images, final_distance.unsqueeze(0), params)

        # Compute distance of the output image to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = compute_distance(original_image.unsqueeze(0), out_images, params['distance'])
        idx = torch.argmin(dists)

        dist = dists_post_update[idx].item()
        out_image = out_images[idx]

        return out_image, dist, second_search_queries_counter

    def _batched_line_search_body(self,
                                  model: ModelWrapper,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  target: torch.Tensor | None,
                                  perturbed_images: torch.Tensor,
                                  params,
                                  queries_counter: QueriesCounter,
                                  initial_distance: torch.Tensor,
                                  phase: HSJAttackPhase,
                                  step_size: float,
                                  batch_size: int = MAX_BATCH_SIZE,
                                  equivalent_simulated_queries: int = 1,
                                  count_last_batch_for_sim: bool = False) -> tuple[torch.Tensor, QueriesCounter]:
        success = torch.tensor([True])
        batch_idx = 0
        distances_inner_shape = tuple([1] * (len(x.shape) - 1))
        previous_last_distance = torch.tensor([initial_distance], device=initial_distance.device)
        distances = torch.tensor([initial_distance], device=initial_distance.device)

        while success.all():
            # Update the last distance (in case the whole next batch is unsafe) and the index
            previous_last_distance = distances[-1]
            # Get steps bounds based on the batch index
            start = batch_idx * batch_size
            if batch_idx == 0:
                start = 1
            end = (batch_idx + 1) * batch_size
            # Compute the steps to take
            steps_sizes = torch.arange(start, end, device=x.device) * step_size
            # Subtract the steps from the original distance
            distances = (initial_distance - steps_sizes).reshape(-1, *distances_inner_shape)
            # Compute advex and query the model
            batch = self.project(x, perturbed_images, distances, params)
            batch = self.clip_image(batch, params['clip_min'], params['clip_max'])
            success, queries_counter = self.is_correct_boundary_side_batched(model, batch, y, target, queries_counter,
                                                                             phase, x, equivalent_simulated_queries,
                                                                             count_last_batch_for_sim, batch_idx == 0)
            batch_idx += 1

        # We get the index of the first unsafe query
        unsafe_query_idx = torch.argmin(success.to(torch.int))
        if unsafe_query_idx == 0:
            # If no query was safe in the latest batch, then we return the last lbd from the previous batch
            distance = previous_last_distance
        else:
            distance = distances[unsafe_query_idx - 1]

        perturbed = self.project(x, perturbed_images, distance.unsqueeze(0), params)
        success, _ = self.decision_function(model, perturbed, params, queries_counter, phase, x)
        i = 0
        while not success.all():
            distance += step_size
            out_images = self.project(x, perturbed_images, distance.unsqueeze(0), params)
            success, _ = self.decision_function(model, out_images, params, queries_counter, phase, x)
            i += 1
        if i > 0:
            print(f"Precision issues with second search, increasing distance by step size {i} times")

        return distance, queries_counter

    def initialize(self, model: ModelWrapper, sample: torch.Tensor, params,
                   queries_counter: QueriesCounter) -> tuple[torch.Tensor, QueriesCounter]:
        """
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_evals = 0

        if params['target_image'] is None:
            # Find a misclassified random noise.
            while True:
                random_noise = torch.empty(*params['shape'],
                                           device=sample.device).uniform_(params['clip_min'],
                                                                          params['clip_max'])  # type: ignore
                success_array, queries_counter = self.decision_function(model, random_noise[None], params,
                                                                        queries_counter, HSJAttackPhase.initialization,
                                                                        sample)
                success = success_array[0]
                num_evals += 1
                if success:
                    break
                assert num_evals < 1e4, "Initialization failed! "
                "Use a misclassified image as `target_image`"

            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success, queries_counter = self.decision_function(model, blended[None], params, queries_counter,
                                                                  HSJAttackPhase.initialization_search, sample)
                if success:
                    high = mid
                else:
                    low = mid

            initialization = (1 - high) * sample + high * random_noise

        else:
            initialization = params['target_image']

        return initialization, queries_counter

    def geometric_progression_for_stepsize(self, x: torch.Tensor, update: torch.Tensor, dist: float,
                                           model: ModelWrapper, params, queries_counter: QueriesCounter,
                                           original_sample: torch.Tensor) -> tuple[float, QueriesCounter]:
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        epsilon = dist / math.sqrt(params['cur_iter'])

        def phi(eps: float, phi_queries_counter: QueriesCounter) -> tuple[torch.Tensor, QueriesCounter]:
            new: torch.Tensor = x + eps * update  # type: ignore
            success_, updated_phi_queries_counter = self.decision_function(model, new[None], params,
                                                                           phi_queries_counter,
                                                                           HSJAttackPhase.step_size_search,
                                                                           original_sample)
            return success_, updated_phi_queries_counter

        while True:
            success, queries_counter = phi(epsilon, queries_counter)
            if success:
                break
            # TODO (@edoardo): make this a line search eventually
            epsilon /= 2.0

        return epsilon, queries_counter

    def select_delta(self, params, dist_post_update: float) -> float:
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.

        """
        if params['fixed_delta'] is not None:
            return params['fixed_delta']

        if params['cur_iter'] == 1:
            delta = 0.1 * (params['clip_max'] - params['clip_min'])
        else:
            if params['distance'] == l2:
                delta = math.sqrt(params['d']) * params['theta'] * dist_post_update
            elif params['distance'] == linf:
                delta = params['d'] * params['theta'] * dist_post_update
            else:
                raise ValueError(f"Unknown constraint {params['distance']}")

        return delta