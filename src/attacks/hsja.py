import math

import torch
from foolbox.distances import LpDistance, l2, linf

from src.attacks.base import Bounds, ExtraResultsDict, PerturbationAttack
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.model_wrappers import ModelWrapper
from src.utils import compute_distance


class HSJAttackPhase(AttackPhase):
    gradient_estimation = "gradient_estimation"
    binary_search = "binary_search"
    step_size_search = "step_size_search"
    initialization = "initialization"


class HSJA(PerturbationAttack):
    def __init__(self,
                 epsilon: float | None,
                 distance: LpDistance,
                 bounds: Bounds,
                 discrete: bool,
                 limit_unsafe_queries: bool,
                 num_iterations: int,
                 gamma: float = 1.0,
                 fixed_delta: float | None = None,
                 stepsize_search: str = "geometric_progression",
                 max_num_evals: int = int(1e4),
                 init_num_evals: int = 100):
        super().__init__(epsilon, distance, bounds, discrete, limit_unsafe_queries)
        self.init_num_evals = init_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.fixed_delta = fixed_delta

    def __call__(self,
                 model: ModelWrapper,
                 x: torch.Tensor,
                 label: torch.Tensor,
                 target: torch.Tensor | None = None,
                 query_limit: int = 10_000) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        return self.hsja(model, x, label, self.bounds.upper, self.bounds.lower, self.distance, self.num_iterations,
                         self.gamma, self.fixed_delta, target, None, query_limit, self.stepsize_search,
                         self.max_num_evals, self.init_num_evals)

    def hsja(self,
             model: ModelWrapper,
             sample: torch.Tensor,
             original_label: torch.Tensor,
             clip_max: float = 1,
             clip_min: float = 0,
             distance: LpDistance = l2,
             num_iterations: int = 40,
             gamma: float = 1.0,
             fixed_delta: float | None = None,
             target_label: torch.Tensor | None = None,
             target_image: torch.Tensor | None = None,
             max_queries: int | None = None,
             stepsize_search: str = 'geometric_progression',
             max_num_evals: int = int(1e4),
             init_num_evals: int = 100,
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
        }

        # Set binary search threshold.
        if params['distance'] == l2:
            params['theta'] = params['gamma'] / (math.sqrt(params['d']) * params['d'])
        else:
            params['theta'] = params['gamma'] / (params['d']**2)
        params['theta'] = torch.tensor([params['theta']], device=sample.device)

        queries_counter = QueriesCounter(max_queries, limit_unsafe_queries=self.limit_unsafe_queries)

        # Initialize.
        perturbed, queries_counter = self.initialize(model, sample, params, queries_counter)

        # Project the initialization to the boundary.
        perturbed, dist_post_update, queries_counter = self.binary_search_batch(sample, torch.unsqueeze(perturbed, 0),
                                                                                model, params, queries_counter)
        dist = compute_distance(perturbed, sample.unsqueeze(0), distance).item()

        for j in range(params['num_iterations']):
            params['cur_iter'] = j + 1

            # Choose delta.
            delta = self.select_delta(params, dist_post_update)

            # Choose number of evaluations.
            num_evals = int(params['init_num_evals'] * math.sqrt(j + 1))
            num_evals = int(min([num_evals, params['max_num_evals']]))

            # approximate gradient.
            gradf, queries_counter = self.approximate_gradient(model, perturbed, num_evals, delta, params,
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
                perturbed, dist_post_update, queries_counter = self.binary_search_batch(
                    sample, perturbed[None], model, params, queries_counter)
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
                    perturbed, dist_post_update, queries_counter = self.binary_search_batch(
                        sample, perturbeds[idx_perturbed], model, params, queries_counter)

            # compute new distance.
            dist = compute_distance(perturbed, sample.unsqueeze(0), distance).item()
            if verbose:
                print('iteration: {:d}, {:f} distance {:.4f}, total queries {:.4f} total unsafe queries {:.4f}'.format(
                    j + 1, distance.p, dist, queries_counter.total_queries, queries_counter.total_unsafe_queries))

            if queries_counter.is_out_of_queries():
                print("Out of queries")
                break

        return perturbed, queries_counter, dist, not queries_counter.is_out_of_queries(), {}

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

    def approximate_gradient(self, model: ModelWrapper, sample: torch.Tensor, num_evals: int, delta, params,
                             queries_counter: QueriesCounter,
                             original_sample: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter]:
        clip_max, clip_min = params['clip_max'], params['clip_min']

        # Generate random vectors.
        noise_shape = [num_evals] + list(params['shape'])
        if params['distance'] == l2:
            rv = torch.randn(*noise_shape, device=sample.device)
        elif params['distance'] == linf:
            rv = torch.empty(*noise_shape, device=sample.device).uniform_(-1, 1)
        else:
            raise ValueError(f'Unknown constraint {params["constraint"]}.')

        rv = rv / torch.sqrt(torch.sum(rv**2, dim=(1, 2, 3), keepdim=True))
        perturbed = sample + delta * rv
        perturbed = self.clip_image(perturbed, clip_min, clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions, updated_queries_counter = self.decision_function(model, perturbed, params, queries_counter,
                                                                    HSJAttackPhase.gradient_estimation, original_sample)
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

    def binary_search_batch(self, original_image: torch.Tensor, perturbed_images: torch.Tensor, model: ModelWrapper,
                            params, queries_counter: QueriesCounter) -> tuple[torch.Tensor, float, QueriesCounter]:
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
            decisions, queries_counter = self.decision_function(model, mid_images, params, queries_counter,
                                                                HSJAttackPhase.binary_search, original_image)
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

    def initialize(self, model: ModelWrapper, sample: torch.Tensor, params,
                   queries_counter: QueriesCounter) -> tuple[torch.Tensor, QueriesCounter]:
        """
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0

        if params['target_image'] is None:
            # Find a misclassified random noise.
            while True:
                random_noise = torch.empty(*params['shape'],
                                           device=sample.device).uniform_(params['clip_min'], params['clip_max'])
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
            # TODO(@edoardo): make this a line search eventually
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success, queries_counter = self.decision_function(model, blended[None], params, queries_counter,
                                                                  HSJAttackPhase.initialization, sample)
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
            success, updated_phi_queries_counter = self.decision_function(model, new[None], params, phi_queries_counter,
                                                                          HSJAttackPhase.step_size_search,
                                                                          original_sample)
            return success, updated_phi_queries_counter

        while not (iter_result := phi(epsilon, queries_counter))[0]:
            # TODO (@edoardo): make this a line search eventually
            _, queries_counter = iter_result
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
