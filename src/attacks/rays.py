from typing import Callable

import numpy as np
import torch
from foolbox.distances import LpDistance
from torchvision.transforms.functional import rotate

from src.attacks.base import Bounds, DirectionAttack, SearchMode
from src.attacks.queries_counter import QueriesCounter
from src.model_wrappers import ModelWrapper


class RayS(DirectionAttack):

    def __init__(self, epsilon: float, distance: LpDistance, bounds: Bounds, discrete: bool, early_stopping: bool,
                 search: SearchMode, line_search_tol: float | None, flip_squares: bool, flip_rand_pixels: bool):
        super().__init__(epsilon, distance, bounds, discrete, line_search_tol)
        self.early_stopping = early_stopping
        self.search = search
        self.flip_squares = flip_squares
        self.flip_rand_pixels = flip_rand_pixels

        if self.discrete:
            self.epsilon = round(self.epsilon * 255)
            print(f"Making attack discrete with epsilon = {self.epsilon}")

    def attack_hard_label(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            y: torch.Tensor,
            target: torch.Tensor | None = None,
            query_limit: int = 10000) -> tuple[torch.Tensor, QueriesCounter, float, bool, dict[str, float | int]]:
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = int(np.prod(shape[1:]))

        # Init counter and variables
        queries_counter = QueriesCounter(query_limit)
        best_distance = np.inf
        sgn_vector = torch.ones_like(x)
        x_final = self.get_x_adv(x, sgn_vector, best_distance)
        block_level = 0
        block_ind = 0
        search_early_stoppings = 0

        # Set-up search function
        search_fn: Callable[[torch.Tensor, float, QueriesCounter], tuple[float, QueriesCounter, bool]]
        if self.search == SearchMode.binary:
            search_fn = lambda direction, distance, q_counter: self.binary_search(model, x, y, target, direction,
                                                                                  distance, q_counter)
        elif self.search == SearchMode.line:
            search_fn = lambda direction, distance, q_counter: self.line_search(model, x, y, target, direction,
                                                                                distance, q_counter)
        else:
            raise ValueError(f"Search method '{self.search}' not supported")

        max_block_ind = 2**block_level
        if not self.flip_squares:
            rotate_to_flip = None
        else:
            rotate_to_flip = False

        updated_queries_counter = queries_counter
        i = 0
        while True:
            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))

            # Compute which blocks to flip and flip them
            start, end = get_start_end(dim, block_ind, block_size)
            if self.flip_squares:
                assert rotate_to_flip is not None
                attempt = flip_sign_alternate(sgn_vector, shape, dim, rotate_to_flip, start, end)
                rotate_to_flip = not rotate_to_flip
            elif self.flip_rand_pixels:
                attempt = flip_random_pixels(sgn_vector, shape, dim, start, end)
            else:
                attempt = flip_sign(sgn_vector, shape, dim, start, end)

            # Compute the distance attained with this attempt direction
            d_end, updated_queries_counter, stopped_early = search_fn(attempt, best_distance, updated_queries_counter)
            if stopped_early:
                search_early_stoppings += 1

            # If direction is better update best distance and direction
            if d_end < best_distance:
                best_distance = d_end
                sgn_vector = attempt
                x_final = self.get_x_adv(x, sgn_vector, best_distance)

            # Update block flipping information
            if rotate_to_flip is None or not rotate_to_flip:
                block_ind += 1
            if block_ind == max_block_ind or end == dim:
                block_level += 1
                block_ind = 0
                max_block_ind = 2**block_level

            # Stop if the attack was successful or if we're out of queries
            if self.early_stopping and (best_distance <= self.epsilon):
                break
            if updated_queries_counter.is_out_of_queries():
                print('Out of queries')
                break

            i += 1
            if i % 10 == 0:
                print("Iter %3d d_t %.8f queries %d bad queries %d" %
                      (i + 1, best_distance, updated_queries_counter.total_queries,
                       updated_queries_counter.total_unsafe_queries))

        print(
            "Iter %3d d_t %.6f queries %d bad queries %d" %
            (i + 1, best_distance, updated_queries_counter.total_queries, updated_queries_counter.total_unsafe_queries))

        extra_results: dict[str, float | int] = {"search_early_stoppings": search_early_stoppings}

        return x_final, updated_queries_counter, best_distance, best_distance <= self.epsilon, extra_results

    def __call__(self,
                 model: ModelWrapper,
                 x: torch.Tensor,
                 label: torch.Tensor,
                 target: torch.Tensor | None = None,
                 query_limit: int = 10000) -> tuple[torch.Tensor, QueriesCounter, float, bool, dict[str, float | int]]:
        return self.attack_hard_label(model, x, label, target, query_limit)


def flip_random_pixels(sgn_vector: torch.Tensor, shape: list[int], dim: int, start: int, end: int):
    attempt = sgn_vector.clone().view(shape[0], dim)
    flipping_signs = torch.ones_like(attempt)
    flipping_signs[:, start:end] *= -1.
    permuted_indices = torch.randperm(dim, dtype=torch.long, device=attempt.device)
    permuted_flipping_signs = flipping_signs[:, permuted_indices]
    attempt *= permuted_flipping_signs
    return attempt.view(shape)


def flip_sign_alternate(sgn_vector: torch.Tensor, shape: list[int], dim: int, to_rotate: bool, start: int,
                        end: int) -> torch.Tensor:
    if to_rotate:
        attempt = rotate(sgn_vector.clone(), 90).view(shape[0], dim)
    else:
        attempt = sgn_vector.clone().view(shape[0], dim)
    attempt[:, start:end] *= -1.
    attempt = attempt.view(shape)
    if to_rotate:
        attempt = rotate(attempt, 270)
    return attempt


def flip_sign(sgn_vector: torch.Tensor, shape: list[int], dim: int, start: int, end: int) -> torch.Tensor:
    attempt = sgn_vector.clone().view(shape[0], dim)
    attempt[:, start:end] *= -1.
    attempt = attempt.view(shape)
    return attempt


def get_start_end(dim: int, block_ind: int, block_size: int) -> tuple[int, int]:
    return block_ind * block_size, min(dim, (block_ind + 1) * block_size)
