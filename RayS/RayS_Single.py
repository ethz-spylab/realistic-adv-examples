import math

import numpy as np
import torch
from torchvision.transforms.functional import rotate

from model_wrappers.general_model import ModelWrapper


class RayS:

    def __init__(self,
                 model: ModelWrapper,
                 order: float = np.inf,
                 epsilon: float = 0.3,
                 early_stopping: bool = True,
                 search: str = "binary",
                 line_search_tol: float | None = None,
                 conf_early_stopping: float | None = None,
                 flip_squares: bool = False,
                 flip_rand_pixels: bool = False,
                 discrete_attack: bool = False):
        self.model = model
        self.order = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = np.inf
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}
        self.early_stopping = early_stopping
        self.search = search
        self.line_search_tol = line_search_tol
        self.conf_early_stopping = conf_early_stopping
        self.flip_squares = flip_squares
        self.flip_rand_pixels = flip_rand_pixels
        self.n_early_stopping = 0
        self.discrete_attack = discrete_attack
        
        if self.discrete_attack:
            self.epsilon = round(self.epsilon * 255)
            print(f"Making attack discrete with epsilon = {self.epsilon}")

    def get_xadv(self, x, v, d, lb=0., rb=1.):
        if self.discrete_attack:
            d = d / 255
        out = x + d * v
        return torch.clamp(out, lb, rb)

    def attack_hard_label(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          target: torch.Tensor | None = None,
                          query_limit: int = 10000,
                          seed: int | None = None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = int(np.prod(shape[1:]))
        if seed is not None:
            np.random.seed(seed)

        self.queries = 0
        self.bad_queries = 0
        self.wasted_queries = 0
        self.d_t = np.inf
        dist = np.inf
        self.sgn_t = torch.ones_like(x)
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        block_level = 0
        block_ind = 0
        self.n_early_stopping = 0

        if self.search == "binary":
            search_fn = lambda attempt: self.binary_search(x, y, target, attempt)
        elif self.search == "line":
            search_fn = lambda attempt: self.line_search(x, y, target, attempt)
        else:
            raise ValueError(f"Search method '{self.search}' not supported")

        max_block_ind = 2**block_level
        if not self.flip_squares:
            rotate_to_flip = None
        else:
            rotate_to_flip = False

        for i in range(query_limit):
            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))

            start, end = self.get_start_end(dim, block_ind, block_size)
            if self.flip_squares:
                assert rotate_to_flip is not None
                attempt = self.flip_sign_alternate(shape, dim, rotate_to_flip, start, end)
                rotate_to_flip = not rotate_to_flip
            elif self.flip_rand_pixels:
                attempt = self.flip_random_pixels(shape, dim, start, end)
            else:
                attempt = self.flip_sign(shape, dim, start, end)

            d_end = search_fn(attempt)
            if d_end < self.d_t:
                self.d_t = d_end
                self.sgn_t = attempt
                self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)

            if rotate_to_flip is None or not rotate_to_flip:
                block_ind += 1
            if block_ind == max_block_ind or end == dim:
                block_level += 1
                block_ind = 0
                max_block_ind = 2**block_level

            dist = self.d_t
            if self.early_stopping and (dist <= self.epsilon):
                break

            if self.queries >= query_limit:
                print('out of queries')
                break

            if i % 10 == 0:
                print("Iter %3d d_t %.8f dist %.8f queries %d bad queries %d" %
                      (i + 1, self.d_t, dist, self.queries, self.bad_queries))

        print("Iter %3d d_t %.6f dist %.6f queries %d bad queries %d" %
              (i + 1, self.d_t, dist, self.queries, self.bad_queries))
        return self.x_final, self.queries, self.bad_queries, self.wasted_queries, dist, float(dist <= self.epsilon)

    def get_start_end(self, dim: int, block_ind: int, block_size: int) -> tuple[int, int]:
        return block_ind * block_size, min(dim, (block_ind + 1) * block_size)

    def flip_sign(self, shape: list[int], dim: int, start: int, end: int) -> torch.Tensor:
        attempt = self.sgn_t.clone().view(shape[0], dim)
        attempt[:, start:end] *= -1.
        attempt = attempt.view(shape)
        return attempt

    def flip_sign_alternate(self, shape: list[int], dim: int, to_rotate: bool, start: int, end: int) -> torch.Tensor:
        if to_rotate:
            attempt = rotate(self.sgn_t.clone(), 90).view(shape[0], dim)
        else:
            attempt = self.sgn_t.clone().view(shape[0], dim)
        attempt[:, start:end] *= -1.
        attempt = attempt.view(shape)
        if to_rotate:
            attempt = rotate(attempt, 270)
        return attempt

    def flip_random_pixels(self, shape: list[int], dim: int, start: int, end: int):
        attempt = self.sgn_t.clone().view(shape[0], dim)
        flipping_signs = torch.ones_like(attempt)
        flipping_signs[:, start:end] *= -1.
        permuted_indices = torch.randperm(dim, dtype=torch.long, device=attempt.device)
        permuted_flipping_signs = flipping_signs[:, permuted_indices]
        attempt *= permuted_flipping_signs
        return attempt.view(shape)

    def search_succ(self, x, y, target):
        self.queries += 1
        if target is not None:
            success = self.model.predict_label(x) == target
        else:
            success = self.model.predict_label(x) != y
        if not success:
            self.bad_queries += 1
        return success

    def init_lin_search(self, x, y, target, sgn):
        d_end = np.inf
        start = 1
        end = self.lin_search_rad
        if self.discrete_attack:
            start *= 255
            end *= 255
        for d in range(start, end + 1):
            if self.search_succ(self.get_xadv(x, sgn, d), y, target):
                d_end = d
                break
        return d_end

    def line_search(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    target: torch.Tensor | None,
                    sgn: torch.Tensor,
                    max_steps=200) -> float:

        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn, self.d_t), y, target):
                self.wasted_queries += 1
                return np.inf
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.init_lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d
            else:
                return np.inf
        if not self.discrete_attack:
            step_size = d_end / max_steps
        else:
            step_size = math.ceil(d_end / max_steps)
        d_beginning = d_end
        for i in range(1, max_steps):
            d_end_tmp = d_beginning - step_size * i
            if not self.search_succ(self.get_xadv(x, sgn, d_end_tmp), y, target):
                break
            d_end = d_end_tmp

            if self.line_search_tol is not None and 1 - (d_end / self.d_t) >= self.line_search_tol:
                self.n_early_stopping += 1
                break

        return d_end

    def binary_search(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      target: torch.Tensor | None,
                      sgn: torch.Tensor,
                      tol: float = 1e-3) -> float:

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn, self.d_t), y, target):
                self.wasted_queries += 1
                return np.inf
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.init_lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d
            else:
                return np.inf
            
        if not self.discrete_attack:
            condition = lambda end, start: (end - start) > tol
        else:
            condition = lambda end, start: (end - start) > 1

        while condition(d_end, d_start):
            if not self.discrete_attack:
                d_mid = (d_start + d_end) / 2.0
            else:
                d_mid = math.ceil((d_start + d_end) / 2.0)
            if self.search_succ(self.get_xadv(x, sgn, d_mid), y, target):
                d_end = d_mid
            else:
                d_start = d_mid

        return d_end

    def __call__(self, data, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, seed=seed, query_limit=query_limit)
