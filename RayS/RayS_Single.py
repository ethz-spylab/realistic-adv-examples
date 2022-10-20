from einops import rearrange

import numpy as np
import torch
from numpy import testing as npt
from torch import nn
from torchvision.transforms.functional import rotate

from general_torch_model import GeneralTorchModel


class RayS:

    def __init__(self,
                 model: GeneralTorchModel,
                 order: float = np.inf,
                 epsilon: float = 0.3,
                 early_stopping: bool = True,
                 search: str = "binary",
                 line_search_tol: float | None = None,
                 conf_early_stopping: float | None = None,
                 flip_squares: bool = False):
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
        self.n_early_stopping = 0

    def get_xadv(self, x, v, d, lb=0., rb=1.):
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
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        dist = np.inf
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
            transpose_to_flip = None
        else:
            transpose_to_flip = False

        for i in range(query_limit):
            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))

            start, end = self.get_start_end(dim, block_ind, block_size)
            if not self.flip_squares:
                attempt = self.flip_sign(shape, dim, start, end)
            else:
                assert transpose_to_flip is not None
                attempt = self.flip_sign_alternate(shape, dim, transpose_to_flip, start, end)
                transpose_to_flip = not transpose_to_flip

            d_end = search_fn(attempt)
            if d_end < self.d_t:
                self.d_t = d_end
                self.sgn_t = attempt
                unit_sgn = self.sgn_t / torch.norm(self.sgn_t)
                self.x_final = self.get_xadv(x, unit_sgn, self.d_t)

            if transpose_to_flip is None or not transpose_to_flip:
                block_ind += 1
            if block_ind == max_block_ind or end == dim:
                block_level += 1
                block_ind = 0
                max_block_ind = 2**block_level

            dist = torch.norm(self.x_final - x, self.order)
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
        return self.x_final, self.queries, self.bad_queries, self.wasted_queries, dist, (dist <= self.epsilon).float()

    def get_start_end(self, dim: int, block_ind: int, block_size: int) -> tuple[int, int]:
        return block_ind * block_size, min(dim, (block_ind + 1) * block_size)

    def flip_sign(self, shape: list[int], dim: int, start: int, end: int) -> torch.Tensor:
        attempt = self.sgn_t.clone().view(shape[0], dim)
        attempt[:, start:end] *= -1.
        attempt = attempt.view(shape)
        return attempt
    
    def flip_sign_alternate(self, shape: list[int], dim: int, transpose: bool, start: int, end: int) -> torch.Tensor:
        if transpose:
            attempt = rotate(self.sgn_t.clone(), 90).view(shape[0], dim)
        else:
            attempt = self.sgn_t.clone().view(shape[0], dim)
        attempt[:, start:end] *= -1.
        attempt = attempt.view(shape)
        if transpose:
            attempt = rotate(attempt, 270)
        return attempt

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
        for d in range(1, self.lin_search_rad + 1):
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
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                self.wasted_queries += 1
                return np.inf
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.init_lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return np.inf

        step_size = d_end / max_steps
        d_beginning = d_end
        for i in range(1, max_steps):
            d_end_tmp = d_beginning - step_size * i
            if not self.search_succ(self.get_xadv(x, sgn_unit, d_end_tmp), y, target):
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
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                self.wasted_queries += 1
                return np.inf
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.init_lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return np.inf

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target):
                d_end = d_mid
            else:
                d_start = d_mid

        return d_end

    def __call__(self, data, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, seed=seed, query_limit=query_limit)


class SafeSideRayS(RayS):

    def attack_hard_label(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          target: torch.Tensor | None = None,
                          query_limit: int = 100000,
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

        x_safe, self.d_t, self.sgn_t = self.get_safe_x_specular(x, y, target)
        self.x_safe_distance = torch.norm(x - x_safe, self.order)  # type: ignore
        self.x_final = self.get_xadv(x_safe, self.sgn_t, self.d_t)
        self.best_distance = self.x_safe_distance
        print(f"Initial best distance: {self.best_distance}, queries: {self.queries}, bad_queries: {self.bad_queries}")

        dist = np.inf
        block_level = 1
        block_ind = 0
        self.n_early_stopping = 0

        if self.search == "line":
            search_fn = lambda attempt: self.line_search(x_safe, x, y, target, attempt)
        else:
            raise ValueError(f"Search method '{self.search}' for `OtherSideRayS` not supported")

        if self.flip_squares:
            max_block_ind = (2**block_level)**2
        else:
            max_block_ind = 2**block_level

        for i in range(query_limit):
            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = self.get_start_end(dim, block_ind, block_size)

            if not self.flip_squares:
                attempt = self.flip_sign(shape, dim, start, end)
            else:
                attempt = self.flip_square(block_level, block_ind)

            attempt_d = search_fn(attempt)
            attempt_unit_sgn = attempt / torch.norm(attempt)
            attempt_x_adv = self.get_xadv(x_safe, attempt_unit_sgn, attempt_d)
            attempt_distance = torch.norm(x - attempt_x_adv, self.order)  # type: ignore

            if attempt_distance < self.best_distance:
                print(f"Updating best distance from {self.best_distance} to {attempt_distance}")
                self.d_t = attempt_d
                self.sgn_t = attempt
                self.x_final = self.get_xadv(x_safe, attempt_unit_sgn, self.d_t)
                self.best_distance = attempt_distance

            block_ind += 1
            if block_ind == max_block_ind or end == dim:
                block_level += 1
                block_ind = 0
                if self.flip_squares:
                    max_block_ind = (2**block_level)**2
                else:
                    max_block_ind = 2**block_level

            dist = torch.norm(self.x_final - x, self.order)  # type: ignore
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
        return self.x_final, self.queries, self.bad_queries, self.wasted_queries, dist, (dist <= self.epsilon).float()

    def compute_distance(self, x: torch.Tensor, x_safe: torch.Tensor, sgn_t: torch.Tensor, d_t: float) -> torch.Tensor:
        unit_sgn = sgn_t / torch.norm(sgn_t)
        x_adv = self.get_xadv(x_safe, unit_sgn, d_t)
        return torch.norm(x_adv - x, self.order)  # type: ignore

    def line_search(self,
                    x_safe: torch.Tensor,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    target: torch.Tensor | None,
                    sgn: torch.Tensor,
                    step_size=1e-1) -> float:

        sgn_unit = sgn / torch.norm(sgn)
        direction_best_distance = self.x_safe_distance

        d_end = 0.
        i = 1.
        temp_d_end = direction_best_distance / 10

        while True:
            step_x_adv = self.get_xadv(x_safe, sgn_unit, temp_d_end)
            if self.hits_boundary(step_x_adv, y, target):
                print(f"Boundary hit at {i=}")
                break

            step_x_adv_distance = torch.norm(x - step_x_adv, self.order)  # type: ignore
            if step_x_adv_distance >= direction_best_distance:
                print(f"Distance stopped improving at {i=}, d={temp_d_end.item():.4f}")
                break

            if int(i) % 100 == 0:
                print("step_x_adv_distance: ", step_x_adv_distance.item())

            direction_best_distance = step_x_adv_distance
            temp_d_end += direction_best_distance / 10
            d_end = temp_d_end
            i += 1.

        return d_end

    def hits_boundary(self, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None):
        # The difference with the analogous method in the parent class is that here we check whether we hit a boundary we should not hit,
        # instead of the opposite, i.e., in the untargeted case, if we go from the safe to the unsafe region
        self.queries += 1
        if target is not None:
            hit = self.model.predict_label(x) != target
        else:
            hit = self.model.predict_label(x) == y
        if hit:
            self.bad_queries += 1
        return hit

    def hits_boundary_no_count(self, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None):
        # Use this only to generate the starting image as it does not increment the query count!
        if target is not None:
            return self.model.predict_label(x) != target
        return self.model.predict_label(x) == y

    def get_safe_x(self, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor | None, max_attempts=1000):
        # TODO: for a targeted attack it would be better to start with an image from the target class
        x_safe = torch.randn_like(x)
        while self.hits_boundary_no_count(x_safe, y, target):
            x_safe = torch.randn_like(x)
        return x_safe

    def get_safe_x_specular(self, x: torch.Tensor, y: torch.Tensor,
                            target: torch.Tensor | None) -> tuple[torch.Tensor, float, torch.Tensor]:
        if target is not None:
            raise ValueError("Specular initialization does not work for targeted attacks")
        sgn = torch.ones_like(x)
        
        # Check along the positive ones direction
        unit_sgn_plus = sgn / torch.norm(sgn)
        d_plus = torch.norm(sgn)
        step_size = d_plus / 100
        
        while not self.hits_boundary(self.get_xadv(x, unit_sgn_plus, d_plus), y, target):
            d_plus -= step_size
        
        # Check along the negative ones direction
        unit_sgn_minus = -sgn / torch.norm(sgn)
        d_minus = torch.norm(sgn)
        while not self.hits_boundary(self.get_xadv(x, unit_sgn_minus, d_minus), y, target):
            d_minus -= step_size

        # Return the direction with the closest boundary
        print(f"{d_plus=}, {d_minus=}")
        if d_plus < d_minus:
            x_safe = self.get_xadv(x, unit_sgn_plus, d_plus * 2)
            return x_safe, d_plus, -unit_sgn_plus
        
        x_safe = self.get_xadv(x, unit_sgn_minus, d_minus * 2)
        return x_safe, d_minus, -unit_sgn_minus


def test_flip_square():
    model = GeneralTorchModel(nn.Identity())
    attack = RayS(model, flip_squares=True)
    attack.sgn_t = torch.ones(1, 1, 8, 8)

    flipped = attack.flip_square(2, 0)
    exp_result = torch.ones(1, 1, 8, 8)
    exp_result[:, :, 0:2, 0:2] *= -1
    npt.assert_equal(flipped.numpy(), exp_result.numpy())

    flipped = attack.flip_square(2, 2)
    exp_result = torch.ones(1, 1, 8, 8)
    exp_result[:, :, 0:2, 4:6] *= -1
    npt.assert_equal(flipped.numpy(), exp_result.numpy())

    flipped = attack.flip_square(2, 4)
    exp_result = torch.ones(1, 1, 8, 8)
    exp_result[:, :, 2:4, 0:2] *= -1
    npt.assert_equal(flipped.numpy(), exp_result.numpy())
