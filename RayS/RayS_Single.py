from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from numpy import testing as npt

class RayS(object):

    def __init__(self,
                 model,
                 order=np.inf,
                 epsilon=0.3,
                 early_stopping=True,
                 search="binary",
                 line_search_tol=None,
                 conf_early_stopping: float | None = None,
                 flip_squares: bool = False):
        self.model = model
        self.order = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}
        self.early_stopping = early_stopping
        self.search = search
        self.line_search_tol = line_search_tol
        self.conf_early_stopping = conf_early_stopping
        self.flip_squares = flip_squares

    def get_xadv(self, x, v, d, lb=0., rb=1.):
        out = x + d * v
        return torch.clamp(out, lb, rb)

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
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
        dist = torch.tensor(np.inf)
        block_level = 0
        block_ind = 0
        
        if self.flip_squares:
            max_block_ind = (2 ** block_level) ** 2
        else:
            max_block_ind = 2 ** block_level

        for i in range(query_limit):
            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = self.get_start_end(dim, block_ind, block_size)

            if not self.flip_squares:
                attempt = self.flip_sign(shape, dim, start, end)
            elif block_level == 0:
                attempt = self.flip_sign(shape, dim, start, end)
            else:
                attempt = self.flip_square(block_level, block_ind)

            if self.search == "binary":
                self.binary_search(x, y, target, attempt)
            elif self.search == "line":
                self.line_search(x, y, target, attempt)

            block_ind += 1
            if block_ind == max_block_ind or end == dim:
                block_level += 1
                block_ind = 0
                if self.flip_squares:
                    max_block_ind = (2 ** block_level) ** 2
                else:
                    max_block_ind = 2 ** block_level

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

    def flip_square(self, block_level: int, block_ind: int):
        assert self.sgn_t is not None
        num_squares_per_side = 2 ** block_level
        
        if self.sgn_t.shape[2] % num_squares_per_side != 0:
            num_squares_per_side = self.sgn_t.shape[2]
            
        attempt = rearrange(self.sgn_t,
                            'b c (h1 h) (w1 w) -> b c (h1 w1) h w',
                            h1=num_squares_per_side,
                            w1=num_squares_per_side)
        attempt[:, :, block_ind, :, :] *= -1
        return rearrange(attempt,
                         'b c (h1 w1) h w -> b c (h1 h) (w1 w)',
                         h1=num_squares_per_side,
                         w1=num_squares_per_side)

    def search_succ(self, x, y, target):
        self.queries += 1
        if target:
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

    def line_search(self, x, y, target, sgn, max_steps=200):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                self.wasted_queries += 1
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.init_lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        step_size = d_end / max_steps
        d_beginning = d_end
        for i in range(1, max_steps):
            d_end_tmp = d_beginning - step_size * i
            if not self.search_succ(self.get_xadv(x, sgn_unit, d_end_tmp), y, target):
                break
            d_end = d_end_tmp

            if self.line_search_tol is not None and 1 - (d_end / self.d_t) >= self.line_search_tol:
                break

        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False

    def binary_search(self, x, y, target, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                self.wasted_queries += 1
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.init_lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target):
                d_end = d_mid
            else:
                d_start = d_mid
        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False

    def __call__(self, data, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, seed=seed, query_limit=query_limit)


def test_flip_square():
    model = nn.Identity()
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