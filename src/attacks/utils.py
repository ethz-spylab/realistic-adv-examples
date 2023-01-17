# From https://github.com/bethgelab/foolbox/blob/master/foolbox/devutils.py
import eagerpy as ep


def atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor:
    shape = x.shape + (1, ) * (k - x.ndim)
    return x.reshape(shape)


def flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor:
    return x.flatten(start=keep)
