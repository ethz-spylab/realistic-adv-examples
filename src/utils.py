import torch
from foolbox.distances import l2, linf, LpDistance


def compute_distance(x_ori: torch.Tensor, x_pert: torch.Tensor, distance: LpDistance = l2) -> float:
    # Compute the distance between two images.
    return distance(x_ori, x_pert).item()
