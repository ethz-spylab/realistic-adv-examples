import torch
from foolbox.distances import LpDistance, l2


def compute_distance(x_ori: torch.Tensor, x_pert: torch.Tensor, distance: LpDistance = l2) -> float:
    # Compute the distance between two images.
    return distance(x_ori, x_pert).item()
