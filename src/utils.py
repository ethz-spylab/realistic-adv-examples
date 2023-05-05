import hashlib
from pathlib import Path

import torch
from foolbox.distances import LpDistance, l2


def compute_distance(x_ori: torch.Tensor, x_pert: torch.Tensor, distance: LpDistance = l2) -> torch.Tensor:
    # Compute the distance between two images.
    return distance(x_ori, x_pert)


def sha256sum(filename: Path) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def write_sha256sum(src: Path, dst: Path):
    with open(dst, "w") as f:
        f.write(sha256sum(src))


def read_sha256sum(path: Path) -> str:
    with open(path, "r") as f:
        return f.read().strip()
