import datetime
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F
from pydantic import BaseSettings, Field


class ImageNormalizer(nn.Module):
    """From
    https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py#L8"""
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: tuple[float, float, float], std: tuple[float, float, float]) -> nn.Module:
    """From
    https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py#L20"""
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)), ('model', model)])
    return nn.Sequential(layers)


def show_grid(xs: Iterable[torch.Tensor],
              ncols: int = 4,
              cmap: str | None = None,
              labels: list[str] | None = None,
              filename: Path | None = None,
              axes_pad: float = 1.5) -> None:
    xs_np = [np.asarray(F.to_pil_image(x)) for x in xs]
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(len(xs_np) // ncols, ncols),  # creates 2x2 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
    )

    for i, (ax, im) in enumerate(zip(grid, xs_np)):  # type: ignore
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap)
        if labels is not None:
            ax.set_title(labels[i], fontdict={'fontsize': 20}, pad=20)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if filename is not None:
        fig.savefig(str(filename))


def init_targets(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_target = torch.rand_like(x)
    while (y_target := (model(x_target).argmax(-1))) and (y_target == y).any():
        x_target = torch.rand_like(x)
    return x_target, y_target


class AttackSettings(BaseSettings):
    model: str = Field(default="resnet20_cifar10", description="The model to attack")
    data_dir: str = Field(default="~/data", description="The directory of the data")
    targeted: bool = Field(default=False, description="Whether the attacks should be targeted")
    steps: int = Field(default=10_000, description="Attack steps")
    step_size: float = Field(default=0.005, description="Step size")
    c: float = Field(default=0.01, description="The `c` factor for the confidence loss")
    n_points: int = Field(default=200, description="The number of points to use for gradient estimation")
    random_radius: float = Field(default=0.005,
                                 description="The radius of the random points for the gradient estimation")
    eps: float = Field(default=0.5, description="The epsilon of the attack")
    log_dir: str = Field(default="logs", description="Where to log stuff")
    log_images: bool = Field(default=False, description="Whether to log images to disk")


def init_attack_run(settings: AttackSettings) -> Path:
    base_dir = Path(settings.log_dir)
    run_dir_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_dir = base_dir / run_dir_name
    experiment_dir.mkdir(parents=True)
    with open(experiment_dir / "config.json", "w") as f:
        f.write(settings.json(sort_keys=True, indent=4))
        f.write("\n")
    return experiment_dir
