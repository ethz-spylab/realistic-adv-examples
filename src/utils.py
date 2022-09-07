from collections import OrderedDict
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F


class ImageNormalizer(nn.Module):
    """From
    https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py#L8"""
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    """From
    https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py#L20"""
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)), ('model', model)])
    return nn.Sequential(layers)


def show_grid(xs, ncols=4, cmap=None, labels=None, filename=None, axes_pad=1.5):
    xs = [np.asarray(F.to_pil_image(x)) for x in xs]
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(len(xs) // ncols, ncols),  # creates 2x2 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
    )

    for i, (ax, im) in enumerate(zip(grid, xs)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap)
        if labels is not None:
            ax.set_title(labels[i], fontdict={'fontsize': 20}, pad=20)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if filename is not None:
        fig.savefig(filename)
