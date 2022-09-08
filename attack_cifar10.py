from typing import Tuple
from accelerate import Accelerator
from pydantic_cli import run_and_exit
import timm
from timm.utils import unwrap_model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from src import attacks
from src import models  # pyright: ignore [reportUnusedImport]
from src import utils


def main(settings: utils.AttackSettings) -> int:
    model = timm.create_model(settings.model, pretrained=True)

    data_dir = "~/data"
    default_cfg = unwrap_model(model).default_cfg
    model: nn.Module = utils.normalize_model(model, mean=default_cfg["mean"], std=default_cfg["std"])
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)
    dl: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(ds, batch_size=1, shuffle=True)

    distance_fn: attacks.DistanceFn = lambda x1, x2: (x1 - x2).norm(p=2)  # type: ignore
    loss_fn: nn.Module = attacks.CWAdvLoss(targeted=settings.targeted)

    estimation_steps_factor = 2

    accelerator = Accelerator(split_batches=True)
    model, dl, loss_fn = accelerator.prepare(model, dl, loss_fn)  # type: ignore

    attack: attacks.AttackFn = lambda x, y, x_target, y_target: attacks.score_based_attack_exact_loss(
        x, y, x_target, y_target, model, settings.eps, distance_fn, steps * estimation_steps_factor, settings.step_size
        / estimation_steps_factor, settings.c * estimation_steps_factor, loss_fn, settings.n_points, settings.
        random_radius, settings.targeted)

    attack_exact: attacks.AttackFn = lambda x, y, x_target, y_target: attacks.gradient_based_attack(
        x, y, x_target, y_target, model, settings.eps, distance_fn, steps, settings.step_size, settings.c, loss_fn,
        settings.targeted)

    for i, (x, y) in enumerate(dl):
        if model(x).argmax().item() != y.item():
            print("Skipping as the first element is already misclassified")
            continue
        x_target, y_target = utils.init_targets(model, x, y)

        original_distance = distance_fn(x, x_target)
        original_confidence = loss_fn(model(x_target), y_target)
        print(f"original_distance = {original_distance.item()}, original_confidece = {original_confidence.item()}")

        x_adv_exact, steps = attack_exact(x, y, x_target, y_target)
        distance_exact = distance_fn(x, x_adv_exact)
        print(f"distance_exact = {distance_exact.item()}, steps = {steps}")

        x_adv_bb, steps = attack(x, y, x_target, y_target)
        distance_bb = distance_fn(x, x_adv_bb)
        print(f"distance_bb = {distance_bb.item()}, steps = {steps}")

        utils.show_grid([x[0], x_target[0], x_adv_exact[0], x_adv_bb[0]], filename=f"imgs/{i}.jpg")

    return 0


if __name__ == "__main__":
    run_and_exit(utils.AttackSettings, main, description="Attack CIFAR-10", version='0.1.0')
