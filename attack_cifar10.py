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
    utils.random_seed()

    base_dir, logger = utils.init_attack_run(settings)
    print(f"Saving logs to {base_dir}")

    data_dir = "~/data"
    default_cfg = unwrap_model(model).default_cfg
    model: nn.Module = utils.normalize_model(model, mean=default_cfg["mean"], std=default_cfg["std"])
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)
    dl: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(ds, batch_size=1)
    clean_dl: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(ds, batch_size=1024)

    distance_fn: attacks.DistanceFn = lambda x1, x2: (x1 - x2).norm(p=2)  # type: ignore
    loss_fn: nn.Module = attacks.CWAdvLoss(targeted=settings.targeted)

    accelerator = Accelerator(split_batches=True)
    model, dl, clean_dl, loss_fn = accelerator.prepare(model, dl, clean_dl, loss_fn)  # type: ignore

    nes_attack: attacks.AttackFn = lambda x, y, x_target, y_target: attacks.score_based_attack_exact_loss(
        x, y, x_target, y_target, model, settings.eps, distance_fn, settings.steps * settings.est_settings_factor,
        settings.step_size / settings.est_settings_factor, settings.c * settings.est_settings_factor, loss_fn, settings.
        n_points, settings.random_radius, settings.targeted)

    wb_attack: attacks.AttackFn = lambda x, y, x_target, y_target: attacks.gradient_based_attack(
        x, y, x_target, y_target, model, settings.eps, distance_fn, settings.steps, settings.step_size, settings.c,
        loss_fn, settings.targeted)

    n_wb_successes = 0
    n_nes_successes = 0
    n_original_misclassified = 0

    clean_accuracy = utils.compute_clean_accuracy(model, clean_dl)
    print(f"Clean accuracy: {clean_accuracy * 100}%")

    for i, (x, y) in enumerate(dl):
        if model(x).argmax().item() != y.item():
            logger.write(utils.AttackResult(i, True))
            n_original_misclassified += 1
            continue
        x_target, y_target = utils.init_targets(model, x, y)

        original_distance = distance_fn(x, x_target)
        if settings.targeted:
            original_margin = loss_fn(model(x_target), y_target)
        else:
            original_margin = loss_fn(model(x_target), y)
        print(f"original_distance = {original_distance.item()}, original_margin = {original_margin.item()}")

        def run_attack(attack: attacks.AttackFn, attack_name: str) -> tuple[bool, torch.Tensor]:
            x_adv, steps = attack(x, y, x_target, y_target)
            distance = distance_fn(x, x_adv)
            success = distance.item() < settings.eps
            logger.write(utils.AttackResult(i, False, distance.item(), success, steps, attack_name))
            return success, x_adv

        # Run WB attack
        if "wb" in settings.attacks:
            wb_success, wb_x_adv = run_attack(wb_attack, "wb")
            n_wb_successes += int(wb_success)
        else:
            wb_x_adv = None

        # Run NES attack
        if "nes" in settings.attacks:
            nes_success, nes_x_adv = run_attack(nes_attack, "nes")
            n_nes_successes += int(nes_success)
        else:
            nes_x_adv = None

        # Log overall results
        n_actual_attacks = i + 1 - n_original_misclassified
        print(
            f"successes WB = {n_wb_successes}/{n_actual_attacks} - successes NES = {n_nes_successes}/{n_actual_attacks}"
        )
        if settings.log_images and wb_x_adv is not None and nes_x_adv is not None:
            utils.show_grid([x[0], x_target[0], wb_x_adv[0], nes_x_adv[0]], filename=base_dir / f"{i}.jpg")

    return 0


if __name__ == "__main__":
    run_and_exit(utils.AttackSettings, main, description="Attack CIFAR-10", version='0.1.0')
