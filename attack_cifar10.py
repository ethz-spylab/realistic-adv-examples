from accelerate import Accelerator
import fire
import timm
import torch
from timm.utils import unwrap_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from src import attacks
from src import models  # pylint: disable=unused-import
from src import utils


def main(model="resnet20_cifar10", data_dir="~/data", attack_name="exact_score_est"):
    model = timm.create_model(model, pretrained=True)

    data_dir = "~/data"
    default_cfg = unwrap_model(model).default_cfg
    model = utils.normalize_model(model, mean=default_cfg["mean"], std=default_cfg["std"])
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    targets_dl = DataLoader(ds, batch_size=1, shuffle=True)

    distance_fn = lambda x1, x2: (x1 - x2).norm(p=2)
    loss_fn = attacks.CWAdvLoss()
    attack_steps = 10000
    step_size = 0.005
    conf_factor = 0.005
    dist_factor = 1
    n_points = 200
    smooth_radius = 0.005

    estimation_steps_factor = 2

    accelerator = Accelerator(split_batches=True)
    model, dl, targets_dl, loss_fn = accelerator.prepare(model, dl, targets_dl, loss_fn)
    assert isinstance(dl, DataLoader)
    assert isinstance(targets_dl, DataLoader)
    assert isinstance(model, nn.Module)
    assert isinstance(loss_fn, nn.Module)

    attack = lambda x, x_target, y_target: attacks.score_based_attack_exact_loss(
        x, x_target, y_target, model, distance_fn, attack_steps * estimation_steps_factor, step_size /
        estimation_steps_factor, conf_factor, dist_factor, loss_fn, n_points, smooth_radius)

    attack_exact = lambda x, x_target, y_target: attacks.gradient_based_attack(
        x, x_target, y_target, model, distance_fn, attack_steps, step_size, conf_factor, dist_factor, loss_fn)

    for i, ((x, y), (x_target, y_target)) in enumerate(zip(dl, targets_dl)):
        if y.item() == y_target.item():
            print("Skipping pair as the class is the same for both inputs")
            continue
        elif model(x).argmax().item() != y.item():
            print("Skipping pair as the first element is already misclassified")
            continue
        elif model(x_target).argmax().item() != y_target.item():
            print("Skipping pair as the second element is already misclassified")
            continue
        original_distance = distance_fn(x, x_target)
        print(f"original_distance = {original_distance.item()}")
        x_adv_exact, steps = attack_exact(x, x_target, y_target)
        distance_exact = distance_fn(x, x_adv_exact)
        print(f"distance_exact = {distance_exact.item()}, steps = {steps}")
        x_adv_bb, steps = attack(x, x_target, y_target)
        distance_bb = distance_fn(x, x_adv_bb)
        print(f"distance_bb = {distance_bb.item()}, steps = {steps}")
        utils.show_grid([x[0], x_target[0], x_adv_exact[0], x_adv_bb[0]], filename=f"imgs/{i}.jpg")


if __name__ == "__main__":
    fire.Fire(main)
