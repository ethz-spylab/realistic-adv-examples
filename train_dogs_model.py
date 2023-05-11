from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import BinaryImageNet
from src.arch.binary_resnet50 import BinaryResNet50


def main(args):
    img_size = 224

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    train_ds = BinaryImageNet(args.data_dir, transform=train_transform)
    train_dl = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=8)
    val_ds = BinaryImageNet(args.data_dir,
                            split="val",
                            transform=val_transform)
    val_dl = DataLoader(val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=8)

    model = BinaryResNet50()

    # training
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=10)
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="/data/imagenet")
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)