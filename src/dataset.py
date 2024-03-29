import random
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import ImageNet
from transformers import CLIPProcessor

from src.imagenet_nsfw import ImageNetNSFW


def load_mnist_test_data(test_batch_size=1) -> data.DataLoader:
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader


def load_cifar10_test_data(test_batch_size=1) -> data.DataLoader:
    # CIFAR10 Dataset
    test_dataset = dsets.CIFAR10('/data', download=True, train=False, transform=transforms.ToTensor())
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader


def load_imagenet_test_data(test_batch_size=1, folder='/data/imagenet/val') -> data.DataLoader:
    val_dataset = dsets.ImageFolder(
        folder, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    rand_seed = 42

    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    val_loader = data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)

    return val_loader


class BinaryImageNet(ImageNet):
    DOG_LABELS = set(range(151, 269))  # It's dogs from 151 (Chihuaha) to 268 (Mexican hairless)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = super().__getitem__(index)
        binary_label = 1 if label in self.DOG_LABELS else 0
        return image, binary_label


def load_binary_imagenet_test_data(test_batch_size=1, data_dir=Path("/data/imagenet")) -> data.DataLoader:
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    val_dataset = BinaryImageNet(root=str(data_dir), split="val", transform=transform)

    rand_seed = 42
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    val_loader: data.DataLoader = data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)

    return val_loader


def load_imagenet_nsfw_test_data(test_batch_size=1, data_dir=Path("/data/imagenet")) -> data.DataLoader:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    im_mean = torch.tensor(processor.feature_extractor.image_mean).view(3, 1, 1)  # type: ignore
    im_std = torch.tensor(processor.feature_extractor.image_std).view(3, 1, 1)  # type: ignore

    def transform(x):
        preprocessed_images = processor(images=x, return_tensors="pt", padding=True)["pixel_values"][0]  # type: ignore
        unnormalized_images = torch.round((preprocessed_images * im_std + im_mean) * 255) / 255
        return unnormalized_images

    val_dataset = ImageNetNSFW(str(data_dir), top_k=1000, split="train", transform=transform)

    rand_seed = 42
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    val_loader = data.DataLoader(val_dataset, batch_size=test_batch_size)  # type: ignore

    return val_loader
