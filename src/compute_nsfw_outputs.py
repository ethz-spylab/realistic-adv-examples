from pathlib import Path

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from transformers import CLIPProcessor

from arch.clip_laion_nsfw import CLIPNSFWDetector


def get_filename(path: str, data_dir: str, split: str) -> str:
    base_path = Path(data_dir) / split
    return Path(path).relative_to(base_path).as_posix()


@torch.no_grad()
def compute_outputs(model: nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    outputs = torch.tensor([], device=device)
    for x, _ in tqdm.tqdm(dataloader):
        y = model(x.to(device)).flatten()
        outputs = torch.cat((outputs, y))
    return outputs


def main(args):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def transform(batch):
        preprocessed_images = processor(images=batch, return_tensors="pt", padding=True)["pixel_values"][0]  # type: ignore
        return preprocessed_images
    
    ds = ImageNet(root=args.data_dir, split=args.split, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=32)
    
    device = torch.device(args.device)
    model = CLIPNSFWDetector().to(device)
    outputs = compute_outputs(model, dl, device).detach().cpu().numpy()
    sorted_indices = np.argsort(-outputs)  # - to sort in descending order
    sorted_outputs = outputs[sorted_indices]
    imgs = list(map(lambda x: (get_filename(x[0], args.data_dir, args.split), x[1]), ds.imgs))
    imgs = np.array(imgs)[sorted_indices].tolist()
    np.savez(args.output, outputs=sorted_outputs, imgs=imgs)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data/imagenet"))
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)