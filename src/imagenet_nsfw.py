from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import ImageNet


class ImageNetNSFW(ImageNet):

    def __init__(self,
                 root: str,
                 nsfw_outputs_path: str | Path,
                 top_k: int | None,
                 threshold: float | None = None,
                 split: str = "train",
                 **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)
        nsfw_outputs_path = Path(nsfw_outputs_path)
        with np.load(nsfw_outputs_path, allow_pickle=False) as f:
            nsfw_outputs = f["outputs"]
            imgs = f["imgs"]
        assert len(nsfw_outputs) == len(imgs)
        assert len(nsfw_outputs) == len(self.samples)
        if top_k is not None:
            assert threshold is None, "Cannot specify both top_k and threshold"
            imgs_to_keep = imgs[:top_k]
        else:
            assert threshold is not None, "Must specify either top_k or threshold"
            imgs_to_keep = np.array(imgs)[nsfw_outputs >= threshold].tolist()
        full_path_imgs_to_keep = list(map(lambda x: (str(Path(self.root) / split / x[0]), x[1]), imgs_to_keep))
        self.samples = full_path_imgs_to_keep
        self.imgs = self.samples
