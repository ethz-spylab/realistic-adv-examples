from pathlib import Path
from typing import Any

import numpy as np
import requests
import tqdm
from torchvision.datasets import ImageNet

from src.utils import sha256sum


class ImageNetNSFW(ImageNet):

    _BASE_FILENAME = "nsfw_imagenet_{split}_outputs.npz"
    _ROOT_FILENAME = "nsfw_imagenet"
    _BASE_URL = "https://github.com/ethz-privsec/realistic-adv-examples/releases/download/{version}/{filename}"
    _VERSION = "v0.1"
    _DOWNLOAD_URL = _BASE_URL.format(version=_VERSION, filename=_BASE_FILENAME)

    _SHA256_HASHES = {
        "val": "9573287aaf5b5788ec9213edcafe5e2fd37099005cc5b63f0bc9b84d689416aa",
        "train": "d2414f72caa1fb40b8a4f0b1fdaac30a8f480d6c429e31623443f37177f1a57c"
    }

    def __init__(self,
                 root: str,
                 top_k: int | None,
                 threshold: float | None = None,
                 split: str = "train",
                 only_positives: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)
        self.nsfw_outputs_path = Path(root).parent / self._ROOT_FILENAME / self._BASE_FILENAME.format(split=split)
        if not self.nsfw_outputs_path.parent.exists():
            self.nsfw_outputs_path.parent.mkdir(parents=True)

        if not self.nsfw_outputs_path.exists():
            self._download_outputs()

        if sha256sum(self.nsfw_outputs_path) != self._SHA256_HASHES[split]:
            raise ValueError("The SHA256 hash of downloaded file does not match expected hash.")

        with np.load(self.nsfw_outputs_path, allow_pickle=False) as f:
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
        imgs_to_keep = list(map(lambda x: (x[0], 1), imgs_to_keep))

        if not only_positives:
            if top_k is not None:
                positives = imgs[top_k:]
            else:
                positives = np.array(imgs)[nsfw_outputs < threshold].tolist()
            imgs_to_keep += list(map(lambda x: (x[0], 0), positives))

        full_path_imgs_to_keep = list(map(lambda x: (str(Path(self.root) / split / x[0]), x[1]), imgs_to_keep))

        self.samples = full_path_imgs_to_keep
        self.imgs = self.samples

    def _download_outputs(self) -> None:
        download_url = self._DOWNLOAD_URL.format(split=self.split)
        print(f"Downloading NSFW outputs from {download_url}")

        response = requests.get(download_url, stream=True)
        with self.nsfw_outputs_path.open("wb") as f:
            for data in tqdm.tqdm(response.iter_content()):
                f.write(data)
